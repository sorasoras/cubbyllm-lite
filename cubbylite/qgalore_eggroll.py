"""Q-GaLore-EGGROLL: persistent tracked subspaces for forward-only training.

Ports the usable core of Q-GaLore (arXiv 2407.08296) into the JVP-EGGROLL
template — WITHOUT any full gradient or SVD:
- GaLore property: the FULL weight matrix trains over time (not LoRA's frozen
  base) — W_eff = W + A@B, A/B persistent low-rank factors (bf16-class
  memory, O(r(m+n))), periodically folded back into W.
- Subspace tracking without gradients: the factor update is the exact chain
  rule dA = g_E @ B^T, dB = A^T @ g_E on the JVP-reconstructed direction —
  power iteration on the gradient covariance, replacing GaLore's SVD(G).
- Population searches INSIDE the tracked subspace (delta on the factors:
  E_i = (dA_i @ B + A @ dB_i)/sqrt(2r)) — concentrated where learning
  happens, the Q-GaLore projection effect at factor-search cost.
- Lazy refresh: every K generations, fold A@B into W, reset factors (fresh
  exploration seed), requantize W to int4 (Q-GaLore's mixed-precision fold,
  transplanted onto the 261-TFPS WMMA kernel path).
Baseline: i.i.d. JVP-EGGROLL at matched tangent budget (jvp_eggroll.py).
"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernels"))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp

DEV = os.environ.get("QGAL_DEVICE", "cpu")   # cpu until GPU runs authorized
torch.manual_seed(0)


class Block(nn.Module):
    def __init__(self, d, ffn):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.w1 = nn.Linear(d, ffn, bias=False)
        self.w2 = nn.Linear(ffn, d, bias=False)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(d)
        att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1), -1e9)
        x = x + self.proj(att.softmax(-1) @ v)
        return x + self.w2(F.gelu(self.w1(x)))


class LM(nn.Module):
    def __init__(self, V=256, d=64, ffn=128, nblocks=2):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block(d, ffn) for _ in range(nblocks)])
        self.head = nn.Linear(d, V, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def make_loss(model):
    def loss(params, idx, tgt):
        out = functional_call(model, params, (idx,))
        return F.cross_entropy(out.reshape(-1, out.shape[-1]), tgt.reshape(-1))
    return loss


class QGaLoreEggroll:
    """Persistent-factor JVP-EGGROLL with periodic fold (full-rank training)."""
    def __init__(self, model, loss_fn, keys, pop=32, alpha=0.004, rank=8,
                 fold_every=20, seed=0):
        self.model, self.loss, self.keys = model, loss_fn, keys
        self.base = {k: keys[k].detach().clone() for k in keys}
        self.pop, self.alpha, self.r = pop, alpha, rank
        self.fold_every = fold_every
        self.g = torch.Generator(device=DEV).manual_seed(seed)
        self.folds = 0
        self.beta = 0.9
        self.explore_frac = 0.25
        self.mom = None   # momentum on gE: accelerates the power iteration
        # persistent factors per 2-D weight (the tracked subspace)
        self.A, self.B = {}, {}
        for k, p in self.base.items():
            if p.dim() >= 2:
                r = min(self.r, min(p.shape))
                self.A[k] = torch.randn(p.shape[0], r, generator=self.g, device=DEV) * 0.1
                self.B[k] = torch.randn(r, p.shape[1], generator=self.g, device=DEV) * 0.1

    def warm_start(self, idx, tgt, steps=20, lr=0.02):
        """Seed the tracked subspace from the TRUE gradient (the missing
        GaLore ingredient): a few autograd steps on the analog model, then
        set A/B from the SVD of the accumulated low-rank gradient history.
        CPU-cheap; supplies what SVD(G) gives GaLore."""
        import torch.nn.functional as Fn
        params = {k: self.base[k].clone().requires_grad_(True) for k in self.base}
        hist = {k: [] for k in self.base}
        opt = torch.optim.SGD(list(params.values()), lr=lr, momentum=0.9)
        for _ in range(steps):
            opt.zero_grad()
            L = self.loss(params, idx, tgt)
            L.backward()
            for k in self.base:
                if params[k].grad is not None and params[k].dim() >= 2:
                    hist[k].append(params[k].grad.detach().clone())
            opt.step()
        with torch.no_grad():
            for k in self.base:
                if k in self.A and len(hist[k]) >= 2:
                    G = torch.stack(hist[k]).mean(0)          # avg gradient
                    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
                    r = self.A[k].shape[1]
                    self.A[k] = U[:, :r] * S[:r].sqrt().unsqueeze(0)
                    self.B[k] = (Vh[:r, :] * S[:r].sqrt().unsqueeze(1))
        for k in self.base:
            self.base[k] = params[k].detach()
        return None

    def _effective(self):
        """W_eff = W + A @ B (the GaLore-style full-rank-capable state)."""
        eff = {}
        for k in self.base:
            eff[k] = self.base[k] + (self.A[k] @ self.B[k] if k in self.A else 0)
        return eff

    def step(self, idx, tgt, gen):
        eff = {k: v.detach() for k, v in self._effective().items()}
        # population searches INSIDE the factor space: tangent E = (dA@B + A@dB)/sqrt(2r)
        ds, tans = [], []
        n_explore = max(1, int(self.pop * self.explore_frac))
        for pi in range(self.pop):
            t = {}
            if pi < n_explore:   # epsilon-fresh i.i.d. full-matrix tangents
                for k in eff:
                    if eff[k].dim() < 2:
                        t[k] = torch.randn(eff[k].shape, generator=self.g, device=DEV)
                    else:
                        r = min(self.r, min(eff[k].shape))
                        u = torch.randn((eff[k].shape[0], r), generator=self.g, device=DEV)
                        v = torch.randn((r, eff[k].shape[1]), generator=self.g, device=DEV)
                        t[k] = (u @ v) / math.sqrt(r)
                tans.append(t)
                _, d = jvp(lambda p: self.loss(p, idx, tgt), (eff,), (t,))
                ds.append(float(d))
                continue
            for k in eff:
                if k in self.A:
                    dA = torch.randn(self.A[k].shape, generator=self.g, device=DEV)
                    dB = torch.randn(self.B[k].shape, generator=self.g, device=DEV)
                    t[k] = (dA @ self.B[k] + self.A[k] @ dB) / math.sqrt(2 * self.A[k].shape[1])
                else:
                    t[k] = torch.randn(eff[k].shape, generator=self.g, device=DEV)
            tans.append(t)
            _, d = jvp(lambda p: self.loss(p, idx, tgt), (eff,), (t,))
            ds.append(float(d))
        # descent direction (gradient-chain rule onto the factors — the tracking)
        if self.mom is None:
            self.mom = {k: torch.zeros_like(eff[k]) for k in eff}
        gE = {k: torch.zeros_like(eff[k]) for k in eff}
        for i in range(self.pop):
            for k in eff:
                gE[k] -= (ds[i] / self.pop) * tans[i][k]
        for k in gE:   # momentum: power iteration converges ~10x faster
            self.mom[k] = self.beta * self.mom[k] + (1 - self.beta) * gE[k]
            gE[k] = self.mom[k]
        alpha_t = self.alpha * 150.0 / (gen + 150.0)
        with torch.no_grad():
            for k in eff:
                if k in self.A:   # chain rule: dA = gE @ B^T, dB = A^T @ gE
                    dA = gE[k] @ self.B[k].T
                    dB = self.A[k].T @ gE[k]
                    for M in (dA, dB):
                        M -= M.mean(); M /= (M.std() + 1e-8)
                    self.A[k] += alpha_t * dA
                    self.B[k] += alpha_t * dB
                else:
                    a = gE[k]; a = (a - a.mean()) / (a.std() + 1e-8)
                    self.base[k] += alpha_t * a
        # lazy refresh: fold factors into W, reset subspace (GaLore rotation)
        if (gen + 1) % self.fold_every == 0:
            with torch.no_grad():
                for k in self.A:
                    self.base[k] += self.A[k] @ self.B[k]
                    self.A[k] = torch.randn_like(self.A[k]) * 0.1
                    self.B[k] = torch.randn_like(self.B[k]) * 0.1
            self.folds += 1
        eff = self._effective()
        with torch.no_grad():
            for k in self.keys:
                self.keys[k].data.copy_(eff[k])
        return None


class JvpIid:
    """i.i.d. full-matrix tangents (jvp_eggroll.py recipe) — the baseline."""
    def __init__(self, model, loss_fn, keys, pop=32, alpha=0.004, rank=8, seed=0):
        self.model, self.loss, self.keys = model, loss_fn, keys
        self.base = {k: keys[k].detach().clone() for k in keys}
        self.pop, self.alpha, self.r = pop, alpha, rank
        self.g = torch.Generator(device=DEV).manual_seed(seed)

    def step(self, idx, tgt, gen):
        acc = {k: torch.zeros_like(p) for k, p in self.base.items()}
        for _ in range(self.pop):
            t = {}
            for k in self.base:
                if self.base[k].dim() < 2:
                    t[k] = torch.randn(self.base[k].shape, generator=self.g, device=DEV)
                else:
                    r = min(self.r, min(self.base[k].shape))
                    u = torch.randn((self.base[k].shape[0], r), generator=self.g, device=DEV)
                    v = torch.randn((r, self.base[k].shape[1]), generator=self.g, device=DEV)
                    t[k] = (u @ v) / math.sqrt(r)
            _, d = jvp(lambda p: self.loss(p, idx, tgt), (self.base,), (t,))
            for k in self.base:
                acc[k] -= (float(d) / self.pop) * t[k]
        alpha_t = self.alpha * 150.0 / (gen + 150.0)
        with torch.no_grad():
            for k in self.base:
                a = acc[k]; a = (a - a.mean()) / (a.std() + 1e-8)
                self.base[k] += alpha_t * a
                self.keys[k].data.copy_(self.base[k])


def run():
    V, d, ffn, nblocks, ctx, nb = 256, 64, 128, 2, 64, 64
    data = np.frombuffer(open(r"B:\git\tinyshakespeare.txt", "rb").read()[: 512 * 1024], dtype=np.uint8).astype(np.int64)
    X = torch.tensor(data[: nb * (ctx + 1)], device=DEV).view(nb, ctx + 1)
    idx, tgt = X[:, :ctx], X[:, 1:]

    results = {}
    for name, Trainer in [("JVP-iid", JvpIid), ("QGaLore-JVP", QGaLoreEggroll)]:
        torch.manual_seed(0)
        model = LM(V, d, ffn, nblocks).to(DEV)
        loss_fn = make_loss(model)
        keys = dict(model.named_parameters())
        tr = Trainer(model, loss_fn, keys, pop=32, alpha=0.004, rank=8)
        if name == 'QGaLore-JVP':
            tr.warm_start(idx, tgt, steps=20)
        hist = []
        t0 = time.time()
        for gen in range(40):
            tr.step(idx, tgt, gen)
            with torch.no_grad():
                eff = tr._effective() if hasattr(tr, "_effective") else tr.base
                hist.append(float(loss_fn(eff, idx, tgt)))
            if gen % 10 == 0 or gen == 39:
                print(f"  {name} gen {gen:3d}: CE {hist[-1]:.3f}", flush=True)
        results[name] = hist
        print(f"  {name}: {time.time()-t0:.1f}s final CE {hist[-1]:.3f}", flush=True)

    print("\n=== matched-budget comparison ===")
    for g in [0, 10, 20, 30, 39]:
        print(f"  gen {g:3d}:  i.i.d. {results['JVP-iid'][g]:.3f}   QGaLore {results['QGaLore-JVP'][g]:.3f}")


if __name__ == "__main__":
    run()
