"""JVP-EGGROLL: experimental forward-only pretraining harness.

Combines three validated pieces of this project stack:
1. EGGROLL (arXiv 2605.30361): rank-r factorized perturbations, O(r(m+n))
   memory per generation.
2. JVP upgrade: instead of noisy antithetic finite differences (2 perturbed
   forwards per member, sigma-smoothed gradient), torch.func.jvp computes the
   EXACT directional derivative dL/dE_i through the analog teacher pass in one
   dual forward. No sigma, no evaluation noise, no antithetic cancellation.
   Update: g = (1/P) sum_i d_i * E_i   (unbiased for isotropic E_i).
3. INT4 FWD: the spike-pass evaluation runs the trained FFN weights through
   the gfx1201 int4xint4 WMMA kernel (kernels/gemm_v19.py machinery,
   261 TFLOPS class) — ternary spike activations are exactly int4, so the
   quantized-model loss is the deployment-format loss.

Training signal = analog pass (differentiable teacher, SpikeLM philosophy).
Evaluation = spike + int4 kernel forward (deployment format).
Baseline = FD-EGGROLL (classic antithetic finite differences, same update rule)
at matched forward-compute budget.
"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernels"))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp

DEV = "cuda"
torch.manual_seed(0)


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, ffn):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.w1 = nn.Linear(d, ffn, bias=False)     # FFN through int4 kernel at eval
        self.w2 = nn.Linear(ffn, d, bias=False)

    def forward(self, x):                            # analog pass (fully differentiable)
        B, T, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(d, dim=-1)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(d)
        att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1), -1e9)
        x = x + self.proj(att.softmax(-1) @ v)
        h = self.w2(F.gelu(self.w1(x)))
        return x + h


class AnalogLM(nn.Module):
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


# ------------------------------------------------------- JVP-EGGROLL trainer
class JvpEggroll:
    """Exact directional derivatives via forward-mode AD on rank-r tangents."""
    def __init__(self, model, loss_fn, keys, pop=32, alpha=0.05, rank=8, seed=0):
        self.model, self.loss, self.keys = model, loss_fn, keys
        self.base = {k: keys[k].detach().clone() for k in keys}
        self.pop, self.alpha, self.rank = pop, alpha, rank
        self.g = torch.Generator(device=DEV).manual_seed(seed)

    def _tangent(self):
        d = {}
        for k, p in self.base.items():
            if p.dim() < 2:
                d[k] = torch.randn(p.shape, generator=self.g, device=DEV)
            else:
                r = min(self.rank, min(p.shape))
                u = torch.randn((p.shape[0], r), generator=self.g, device=DEV)
                v = torch.randn((r, p.shape[1]), generator=self.g, device=DEV)
                d[k] = (u @ v) / math.sqrt(r)
        return d

    def step(self, idx, tgt, gen):
        acc = {k: torch.zeros_like(p) for k, p in self.base.items()}
        for _ in range(self.pop):
            E = self._tangent()
            _, d = jvp(lambda p: self.loss(p, idx, tgt), (self.base,), (E,))
            for k in self.keys:
                acc[k] -= (float(d) / self.pop) * E[k]
        alpha_t = self.alpha * 150.0 / (gen + 150.0)
        with torch.no_grad():
            for k in self.base:
                a = acc[k]
                a = (a - a.mean()) / (a.std() + 1e-8)     # scale-free step (validated)
                self.base[k] += alpha_t * a
                self.keys[k].data.copy_(self.base[k])
        return None


class FdEggroll:
    """Classic antithetic finite-difference EGGROLL at the same compute budget."""
    def __init__(self, model, loss_fn, keys, pop=64, sigma=0.05, alpha=0.05, rank=8, seed=0):
        self.model, self.loss, self.keys = model, loss_fn, keys
        self.base = {k: keys[k].detach().clone() for k in keys}
        self.pop, self.sigma, self.alpha, self.rank = pop, sigma, alpha, rank
        self.g = torch.Generator(device=DEV).manual_seed(seed)

    def _tangent(self):
        d = {}
        for k, p in self.base.items():
            if p.dim() < 2:
                d[k] = torch.randn(p.shape, generator=self.g, device=DEV)
            else:
                r = min(self.rank, min(p.shape))
                u = torch.randn((p.shape[0], r), generator=self.g, device=DEV)
                v = torch.randn((r, p.shape[1]), generator=self.g, device=DEV)
                d[k] = (u @ v) / math.sqrt(r)
        return d

    def step(self, idx, tgt, gen):
        half = self.pop // 2
        fs, dirs = [], []
        for _ in range(half):
            E = self._tangent()
            with torch.no_grad():
                for k in self.keys:
                    self.keys[k].data.copy_(self.base[k] + self.sigma * E[k])
            fs.append(-float(self.loss(self.keys, idx, tgt)))
            with torch.no_grad():
                for k in self.keys:
                    self.keys[k].data.copy_(self.base[k] - self.sigma * E[k])
            fs.append(-float(self.loss(self.keys, idx, tgt)))
            dirs.append(E)
        f = torch.tensor(fs)
        f = (f - f.mean()) / (f.std() + 1e-8)
        acc = {k: torch.zeros_like(p) for k, p in self.base.items()}
        for i in range(self.pop):
            s = 1.0 if i % 2 == 0 else -1.0
            for k in self.keys:
                acc[k] += s * float(f[i]) * dirs[i // 2][k] / self.pop
        alpha_t = self.alpha * 150.0 / (gen + 150.0)
        with torch.no_grad():
            for k in self.base:
                a = acc[k]
                a = (a - a.mean()) / (a.std() + 1e-8)
                self.base[k] += alpha_t * a
                self.keys[k].data.copy_(self.base[k])
        return None


# ------------------------------------------------------------- INT4 FWD eval
class Int4Fwd:
    """Spike-FFN forward through the gfx1201 int4 WMMA kernel (gemm_v19).

    Ternary spike activations pack exactly into int4; trained weights are
    symmetrically quantized to int4. Attention + embeddings stay in torch.
    """
    def __init__(self, model, V=256, d=64, ffn=128, nblocks=2, M=256, ctx=64):
        import gemm_v19 as G
        self.G = G
        self.fn = G.compile_src(G.SRC, "i4fwd")
        self.model = model
        self.M = M
        self.ctx = ctx

    def _gemm(self, W, Xn):  # W (N,K) torch fp, Xn (M,K) ternary int8 -> (M,N) fp
        G = self.G
        M, K = Xn.shape
        N = W.shape[0]
        # quantize W: symmetric int4
        s = 7.0 / W.abs().max().clamp(min=1e-8)
        Wq = torch.clamp((W / s).round(), -7, 7).to(torch.int32)
        Ap = G.pack(Xn).contiguous()                       # (M, K/8)
        Bt = G.pack_transposed(Wq).contiguous()            # (K/8, N)
        Out8 = torch.empty((M, N), device=DEV, dtype=torch.int8)
        seg = [(0, M, M)]
        te, tm, tn, ntiles = G.build_tiles(seg, N)
        scale = torch.full((1,), float(s), device=DEV)
        kw = K // 8
        G.launch_persistent(self.fn, Ap, Bt, te, tm, tn, scale, Out8, N, kw, ntiles, 84)
        return Out8.float()

    @torch.no_grad()
    def loss(self, idx, tgt):
        """Spike forward: LIF membranes ternarize activations; FFN GEMMs via int4 kernel."""
        model = self.model
        d = model.emb.weight.shape[1]
        x = model.emb(idx)                                  # (B,T,d) analog embedding
        B, T, _ = x.shape
        flat = x.reshape(-1, d)                              # (M,d)
        Mpad = ((flat.shape[0] + 255) // 256) * 256
        pad = Mpad - flat.shape[0]
        for blk in model.blocks:
            # attention (torch, analog)
            qkv = F.linear(flat[: flat.shape[0]], blk.qkv.weight)
            q, k, v = qkv.split(d, dim=-1)
            n = T
            att = (q.view(-1, n, d) @ k.view(-1, n, d).transpose(-1, -2)) / math.sqrt(d)
            att = att.masked_fill(torch.triu(torch.ones(n, n, device=DEV, dtype=torch.bool), 1), -1e9)
            o = (att.softmax(-1) @ v.view(-1, n, d)).reshape(-1, d)
            h = F.linear(o, blk.proj.weight) + flat[: o.shape[0]]
            # spike: ternarize membrane (LIF-lite)
            sp = torch.zeros_like(h)
            sp[h > 0.5] = 1.0
            sp[h < -0.5] = -1.0
            sp = sp.to(torch.int8)
            spp = torch.cat([sp, torch.zeros((pad, d), device=DEV, dtype=torch.int8)], 0)
            y = self._gemm(blk.w1.weight, spp)[: sp.shape[0]]       # int4 x int4
            # second layer: ternarize again
            sp2 = torch.zeros_like(y)
            sp2[y > 0.5] = 1.0
            sp2[y < -0.5] = -1.0
            sp2 = sp2.to(torch.int8)
            sp2p = torch.cat([sp2, torch.zeros((pad, y.shape[1]), device=DEV, dtype=torch.int8)], 0)
            z = self._gemm(blk.w2.weight, sp2p)[: sp2.shape[0]]
            flat = h + z
        out = F.linear(flat, model.head.weight)
        ce = F.cross_entropy(out[: tgt.reshape(-1).shape[0]], tgt.reshape(-1))
        return float(ce)


# ----------------------------------------------------------------- experiment
def load_tokens(path, ctx, n):
    data = np.frombuffer(open(path, "rb").read()[: 1024 * 512], dtype=np.uint8).astype(np.int64)
    return torch.tensor(data[: n * (ctx + 1)], device=DEV).view(n, ctx + 1)


def run():
    V, d, ffn, nblocks, ctx, nb = 256, 64, 128, 2, 64, 64
    path = r"B:\git\tinyshakespeare.txt"
    data = np.frombuffer(open(path, "rb").read()[: 512 * 1024], dtype=np.uint8).astype(np.int64)
    ntok = data.shape[0]
    X = torch.tensor(data[: nb * (ctx + 1)], device=DEV).view(nb, ctx + 1)

    results = {}
    for name, Trainer, kw in [
        ("FD-EGGROLL", FdEggroll, dict(pop=64, sigma=0.05, alpha=0.004)),
        ("JVP-EGGROLL", JvpEggroll, dict(pop=32, alpha=0.004)),      # ~same fwd budget (jvp ~2x)
    ]:
        torch.manual_seed(0)
        model = AnalogLM(V, d, ffn, nblocks).to(DEV)
        loss_fn = make_loss(model)
        keys = dict(model.named_parameters())
        tr = Trainer(model, loss_fn, keys, rank=8, **kw)
        hist = []
        t0 = time.time()
        for gen in range(40):
            idx = X[:, :ctx]
            tgt = X[:, 1:]
            tr.step(idx, tgt, gen)
            with torch.no_grad():
                ce = float(loss_fn(tr.base, idx, tgt))
            hist.append(ce)
            if gen % 10 == 0 or gen == 39:
                print(f"  {name} gen {gen:3d}: CE {ce:.3f}", flush=True)
        results[name] = hist
        if name == "JVP-EGGROLL":
            globals()["trained_model"] = model
        print(f"  {name}: {time.time()-t0:.1f}s, final CE {hist[-1]:.3f}", flush=True)

    # INT4 FWD evaluation of the JVP-TRAINED weights
    model = trained_model
    i4 = Int4Fwd(model, V, d, ffn, nblocks, M=256, ctx=ctx)
    idx, tgt = X[:, :ctx], X[:, 1:]
    with torch.no_grad():
        analog_ce = float(make_loss(model)(dict(model.named_parameters()), idx, tgt))
    try:
        q_ce = i4.loss(idx, tgt)
        print(f"INT4 FWD (spike FFN via WMMA kernel): quantized CE {q_ce:.3f} (analog ref {analog_ce:.3f}, "
              f"ln-scale gap {q_ce - analog_ce:.3f})", flush=True)
    except Exception as e:
        print(f"INT4 FWD eval issue: {type(e).__name__}: {e}", flush=True)

    print("\n=== fixed-budget comparison (loss per generation) ===")
    for g in [0, 10, 20, 30, 39]:
        print(f"  gen {g:3d}:  FD {results['FD-EGGROLL'][g]:.3f}   JVP {results['JVP-EGGROLL'][g]:.3f}")
    return results


if __name__ == "__main__":
    run()
