"""Anchored hybrid: periodic backprop anchors inside generation-scale Q-GaLore-JVP.

The K-curve experiment. One anchor every K generations:
  anchor = fold A@B into W -> one Adam step on W with the TRUE gradient G
         -> re-seed the tracked subspace from SVD(G) (the GaLore refresh).
Generations in between are the validated 4.767 recipe (pop=32, alpha=0.004,
rank=8, warm-start, epsilon-exploration, momentum). Everything runs inside
one matched wall-clock budget, against a pure-Adam backprop baseline.

Both phases write the same W_eff = W + A@B state: generations update the
factors A/B, anchors update W (after folding the factors in) and rotate the
subspace the generations search in.

Variants:
  selective_q : anchor backward on the top-q loss tokens only (sorted backprop)
  half_depth  : anchor backward only through the last block + head
  seed_scale  : gamma in A=U*sqrt(S)*gamma, B=sqrt(S)*gamma*V (A@B = gamma^2 G_r;
               gamma=1.0 is the verbatim warm-start seeding, small gamma is a
               pure direction hint)
"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

from qgalore_eggroll import LM, make_loss, QGaLoreEggroll, DEV


class AnchoredQGaLore(QGaLoreEggroll):
    def __init__(self, model, loss_fn, keys, pop=32, alpha=0.004, rank=8,
                 K=10, adam_lr=0.01, selective_q=None, half_depth=False,
                 seed_scale=0.02, fold_every=10 ** 9, seed=0):
        super().__init__(model, loss_fn, keys, pop=pop, alpha=alpha, rank=rank,
                         fold_every=fold_every, seed=seed)
        self.K, self.selective_q = K, selective_q
        self.half_depth, self.gamma = half_depth, seed_scale
        self.adam_lr = adam_lr
        self.anchors = 0
        self._rebind_trainable()

    def _rebind_trainable(self):
        """base as persistent Adam leaves (warm_start replaces self.base)."""
        self.base = {k: v.detach().clone().requires_grad_(True)
                     for k, v in self.base.items()}
        self.anchor_opt = torch.optim.Adam(list(self.base.values()), lr=self.adam_lr)

    def _anchor_loss(self, idx, tgt):
        params = self.base
        if self.half_depth:
            frozen = {k for k in self.base
                      if k.startswith('blocks.0.') or k == 'emb.weight'}
            params = {k: (v.detach() if k in frozen else v)
                      for k, v in self.base.items()}
        out = functional_call(self.model, params, (idx,))
        flat = out.reshape(-1, out.shape[-1])
        if self.selective_q:
            tok = F.cross_entropy(flat, tgt.reshape(-1), reduction='none')
            v, _ = torch.topk(tok, max(1, int(tok.numel() * self.selective_q)))
            return v.mean()
        return F.cross_entropy(flat, tgt.reshape(-1))

    def _svd_reseed(self, k, g):
        r = self.A[k].shape[1]
        # tiny matrices: CPU LAPACK SVD is ~30x faster than rocSOLVER here
        gc = g.detach().float().cpu()
        if not bool(torch.isfinite(gc).all()):
            self.A[k] = torch.randn_like(self.A[k]) * 0.1
            self.B[k] = torch.randn_like(self.B[k]) * 0.1
            return
        U, S, Vh = torch.linalg.svd(gc, full_matrices=False)
        U, S, Vh = U.to(g.device), S.to(g.device), Vh.to(g.device)
        self.A[k] = U[:, :r] * (S[:r].sqrt() * self.gamma).unsqueeze(0)
        self.B[k] = Vh[:r, :] * (S[:r].sqrt() * self.gamma).unsqueeze(1)

    def anchor(self, idx, tgt):
        with torch.no_grad():
            for k in self.A:                       # fold: factor state -> W
                self.base[k].data += self.A[k] @ self.B[k]
        self.anchor_opt.zero_grad(set_to_none=True)
        self._anchor_loss(idx, tgt).backward()
        gs = [v.grad for v in self.base.values() if v.grad is not None]
        finite = bool(torch.isfinite(torch.cat([g.reshape(-1) for g in gs]).all())) if gs else True
        with torch.no_grad():
            for k in self.A:
                g = self.base[k].grad
                if g is None or not finite:
                    self.A[k] = torch.randn_like(self.A[k]) * 0.1
                    self.B[k] = torch.randn_like(self.B[k]) * 0.1
                    continue
                self._svd_reseed(k, g)
        if finite:
            torch.nn.utils.clip_grad_norm_(
                [v for v in self.base.values() if v.grad is not None], 1.0)
            self.anchor_opt.step()
        self.anchors += 1
        self.mom = None    # fresh subspace: restart the power-iteration momentum

    def step(self, idx, tgt, gen):
        """QGaLoreEggroll.step with exploration tangents scaled to the
        factor-tangent magnitude — the anchor re-seeds A/B at small scale
        (A@B ~ gamma^2 G_r ~ 0), so unit-scale exploration tangents would
        otherwise dominate gE and drown the subspace search."""
        eff = {k: v.detach() for k, v in self._effective().items()}
        ds, tans = [], []
        n_explore = max(1, int(self.pop * self.explore_frac))
        for pi in range(self.pop):
            t = {}
            if pi < n_explore:
                for k in eff:
                    if eff[k].dim() < 2:
                        t[k] = torch.randn(eff[k].shape, generator=self.g, device=DEV)
                    else:
                        r = min(self.r, min(eff[k].shape))
                        u = torch.randn((eff[k].shape[0], r), generator=self.g, device=DEV)
                        v = torch.randn((r, eff[k].shape[1]), generator=self.g, device=DEV)
                        sc = self.A[k].abs().mean().clamp(min=1e-6) if k in self.A else 1.0
                        t[k] = (u @ v) * (sc / math.sqrt(r))
                tans.append(t)
                _, d = jvp(lambda p: self.loss(p, idx, tgt), (eff,), (t,))
                ds.append(d)          # 0-dim GPU tensor: no per-jvp sync
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
            ds.append(d)              # 0-dim GPU tensor: no per-jvp sync
        dst = torch.stack(ds)         # one tensor, no host round-trips
        if self.mom is None:
            self.mom = {k: torch.zeros_like(eff[k]) for k in eff}
        gE = {k: torch.zeros_like(eff[k]) for k in eff}
        for i in range(self.pop):
            for k in eff:
                gE[k] -= (dst[i] / self.pop) * tans[i][k]
        for k in gE:
            self.mom[k] = self.beta * self.mom[k] + (1 - self.beta) * gE[k]
            gE[k] = self.mom[k]
        alpha_t = self.alpha * 150.0 / (gen + 150.0)
        with torch.no_grad():
            for k in eff:
                if k in self.A:
                    dA = gE[k] @ self.B[k].T
                    dB = self.A[k].T @ gE[k]
                    for M in (dA, dB):
                        M -= M.mean(); M /= (M.std() + 1e-8)
                    self.A[k] += alpha_t * dA
                    self.B[k] += alpha_t * dB
                else:
                    a = gE[k]; a = (a - a.mean()) / (a.std() + 1e-8)
                    self.base[k] += alpha_t * a
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


V, d, ffn, nblocks, ctx, nb = 256, 64, 128, 2, 64, 64
data = np.frombuffer(open(r"B:\git\tinyshakespeare.txt", "rb").read()[: 512 * 1024],
                     dtype=np.uint8).astype(np.int64)

# fresh minibatches per step (fixed-batch protocol saturates: Adam memorizes
# 4K tokens in <2s at lr=0.01) + held-out eval from the file tail (disjoint).
_r = np.random.default_rng(0)
_win = nb * (ctx + 1)
_train_off = _r.integers(0, len(data) - 64 * 1024 - _win, size=256)
train_pool = []
for o in _train_off:
    xx = torch.tensor(data[o:o + _win], device=DEV).view(nb, ctx + 1)
    train_pool.append((xx[:, :ctx], xx[:, 1:]))
eval_windows = []
for i in range(8):
    xx = torch.tensor(data[len(data) - (i + 1) * 8192:][:_win], device=DEV).view(nb, ctx + 1)
    eval_windows.append((xx[:, :ctx], xx[:, 1:]))

if DEV == 'cuda':
    torch.linalg.svd(torch.eye(8, device=DEV))   # absorb ~4s one-time rocSOLVER init


def run_config(name, seconds, K=None, adam_lr=0.01, selective_q=None,
               half_depth=False, seed_scale=0.02, warm=20):
    torch.manual_seed(0)
    model = LM(V, d, ffn, nblocks).to(DEV)
    loss_fn = make_loss(model)
    keys = dict(model.named_parameters())
    marks, t0 = {}, time.time()

    def heldout(params):
        with torch.no_grad():
            pd = {k: v.detach() for k, v in params.items()}
            return float(np.mean([float(loss_fn(pd, i, t)) for i, t in eval_windows]))

    if K == 'adam':                                # pure backprop baseline
        opt = torch.optim.Adam(model.parameters(), lr=adam_lr)
        params = dict(model.named_parameters())
        it = pi = 0
        while True:
            idx, tgt = train_pool[pi & 255]
            pi += 1
            opt.zero_grad(set_to_none=True)
            loss_fn(params, idx, tgt).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            it += 1
            if it % 8 == 0:
                if DEV == 'cuda':
                    torch.cuda.synchronize()
                el = time.time() - t0
                if el >= seconds:
                    break
                for q in (0.25, 0.5, 0.75):
                    if q not in marks and el >= q * seconds:
                        marks[q] = heldout(params)
        final, n_gen, n_anc = heldout(params), it, 0
    else:
        fold_every = 20 if K is None else 10 ** 9
        tr = AnchoredQGaLore(model, loss_fn, keys, K=K, adam_lr=adam_lr,
                             selective_q=selective_q, half_depth=half_depth,
                             seed_scale=seed_scale, fold_every=fold_every)
        gen = gen_since = pi = 0
        warmed = False

        def ce_eff():
            return heldout(tr._effective())

        while True:
            if DEV == 'cuda':
                torch.cuda.synchronize()
            el = time.time() - t0
            if el >= seconds:
                break
            for q in (0.25, 0.5, 0.75):
                if q not in marks and el >= q * seconds:
                    marks[q] = ce_eff()
            if not warmed:
                tr.base = {k: v.detach() for k, v in tr.base.items()}  # leaf-ify for warm_start
                tr.warm_start(*train_pool[pi & 255], steps=warm)
                pi += 1
                tr._rebind_trainable()
                tr.mom = None
                if K is not None:   # kill the warm-start A@B=G_avg offset:
                    for k in tr.A:  # the first anchor's fold would inject it as ascent
                        tr.A[k] = tr.A[k] * 0.1
                        tr.B[k] = tr.B[k] * 0.1
                warmed = True
            elif K is not None and (K == 0 or gen_since >= K):
                tr.anchor(*train_pool[pi & 255])
                pi += 1
                gen_since = 0
            else:
                tr.step(*train_pool[pi & 255], gen)
                pi += 1
                gen += 1
                gen_since += 1
        n_gen, n_anc = gen, tr.anchors
        final = ce_eff()

    qs = "  ".join(f"{int(q * 100)}%:{marks[q]:.3f}" for q in sorted(marks))
    print(f"{name:24s} heldout CE {final:8.4f} | {qs} | gens {n_gen:4d} "
          f"anchors {n_anc:4d} | {time.time() - t0:5.1f}s", flush=True)
    return final


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    SEC = float(os.environ.get("HYB_SECONDS", 20))
    if mode == "calib":
        for lr in (0.003, 0.01, 0.03):
            run_config(f"adam lr={lr}", SEC, K='adam', adam_lr=lr)
    else:
        lr = float(os.environ.get("HYB_LR", 0.01))
        run_config("adam-pure", SEC, K='adam', adam_lr=lr)
        for K in (1, 2, 5, 10, 20, 50, None, 0):
            tag = "K=inf" if K is None else f"K={K}"
            run_config(f"hybrid {tag}", SEC, K=K, adam_lr=lr)
        run_config("hybrid K=10 q=0.3", SEC, K=10, adam_lr=lr, selective_q=0.3)
        run_config("hybrid K=10 q=0.5", SEC, K=10, adam_lr=lr, selective_q=0.5)
        run_config("hybrid K=10 half-depth", SEC, K=10, adam_lr=lr, half_depth=True)
        run_config("hybrid K=1 gamma=0.25", SEC, K=1, adam_lr=lr, seed_scale=0.25)
        run_config("hybrid K=1 gamma=0.05", SEC, K=1, adam_lr=lr, seed_scale=0.05)
