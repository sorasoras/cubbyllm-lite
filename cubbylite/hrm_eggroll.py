"""Backprop-free HRM: one-step-gradient substrate (arXiv 2506.21734) with
forward-only training signals, tested against controls.

Micro-HRM: L-block reused nL times per cycle, H-block once per cycle
(nH cycles/segment) -> effective depth nL*nH from TWO shared blocks.
One-step gradient path (the paper's, minus ACT/deep-supervision segments):
  head -> final H application -> final L application -> input embedding
All other states treated as constants (O(1) memory, no BPTT).

Arms (fresh minibatches + held-out eval + matched wall-clock, per
hybrid_anchor.py; CE also recorded at matched update counts):
  D xformer-adam : plain 2-block transformer + Adam              [reference]
  A hrm-adam1s   : HRM + Adam + exact one-step gradients         [reference]
  B hrm-es-global: HRM + global-ES population through the FULL unroll [control]
  C hrm-bpl      : backpropless — closed-form head gradient
                   (dW=(p-onehot)h^T) + short-path JVP populations (P=32
                   rank-8 tangents through ONLY the 2 one-step applications)
"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp, vmap

from qgalore_eggroll import LM, Block, make_loss, DEV
from hybrid_anchor import train_pool, eval_windows, V, d, ffn, ctx


def pref(p, prefix):
    return {k[len(prefix):]: v for k, v in p.items() if k.startswith(prefix)}


class HRMBlock(nn.Module):
    """POST-norm transformer block, parameter-free RMSNorm, truncated LeCun
    init — the paper's stability recipe. Post-norm (not prenorm) is load-
    bearing: it bounds the recurrent state magnitude at every application,
    which is what makes an 18-deep weight-shared recurrence trainable."""

    def __init__(self, d, ffn):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.w1 = nn.Linear(d, ffn, bias=False)
        self.w2 = nn.Linear(ffn, d, bias=False)
        for m in (self.qkv, self.proj, self.w1, self.w2):
            std = math.sqrt(1.0 / m.weight.shape[1])
            torch.nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(d)
        att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1), -1e9)
        x = self._rms(x + self.proj(att.softmax(-1) @ v))
        return self._rms(x + self.w2(F.gelu(self.w1(x))))

    @staticmethod
    def _rms(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class HRM(nn.Module):
    def __init__(self, V=V, d=d, ffn=ffn, nL=5, nH=3, T=ctx):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.L = HRMBlock(d, ffn)
        self.H = HRMBlock(d, ffn)
        self.head = nn.Linear(d, V, bias=False)
        # per-cycle aux readout heads: deep supervision without corrupting
        # the final head (a shared head trained on mid-computation states
        # blows up the final readout — measured)
        self.aux = nn.ModuleList([nn.Linear(d, V, bias=False)
                                  for _ in range(nH - 1)])
        self.nL, self.nH = nL, nH
        g = torch.Generator().manual_seed(7)
        self.register_buffer('h0', torch.randn(1, T, d, generator=g) * 0.1)
        self.register_buffer('z0', torch.randn(1, T, d, generator=g) * 0.1)

    def unroll(self, x, h=None, z=None, u=None):
        """No-grad unroll of one segment. Returns final (h, z), the one-step
        constants (h_prev, z_prev), and u."""
        if h is None: h = self.h0
        if z is None: z = self.z0
        with torch.no_grad():
            if u is None: u = self.emb(x)
            z_prev = h_prev = None
            for c in range(self.nH):
                for t in range(self.nL):
                    if c == self.nH - 1 and t == self.nL - 1:
                        z_prev = z
                    z = self.L(z + h + u)
                if c == self.nH - 1:
                    h_prev = h
                h = self.H(h + z + u)
        return h, z, h_prev, z_prev, u

    def one_step_logits(self, h_prev, z_prev, u):
        """The 2-application graph-carrying path: final L app, final H app, head."""
        z_star = self.L(z_prev + h_prev + u)
        h_star = self.H(h_prev + z_star + u)
        return self.head(h_star)


def heldout_hrm(model):
    with torch.no_grad():
        ces = []
        for i, t in eval_windows:
            h, z, _, _, _ = model.unroll(i)
            out = model.head(h)
            ces.append(float(F.cross_entropy(out.reshape(-1, out.shape[-1]), t.reshape(-1))))
        return float(np.mean(ces))


def heldout_lm(model, loss_fn):
    with torch.no_grad():
        p = dict(model.named_parameters())
        return float(np.mean([float(loss_fn(p, i, t)) for i, t in eval_windows]))


def run_loop(name, seconds, step_fn, eval_fn, sync_every=1):
    t0, marks, umark = time.time(), {}, {}
    upd = [0]
    while True:
        step_fn(upd)
        u = upd[0]
        if u in (30, 100) and u not in umark:
            umark[u] = eval_fn()
        if u % sync_every == 0:
            if DEV == 'cuda':
                torch.cuda.synchronize()
            el = time.time() - t0
            if el >= seconds:
                break
            for q in (.25, .5, .75):
                if q not in marks and el >= q * seconds:
                    marks[q] = eval_fn()
    final = eval_fn()
    qs = "  ".join(f"{int(q * 100)}%:{marks[q]:.3f}" for q in sorted(marks))
    us = "  ".join(f"u{u}:{umark[u]:.3f}" for u in sorted(umark))
    print(f"{name:18s} heldout CE {final:8.4f} | {qs} | {us} | "
          f"updates {upd[0]:5d} | {time.time() - t0:5.1f}s", flush=True)
    return final


def arm_D(seconds, lr=0.01):
    torch.manual_seed(0)
    model = LM(V, d, ffn, 2).to(DEV)
    loss_fn = make_loss(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    params = dict(model.named_parameters())
    pi = [0]

    def step(upd):
        idx, tgt = train_pool[pi[0] & 255]; pi[0] += 1
        opt.zero_grad(set_to_none=True)
        loss_fn(params, idx, tgt).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); upd[0] += 1

    run_loop('xformer-adam', seconds, step, lambda: heldout_lm(model, loss_fn), sync_every=8)


def arm_A(seconds, lr=0.01):
    torch.manual_seed(0)
    model = HRM().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pi = [0]

    def step(upd):
        idx, tgt = train_pool[pi[0] & 255]; pi[0] += 1
        u = model.emb(idx)                                    # graph-carrying
        h, z, h_prev, z_prev, _ = model.unroll(idx, u=u)
        logits = model.one_step_logits(h_prev, z_prev, u)
        loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); upd[0] += 1

    run_loop('hrm-adam1s', seconds, step, lambda: heldout_hrm(model), sync_every=8)


class EsHrm:
    """Population trainer over HRM params. mode='short': tangents through the
    2 one-step applications only, head updated by the closed-form exact
    gradient (the backpropless stack). mode='global': tangents through the
    FULL unroll including head (the transformer-style global-ES control)."""

    def __init__(self, model, mode='short', pop=32, alpha=0.004, rank=8,
                 head_lr=0.01, seed=0):
        self.model, self.mode, self.pop, self.alpha, self.r = model, mode, pop, alpha, rank
        named = dict(model.named_parameters())
        self.keys = ([k for k in named if not k.startswith('head.')
                      and not k.startswith('aux.')]
                     if mode == 'short' else list(named))
        self.base = {k: named[k].detach().clone() for k in self.keys}
        self.model_params = named
        self.head_opt = torch.optim.Adam([named['head.weight']], lr=head_lr)
        self.g = torch.Generator(device=DEV).manual_seed(seed)

    def _tangent(self, k):
        s = self.base[k].shape
        if len(s) < 2:
            return torch.randn(s, generator=self.g, device=DEV)
        r = min(self.r, min(s))
        u = torch.randn((s[0], r), generator=self.g, device=DEV)
        v = torch.randn((r, s[1]), generator=self.g, device=DEV)
        return (u @ v) / math.sqrt(r)

    def _short_loss(self, p, x, h_prev, z_prev, tgt):
        m = self.model
        uu = functional_call(m.emb, pref(p, 'emb.'), (x,))
        z_star = functional_call(m.L, pref(p, 'L.'), (z_prev + h_prev + uu,))
        h_star = functional_call(m.H, pref(p, 'H.'), (h_prev + z_star + uu,))
        return F.cross_entropy(m.head(h_star).reshape(-1, V), tgt.reshape(-1))

    def _global_loss(self, p, x, tgt):
        m = self.model
        uu = functional_call(m.emb, pref(p, 'emb.'), (x,))
        h, z = m.h0, m.z0
        for c in range(m.nH):
            for t in range(m.nL):
                z = functional_call(m.L, pref(p, 'L.'), (z + h + uu,))
            h = functional_call(m.H, pref(p, 'H.'), (h + z + uu,))
        out = functional_call(m.head, pref(p, 'head.'), (h,))
        return F.cross_entropy(out.reshape(-1, V), tgt.reshape(-1))

    def step(self, x, tgt, gen):
        m = self.model
        h_fin, z_fin, h_prev, z_prev, _ = m.unroll(x)
        loss = (self._short_loss if self.mode == 'short' else self._global_loss)
        args = (x, h_prev, z_prev, tgt) if self.mode == 'short' else (x, tgt)
        params = dict(self.base)
        tans, ds = [], []
        for _ in range(self.pop):
            t = {k: self._tangent(k) for k in self.keys}
            _, dd = jvp(lambda p: loss(p, *args), (params,), (t,))
            ds.append(dd); tans.append(t)
        dst = torch.stack(ds)
        acc = {k: torch.zeros_like(self.base[k]) for k in self.keys}
        for i in range(self.pop):
            for k in self.keys:
                acc[k] -= (dst[i] / self.pop) * tans[i][k]
        alpha_t = self.alpha * 150.0 / (gen + 150.0)
        with torch.no_grad():
            for k in self.keys:
                a = acc[k]; a = (a - a.mean()) / (a.std() + 1e-8)
                self.base[k] += alpha_t * a
                self.model_params[k].data.copy_(self.base[k])
        if self.mode == 'short':      # exact closed-form head gradient
            with torch.no_grad():
                p = m.head(h_fin).softmax(-1)
                onehot = F.one_hot(tgt, V).float()
                diff = (p - onehot).reshape(-1, V)
                m.head.weight.grad = (diff.T @ h_fin.reshape(-1, h_fin.shape[-1])
                                      / diff.shape[0])
            self.head_opt.step(); self.head_opt.zero_grad()
        return float(dst.mean())


def gelu_prime(x):
    return (0.5 * (1 + torch.erf(x / math.sqrt(2.0)))
            + x * torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi))


def block_features(blk, x):
    """Forward run of a block returning its internal features (for guided
    tangent construction — all forward-computable, no autograd)."""
    B, T, d = x.shape
    q, k, v = blk.qkv(x).split(d, dim=-1)
    att = (q @ k.transpose(-1, -2)) / math.sqrt(d)
    att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1), -1e9)
    a = att.softmax(-1)
    av = a @ v
    z1 = x + blk.proj(av)
    x1 = blk._rms(z1)
    h1 = blk.w1(x1)
    g1 = F.gelu(h1)
    z2 = x1 + blk.w2(g1)
    x2 = blk._rms(z2)
    return {'x0': x, 'q': q, 'k': k, 'v': v, 'att': a, 'av': av, 'z1': z1,
            'x1': x1, 'h1': h1, 'g1': g1, 'z2': z2, 'out': x2}


def rms_jac_apply(x, e):
    """J_rms @ e for y = x / sqrt(mean(x^2)+eps). The rms Jacobian is
    SYMMETRIC, so its transpose is itself — applying it is a forward matvec."""
    r = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    return e / r - x * (x * e).sum(-1, keepdim=True) / (x.shape[-1] * r ** 3)


def mlp_tail_grads(blk, f, e):
    """Exact error chain through the MLP tail (all symmetric/elementwise/
    linear ops — no autograd, no stored activations, no attention backward):
    e -> rms2 -> {w2, w1, proj} gradient outer products + the propagated
    error at the block input (residual path)."""
    dx2 = rms_jac_apply(f['z2'], e)
    dxF = dx2.reshape(-1, dx2.shape[-1])
    g1F = f['g1'].reshape(-1, f['g1'].shape[-1])
    x1F = f['x1'].reshape(-1, f['x1'].shape[-1])
    avF = f['av'].reshape(-1, f['av'].shape[-1])
    g = {'w2.weight': dxF.T @ g1F}
    dg1 = dx2 @ blk.w2.weight
    dh1 = dg1 * gelu_prime(f['h1'])
    g['w1.weight'] = dh1.reshape(-1, dh1.shape[-1]).T @ x1F
    dx1 = dx2 + dh1 @ blk.w1.weight
    dz1 = rms_jac_apply(f['z1'], dx1)
    g['proj.weight'] = dz1.reshape(-1, dz1.shape[-1]).T @ avF
    return g, dz1


def att_backward(blk, f, dz1, x_in):
    """Exact attention backward from saved forward values only. The softmax
    Jacobian is row-symmetric (J = diag(A) - A A^T), so applying it is a
    forward elementwise op; everything else is forward matmuls on saved
    tensors. Returns (dv, dq, dk) and the FULL input gradient dx0."""
    d_av = dz1 @ blk.proj.weight                       # error at av
    dv = f['att'].transpose(-1, -2) @ d_av             # error at v (exact)
    dA = torch.bmm(d_av, f['v'].transpose(-1, -2))     # d_av @ v^T
    A = f['att']
    dS = A * (dA - (dA * A).sum(-1, keepdim=True))      # softmax Jacobian apply
    dq = dS @ f['k'] / math.sqrt(d)
    dk = dS.transpose(-1, -2) @ f['q'] / math.sqrt(d)
    Wq, Wk, Wv = (blk.qkv.weight[:d], blk.qkv.weight[d:2 * d],
                  blk.qkv.weight[2 * d:])
    dx0 = dz1 + dq @ Wq + dk @ Wk + dv @ Wv            # all four paths
    return dv, dq, dk, dx0


class BplHrm2:
    """Refined backpropless stack (arm C2). Three refinements over C:
    1. per-cycle deep supervision — head readout at EVERY H-update, so each
       unroll yields nH short-path populations (3x96 directional
       measurements/update) instead of one;
    2. vmap-batched populations — one batched jvp instead of 32 Python calls
       (the measured step-rate killer: 118 vs 1944 updates in 20s);
    3. momentum (0.9) on the reconstruction — the power-iteration ingredient
       that mattered most in the Q-GaLore fix.
    Head still trained by the exact closed-form gradient, summed over cycles."""

    def __init__(self, model, pop=32, alpha=0.004, rank=8, head_lr=0.003,
                 cycles=3, beta=0.9, reuse=1, decay=150.0, rule='std',
                 adam_lr=0.003, guided=False, seed=0):
        self.model, self.pop, self.alpha, self.r = model, pop, alpha, rank
        self.cycles, self.beta, self.reuse, self.decay = cycles, beta, reuse, decay
        self.rule, self.adam_lr, self.guided = rule, adam_lr, guided
        self._adamopt = None
        self._cache, self._reused = None, 0
        named = dict(model.named_parameters())
        self.keys = [k for k in named
                     if not k.startswith('head.') and not k.startswith('aux.')]
        self.base = {k: named[k].detach().clone() for k in self.keys}
        self.model_params = named
        self.head_opt = torch.optim.Adam(
            [named['head.weight']] + [p for k, p in named.items()
                                      if k.startswith('aux.')], lr=head_lr)
        self.g = torch.Generator(device=DEV).manual_seed(seed)
        self.mom = None
        self.ce_ema = {}          # per-cycle CE EMA — gates aux supervision

    def _head_for(self, c):
        m = self.model
        return m.aux[c] if c < m.nH - 1 else m.head

    def _rand_tangent(self, k):
        s = self.base[k].shape
        if len(s) < 2:
            return torch.randn(s, generator=self.g, device=DEV)
        r = min(s) if self.r is None else min(self.r, min(s))
        uu = torch.randn((s[0], r), generator=self.g, device=DEV)
        vv = torch.randn((r, s[1]), generator=self.g, device=DEV)
        return (uu @ vv) / math.sqrt(r)

    def _tangent_batch(self, k):
        s = self.base[k].shape
        P = self.pop
        if len(s) < 2:
            return torch.randn((P, *s), generator=self.g, device=DEV)
        r = min(s) if self.r is None else min(self.r, min(s))   # None = full rank
        u = torch.randn((P, s[0], r), generator=self.g, device=DEV)
        v = torch.randn((P, r, s[1]), generator=self.g, device=DEV)
        return torch.bmm(u, v) / math.sqrt(r)

    def _cycle_loss(self, p, x, consts, tgt, c):
        z_prev, h_prev = consts
        m = self.model
        uu = functional_call(m.emb, pref(p, 'emb.'), (x,))
        z_star = functional_call(m.L, pref(p, 'L.'), (z_prev + h_prev + uu,))
        h_star = functional_call(m.H, pref(p, 'H.'), (h_prev + z_star + uu,))
        return F.cross_entropy(self._head_for(c)(h_star).reshape(-1, V),
                               tgt.reshape(-1))

    def _guided(self, x, tgt, consts, hs, fL, fH):
        """Composite guided tangent = the exact one-step gradient, built from
        the unroll's captured features via forward-op chains only (symmetric
        rms/softmax Jacobians + forward matmuls — no autograd, no stored
        graph, no activation memory beyond the two final applications)."""
        m = self.model
        with torch.no_grad():
            onehotG = F.one_hot(tgt, V).float()
            p = m.head(hs[-1]).softmax(-1)
            e = (p - onehotG) @ m.head.weight          # (B,T,d) exact
            gH, dz1H = mlp_tail_grads(m.H, fH, e)
            dvH, dqH, dkH, dx0H = att_backward(m.H, fH, dz1H, fH['x0'])
            e_L = dx0H                                   # exact error at L's output
            gL, dz1L = mlp_tail_grads(m.L, fL, e_L)
            dvL, dqL, dkL, dx0L = att_backward(m.L, fL, dz1L, fL['x0'])
            gmap = {}
            for g, pre in ((gH, 'H.'), (gL, 'L.')):
                for k, v in g.items():
                    gmap[pre + k] = v
            for pre, f, dv, dq, dk in (
                    ('H.', fH, dvH, dqH, dkH), ('L.', fL, dvL, dqL, dkL)):
                xF = f['x0'].reshape(-1, d)
                E = torch.empty_like(self.base[pre + 'qkv.weight'])
                E[:d, :] = dq.reshape(-1, d).T @ xF
                E[d:2 * d, :] = dk.reshape(-1, d).T @ xF
                E[2 * d:, :] = dv.reshape(-1, d).T @ xF
                gmap[pre + 'qkv.weight'] = E
            Eemb = torch.zeros_like(self.base['emb.weight'])
            Eemb.index_add_(0, x.reshape(-1), (dx0L + dx0H).reshape(-1, d))
            gmap['emb.weight'] = Eemb
            return {k: (gmap[k] / (gmap[k].std() + 1e-8) if k in gmap
                       else torch.zeros_like(self.base[k]))
                    for k in self.keys}
            return {k: (gmap[k] / (gmap[k].std() + 1e-8) if k in gmap
                       else torch.zeros_like(self.base[k]))
                    for k in self.keys}

    def step(self, x, tgt, gen):
        m = self.model
        if self._cache is None or self._reused >= self.reuse:
            with torch.no_grad():                  # shared unroll + per-cycle constants
                u = m.emb(x)
                h, z = m.h0, m.z0
                consts, hs = [], []
                fL = fH = None
                for c in range(m.nH):
                    for t in range(m.nL):
                        if t == m.nL - 1:
                            zc = z
                        if c == m.nH - 1 and t == m.nL - 1:
                            fL = block_features(m.L, z + h + u)   # capture the
                            z = fL['out']                          # final apps'
                        else:                                       # features in
                            z = m.L(z + h + u)                      # the unroll —
                    consts.append((zc, h))                          # no re-run
                    za = z                          # post-L, pre-H state (last cycle)
                    if c == m.nH - 1:
                        fH = block_features(m.H, h + z + u)
                        h = fH['out']
                    else:
                        h = m.H(h + z + u)
                    hs.append(h)
                self._cache = (x, tgt, consts, hs, za, u, fL, fH)
            self._reused = 0
        self._reused += 1                          # amortize: the one-step approx
        x, tgt, consts, hs, za, u, fL, fH = self._cache   # treats states as constants
        # gate aux cycles on their own learnability: early-cycle readouts at
        # unorganized states produce noise gradients at full standardized
        # step magnitude (measured: blows up to ~5.5), so a cycle joins the
        # module population only once its head has learned to read it
        # (CE meaningfully below the random baseline ln V = 5.545).
        with torch.no_grad():
            onehot = F.one_hot(tgt, V).float()
            for c in range(m.nH - self.cycles, m.nH):
                p_c = self._head_for(c)(hs[c])
                ce_c = float(F.cross_entropy(p_c.reshape(-1, V), tgt.reshape(-1)))
                self.ce_ema[c] = 0.9 * self.ce_ema.get(c, ce_c) + 0.1 * ce_c
        last = m.nH - 1
        cyc = [c for c in range(m.nH - self.cycles, m.nH)
               if c == last or self.ce_ema[c] < 5.0]
        params = dict(self.base)
        acc = {k: torch.zeros_like(self.base[k]) for k in self.keys}

        guided_t = None
        if self.guided:
            guided_t = self._guided(x, tgt, consts, hs, fL, fH)
            lastc = m.nH - 1
            _, dg = jvp(lambda p: self._cycle_loss(p, x, consts[lastc], tgt, lastc),
                        (params,), (guided_t,))
            for k in self.keys:
                acc[k] -= float(dg) * guided_t[k]

        def dval(t, cc):
            _, dv = jvp(lambda p: self._cycle_loss(p, x, consts[cc], tgt, cc),
                        (params,), (t,))
            return dv

        if self.pop > 0:
            for c in cyc:
                T = {k: self._tangent_batch(k) for k in self.keys}
                try:
                    ds = vmap(lambda t: dval(t, c))(T)
                except Exception:
                    ds = torch.stack([dval({k: T[k][i] for k in T}, c)
                                      for i in range(self.pop)])
                for k in self.keys:
                    acc[k] -= torch.einsum('i,i...->...', ds, T[k]) / self.pop
        if self.rule == 'adam':
            if self._adamopt is None:
                self._adamopt = torch.optim.Adam(
                    [self.base[k] for k in self.keys], lr=self.adam_lr)
            for k in self.keys:
                self.base[k].grad = -acc[k].detach()   # descent direction
            self._adamopt.step()
            self._adamopt.zero_grad()
        else:
            if self.mom is None:
                self.mom = {k: torch.zeros_like(self.base[k]) for k in self.keys}
            alpha_t = self.alpha * self.decay / (gen + self.decay)
            with torch.no_grad():
                for k in self.keys:
                    self.mom[k] = self.beta * self.mom[k] + (1 - self.beta) * acc[k]
                    a = self.mom[k]
                    a = (a - a.mean()) / (a.std() + 1e-8)
                    self.base[k] += alpha_t * a
        with torch.no_grad():
            for k in self.keys:
                self.model_params[k].data.copy_(self.base[k])
        with torch.no_grad():                      # exact per-cycle head gradients
            onehot = F.one_hot(tgt, V).float()
            for c in range(m.nH - self.cycles, m.nH):   # all cycles: heads
                h_c = hs[c]                              # always train (the gate
                p_c = self._head_for(c)(h_c).softmax(-1) # only gates populations)
                diff = (p_c - onehot).reshape(-1, V)
                head_c = self._head_for(c)
                head_c.weight.grad = (diff.T @ h_c.reshape(-1, h_c.shape[-1])
                                      / diff.shape[0])
            self.head_opt.step()
            self.head_opt.zero_grad()


def arm_B(seconds, alpha=0.004):
    torch.manual_seed(0)
    model = HRM().to(DEV)
    tr = EsHrm(model, mode='global', alpha=alpha)
    pi = [0]

    def step(upd):
        idx, tgt = train_pool[pi[0] & 255]; pi[0] += 1
        tr.step(idx, tgt, upd[0]); upd[0] += 1

    run_loop('hrm-es-global', seconds, step, lambda: heldout_hrm(model))


def arm_C(seconds, alpha=0.004, head_lr=0.01):
    torch.manual_seed(0)
    model = HRM().to(DEV)
    tr = EsHrm(model, mode='short', alpha=alpha, head_lr=head_lr)
    pi = [0]

    def step(upd):
        idx, tgt = train_pool[pi[0] & 255]; pi[0] += 1
        tr.step(idx, tgt, upd[0]); upd[0] += 1

    run_loop('hrm-bpl', seconds, step, lambda: heldout_hrm(model))


def arm_C2(seconds, alpha=0.004, head_lr=0.003, cycles=3, pop=32,
           beta=0.9, rank=8, reuse=1, decay=150.0, rule='std', adam_lr=0.003,
           guided=False, name='hrm-bpl2'):
    torch.manual_seed(0)
    model = HRM().to(DEV)
    tr = BplHrm2(model, alpha=alpha, head_lr=head_lr, cycles=cycles,
                 pop=pop, beta=beta, rank=rank, reuse=reuse, decay=decay,
                 rule=rule, adam_lr=adam_lr, guided=guided)
    pi = [0]

    def step(upd):
        idx, tgt = train_pool[pi[0] & 255]; pi[0] += 1
        tr.step(idx, tgt, upd[0]); upd[0] += 1

    run_loop(name, seconds, step, lambda: heldout_hrm(model))


def arm_A2(seconds, alpha=0.004, head_lr=0.003, decay=150.0):
    """DIAGNOSTIC: exact one-step gradients through arm C's UPDATE RULE
    (per-key standardize + alpha decay; Adam only on the head, as in C).
    Isolates update-rule quality from estimation quality."""
    torch.manual_seed(0)
    model = HRM().to(DEV)
    named = dict(model.named_parameters())
    mod_keys = [k for k in named if not k.startswith('head.')
                and not k.startswith('aux.')]
    head_opt = torch.optim.Adam([named['head.weight']], lr=head_lr)
    pi = [0]

    def step(upd):
        idx, tgt = train_pool[pi[0] & 255]; pi[0] += 1
        u = model.emb(idx)
        h, z, h_prev, z_prev, _ = model.unroll(idx, u=u)
        logits = model.one_step_logits(h_prev, z_prev, u)
        loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
        model.zero_grad(set_to_none=True)
        loss.backward()
        alpha_t = alpha * decay / (upd[0] + decay)
        with torch.no_grad():
            for k in mod_keys:
                g = named[k].grad
                a = -g
                a = (a - a.mean()) / (a.std() + 1e-8)
                named[k].data += alpha_t * a
        head_opt.step()
        upd[0] += 1

    run_loop('hrm-exact-sgdst', seconds, step, lambda: heldout_hrm(model),
             sync_every=8)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'run'
    SEC = float(os.environ.get('HRM_SECONDS', 20))
    if mode == 'calib':
        for lr in (0.003, 0.01, 0.03):
            arm_A(8, lr=lr)
    elif mode == 'final':
        # the goal-run config: exact forward-op-chain gradients, no population,
        # no reuse, Adam; arm A at its calibrated lr as the matched reference
        arm_A(SEC, lr=0.003)
        arm_C2(SEC, head_lr=0.003, cycles=1, pop=0, beta=0.0,
               guided=True, rule='adam', adam_lr=0.003,
               name='hrm-bpl-final')
    else:
        LRA = float(os.environ.get('HRM_LR_A', 0.003))
        arm_D(SEC, lr=LRA)
        arm_A(SEC, lr=LRA)
        arm_B(SEC)
        arm_C(SEC, head_lr=LRA)
        arm_C(SEC, alpha=0.01, head_lr=LRA)
        arm_C2(SEC, head_lr=LRA)                                   # refined
        arm_C2(SEC, head_lr=LRA, cycles=1, name='hrm-bpl2 c1')     # ablation
