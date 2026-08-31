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
from torch.func import functional_call, jvp

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
        self.keys = ([k for k in named if not k.startswith('head.')]
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


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'run'
    SEC = float(os.environ.get('HRM_SECONDS', 20))
    if mode == 'calib':
        for lr in (0.003, 0.01, 0.03):
            arm_A(8, lr=lr)
    else:
        LRA = float(os.environ.get('HRM_LR_A', 0.01))
        arm_D(SEC, lr=LRA)
        arm_A(SEC, lr=LRA)
        arm_B(SEC)
        arm_C(SEC, head_lr=LRA)
        arm_C(SEC, alpha=0.01, head_lr=LRA)
