"""Scale test: EGGROLL-trained spike-MoE LM at ~200M parameters.

Same validated architecture (frozen hash router, spike K/V attention,
top-1 routed + shared experts, retrieval head), now with:

- byte-level WikiText-2 (V=256, real corpus)
- ScaleEggroll: factor-based ES -- perturbations stay as (u, v) factors,
  antithetic signs folded into fitness weights, update accumulated as one
  U'F V'^T GEMM per matrix. O(r(m+n)) memory, O(r*m*n) FLOPs once per
  generation -- never 32 materialized full-size perturbations.

Usage:
  python scale_train.py [--d 768 --hid 2048 --nexp 8 --blocks 7 --gens 100 ...]
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, r"B:\git\cubbyllm-lite")
from cubbylite.model import SpikeMoELM

p = argparse.ArgumentParser()
p.add_argument("--d", type=int, default=768)
p.add_argument("--hid", type=int, default=2048)
p.add_argument("--nexp", type=int, default=8)
p.add_argument("--blocks", type=int, default=7)
p.add_argument("--ctx", type=int, default=256)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--pop", type=int, default=64)
p.add_argument("--sigma", type=float, default=0.05)
p.add_argument("--alpha", type=float, default=0.01)
p.add_argument("--rank", type=int, default=8)
p.add_argument("--gens", type=int, default=100)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", type=str, default="scale")
args = p.parse_args()

rng = np.random.default_rng(args.seed)

# ---------------- data: byte-level wikitext-2 ----------------
train_text = open(r"B:\git\wikitext2_train.txt", encoding="utf-8").read()
test_text = open(r"B:\git\wikitext2_test.txt", encoding="utf-8").read()
train_ids = np.frombuffer(train_text.encode("utf-8"), dtype=np.uint8).astype(np.int64)
test_ids = np.frombuffer(test_text.encode("utf-8"), dtype=np.uint8).astype(np.int64)
V = 256
CTX = args.ctx
print(f"train {len(train_ids)/1e6:.1f}MB bytes, test {len(test_ids)/1e6:.1f}MB, V={V}", flush=True)

def batch(ids, bs, train=True):
    hi = (len(ids) - CTX - 1) if train else (len(ids) - CTX - 1)
    lo = 0 if train else int(len(ids) * 0.9)   # eval from tail of test file
    ix = rng.integers(lo, hi, size=bs)
    x = np.stack([ids[i:i + CTX] for i in ix])
    y = np.stack([ids[i + 1:i + CTX + 1] for i in ix])
    return x, y

# ---------------- model ----------------
t0 = time.time()
model = SpikeMoELM(V, d=args.d, hid=args.hid, n_exp=args.nexp, n_blocks=args.blocks,
                   router="hash", head="retrieval", seed=args.seed)
n_params = sum(w.size for w in model.W.values())
print(f"model: {n_params/1e6:.1f}M params (d={args.d}, hid={args.hid}, "
      f"{args.nexp}+1 experts, {args.blocks} blocks)  init {time.time()-t0:.0f}s", flush=True)

x_ev, y_ev = batch(test_ids, 16, train=False)

# ---------------- ScaleEggroll: factor-based ES ----------------
class ScaleEggroll:
    def __init__(self, W, keys, pop, sigma, alpha, rank):
        self.W, self.keys = W, keys
        self.pop, self.sigma, self.alpha, self.rank = pop, sigma, alpha, rank
        self.base = {k: W[k].copy() for k in keys}
        self.rng = np.random.default_rng(1234)

    def _factors(self):
        fs = {}
        for k in self.keys:
            s = self.W[k].shape
            if len(s) == 1:
                fs[k] = (self.rng.standard_normal((s[0], 1)).astype(np.float32), None)
                continue
            r = min(self.rank, min(s))
            u = self.rng.standard_normal((s[0], r)).astype(np.float32)
            v = self.rng.standard_normal((r, s[1])).astype(np.float32)
            fs[k] = (u, v)
        return fs

    def apply(self, fs, sign, d_cache=None):
        for k in self.keys:
            u, v = fs[k]
            if d_cache is not None and k in d_cache:
                d = d_cache[k]
            else:
                d = u @ v / np.sqrt(u.shape[1]) if v is not None else u[:, 0]
                if d_cache is not None:
                    d_cache[k] = d
            self.W[k][...] = self.base[k] + sign * self.sigma * d

    def step(self, fitness_fn, gen):
        half = self.pop // 2
        fit = []
        factors = []
        for _ in range(half):
            fs = self._factors()
            dc = {}
            self.apply(fs, +1.0, dc); fit.append(fitness_fn())
            self.apply(fs, -1.0, dc); fit.append(fitness_fn())   # reuses cached d
            factors.append(fs)
        f = np.array(fit, dtype=np.float64)
        f = (f - f.mean()) / (f.std() + 1e-8)
        # antithetic: minus-half gets negated weight; fold into stacked factors
        for k in self.keys:
            u0, v0 = factors[0][k]
            if v0 is None:
                m = u0.shape[0]
                U = np.stack([factors[i // 2][k][0][:, 0] * (1 if i % 2 == 0 else -1)
                              for i in range(self.pop)], axis=1)   # (m, pop)
                acc = (U * f).sum(axis=1)
                a = acc / (acc.std() + 1e-8)
                self.base[k] += self.alpha * 150.0 / (gen + 150.0) * a.astype(np.float32)
                continue
            r = u0.shape[1]; m, n = u0.shape[0], v0.shape[1]
            U = np.empty((m, r * self.pop), dtype=np.float32)
            V = np.empty((r * self.pop, n), dtype=np.float32)
            for i in range(self.pop):
                ui, vi = factors[i // 2][k]
                s = 1.0 if i % 2 == 0 else -1.0
                U[:, i * r:(i + 1) * r] = ui * s
                V[i * r:(i + 1) * r, :] = vi
            acc = (U * np.repeat(f, r)) @ V / np.sqrt(r)      # one factored GEMM
            a = (acc - acc.mean()) / (acc.std() + 1e-8)       # scale-free step
            self.base[k] += self.alpha * 150.0 / (gen + 150.0) * a.astype(np.float32)
        for k in self.keys:
            self.W[k][...] = self.base[k]
        return float(f.mean())

trainer = ScaleEggroll(model.W, model.trainable(), args.pop, args.sigma, args.alpha, args.rank)

def fitness():
    x, y = batch(train_ids, args.batch)
    return -model.ce(x, y)

def evaluate(tag, gen):
    ce = model.ce(x_ev, y_ev)
    _, bal = model.forward(x_ev)
    b = np.mean(bal, axis=0)
    print(f"gen {gen:4d}  ce {ce:.3f}  ppl {np.exp(ce):7.1f}  "
          f"bal {b.min():.2f}/{b.max():.2f}  ({time.time()-t0:.0f}s)", flush=True)
    return ce

print(f"uniform CE = {np.log(V):.3f}", flush=True)
evaluate("init", 0)
t_gen = None
for gen in range(1, args.gens + 1):
    t1 = time.time()
    trainer.step(fitness, gen)
    t_gen = time.time() - t1
    if gen % 10 == 0 or gen == args.gens:
        evaluate("eval", gen)
        print(f"         last gen took {t_gen:.1f}s", flush=True)
print(f"DONE tag={args.tag} params={n_params/1e6:.1f}M gens={args.gens} s/gen={t_gen:.1f}", flush=True)
