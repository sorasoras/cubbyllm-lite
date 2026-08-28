"""Zero-Forgetting Stability Benchmark (CubbyLite).

CubbyLLM's central claim under test, at demo scale: a frozen router makes
sequential learning safe because the allocation never moves -- each expert's
distribution is stationary, so Phase-B training does not re-shuffle what
Phase-A learned. The learned-router arm is the ablation: its allocation can
drift, recreating the interference it was supposed to avoid.

Protocol: correlated domains (rot13 cipher of the same corpus -- same
vocabulary, same language, different surface statistics).
  Phase 0: eval on held-out A and B.
  Phase 1: train on A only.
  Phase 2: train on B only.  Metric: CE_A retention drop during Phase 2,
           CE_B gain, and expert balance.
"""
import io
import sys
import time
import numpy as np

sys.path.insert(0, r"B:\git\cubbyllm-lite")
from cubbylite.model import SpikeMoELM
from cubbylite.eggroll import EggrollTrainer

import codecs
TEXT = open(r"B:\git\tinyshakespeare.txt", encoding="utf-8").read()
CORPUS_A = TEXT
CORPUS_B = codecs.encode(TEXT, "rot13")            # correlated domain, same alphabet

chars = sorted(set(CORPUS_A) | set(CORPUS_B))
stoi = {c: i for i, c in enumerate(chars)}
idsA = np.array([stoi[c] for c in CORPUS_A], dtype=np.int64)
idsB = np.array([stoi[c] for c in CORPUS_B], dtype=np.int64)

CTX, FIT_B, EVAL_N = 64, 24, 64
rng = np.random.default_rng(7)

def make_batches(ids):
    split = int(len(ids) * 0.9)
    def batch(bs, split_name="train"):
        lo, hi = ((0, split) if split_name == "train"
                  else (split, len(ids) - CTX - 1))
        ix = rng.integers(lo, hi, size=bs)
        return (np.stack([ids[i:i + CTX] for i in ix]),
                np.stack([ids[i + 1:i + CTX + 1] for i in ix]))
    return batch

batchA = make_batches(idsA)
batchB = make_batches(idsB)
xA, yA = batchA(EVAL_N, "eval")
xB, yB = batchB(EVAL_N, "eval")

GENS = 100

def run(router_mode):
    model = SpikeMoELM(len(chars), router=router_mode, head="retrieval", seed=1)
    trainer = EggrollTrainer(model.W, model.trainable(), seed=2)
    fit_gen = {"n": 0}

    def phase_fitness(batch):
        def f():
            x, y = batch(FIT_B)
            return -model.ce(x, y)
        return f

    def eval_both(tag):
        ceA, ceB = model.ce(xA, yA), model.ce(xB, yB)
        _, bal = model.forward(xA)
        b = np.mean(bal, axis=0)
        print(f"  [{router_mode:7s}] {tag:10s} CE_A {ceA:.3f}  CE_B {ceB:.3f}  "
              f"bal {b.min():.2f}/{b.max():.2f}", flush=True)
        return ceA, ceB

    print(f"== router={router_mode} ==", flush=True)
    t0 = time.time()
    ceA0, ceB0 = eval_both("init")
    for g in range(1, GENS + 1):
        trainer.step(phase_fitness(batchA), g)
    ceA1, ceB1 = eval_both("after-A")
    for g in range(GENS + 1, 2 * GENS + 1):
        trainer.step(phase_fitness(batchB), g)
    ceA2, ceB2 = eval_both("after-B")
    print(f"  [{router_mode:7s}] retention: dCE_A {ceA2-ceA1:+.3f} "
          f"(forgetting), gainB {ceB1-ceB2:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return dict(ceA0=ceA0, ceA1=ceA1, ceA2=ceA2, ceB1=ceB1, ceB2=ceB2)

if __name__ == "__main__":
    rows = {}
    for mode in ("hash", "learned"):
        rows[mode] = run(mode)
    print("\n=== summary (lower CE better; dCE_A > 0 = forgetting) ===")
    for m, r in rows.items():
        print(f"{m:7s}  A: {r['ceA0']:.2f} -> {r['ceA1']:.2f} -> {r['ceA2']:.2f}  "
              f"dA {r['ceA2']-r['ceA1']:+.3f}   B: {r['ceB1']:.2f} -> {r['ceB2']:.2f}")
