# CubbyLite

A runnable research repo for **backprop-free training of language models** —
started as a mini-implementation of [CubbyLLM](https://github.com/Grillcheese-AI/CubbyLLM)'s
central bet (context-conditioned weights, frozen routing, spike compute) and
grown into a measured campaign spanning three questions:

1. **How fast can int4 compute go on RDNA4?** (a hand-written WMMA kernel)
2. **Can forward-only (gradient-free) training match backprop?** (EGGROLL →
   JVP → Q-GaLore → anchored hybrids, every claim measured against controls)
3. **Is there an architecture where forward-only signals are enough?**
   (the Hierarchical Reasoning Model substrate)

Every result below is in [`kernels/RESULTS.md`](kernels/RESULTS.md) — the
master log with mechanisms, negative results, and honest verdicts.

## Headline results

| campaign | result |
|---|---|
| INT4 WMMA GEMM (gfx1201, hipRTC) | **261.4 TFLOPS** = 39.4% of the 663.5 TOPS real peak, **4.1× AMD's own int8** (63.8), 2.5× their fp16 (105.5) |
| JVP-EGGROLL vs finite-difference ES | exact directional derivatives (torch.func.jvp) beat FD — no σ, no antithetic noise |
| Q-GaLore-EGGROLL (gradient-free subspace tracking) | CE 4.767 vs 5.109 i.i.d. at matched budget, **without any gradient in the loop** |
| Forward-only vs backprop (transformer) | honest negative: ~2.4× worse per unit compute at GEMM-bound scale — the ES compute tax, measured |
| Anchored hybrid (backprop anchor every K gens) | K-curve measured; γ=0.25 diverges by fold-ascent feedback (the design law: anchors leave A@B ≈ 0); half-depth anchoring beats full-depth |
| **Backprop-free HRM (final)** | **CE 2.20–2.30 vs exact-gradient Adam's 2.360** at ~2000 matched updates, **0.98× its per-update compute** |

## The final stack: gradient-exact training without backward machinery

The endpoint of the campaign ([`cubbylite/hrm_eggroll.py`](cubbylite/hrm_eggroll.py)):
an HRM (two-timescale recurrent transformer, arXiv 2506.21734) trained with
one-step gradients computed **entirely from forward operations on saved
features** — no autograd, no stored graph, O(1) activation memory:

- the output error `e = (p − onehot)·W_head` is exact and forward-computable;
- the error chain through `rms → w2 → gelu′ → w1 → rms → proj → attention →
  embedding` uses only **symmetric Jacobians** (rms and softmax — their
  transpose is themselves, so applying them is a forward matvec) and forward
  matmuls;
- verified: cosine **+1.0000 with the true gradient on all 9 weight keys**,
  sustained through training, JVP-weighted (`d_g = ⟨∇, E⟩`);
- the JVP population machinery remains for any component not hand-derived.

The measured spectrum of what each signal class buys: pure JVP population
2.768 → vmap-batched 2.616 → exact forward-op chains 2.20–2.30 (Adam variant:
2.360). The late-training advantage over plain Adam is mechanistically
isolated: per-key gradient normalization drives Adam into its noise-robust
sign regime.

## Repo layout

```
cubbylite/           the training stack
  hrm_eggroll.py      backprop-free HRM + 4-arm controlled comparison (the final work)
  hybrid_anchor.py    anchored-hybrid K-curve harness (fresh-batch, held-out protocol)
  qgalore_eggroll.py  Q-GaLore-EGGROLL: gradient-free subspace tracking
  jvp_eggroll.py      JVP-EGGROLL + int4 dual-lane evaluation
  eggroll.py          the original validated ES trainer
  benchmark.py        frozen-router zero-forgetting benchmark (the origin)
kernels/             the int4 WMMA kernel campaign
  gemm_v19.py         the 261-TFLOPS champion (256×128 tiles, 16 warps)
  RESULTS.md          the master log — every round, mechanism, and verdict
results/              run logs
```

## Run it

```bash
# 4-arm controlled comparison (transformer-Adam vs HRM-Adam vs global-ES vs
# backprop-free), fresh minibatches + held-out eval, matched wall-clock:
QGAL_DEVICE=cuda python cubbylite/hrm_eggroll.py run

# the anchored-hybrid K-curve:
QGAL_DEVICE=cuda python cubbylite/hybrid_anchor.py sweep

# the int4 GEMM benchmark (RX 9070 / gfx1201):
python kernels/gemm_v19.py
```

Dependencies: PyTorch + a ROCm hipRTC runtime (TheRock dist works on
Windows/gfx1201). The kernel campaign assumes an AMD RDNA4 GPU.

## Honest findings worth keeping

- **Forward-only random search cannot match backprop on a standard
  transformer** — not for tuning-lack: rank, population, decay, optimizer
  and momentum were all swept; the information/cost arithmetic (2× per-update
  value for 4.84× cost) is structural. Recorded as such.
- **Post-norm is load-bearing** for weight-shared recurrence (unnormalized
  and prenorm both explode; the HRM paper's recipe independently confirmed).
- **Design law** (three independent confirmations): any update state derived
  from quantities the update itself influences creates positive feedback —
  pending updates must be ~zero at re-seed; search machinery must be
  statistically independent of the state it writes.
- **Fixed-batch smoke tests overstate forward-only methods** (Adam memorizes
  4K tokens in <2s); all comparisons here use fresh minibatches + held-out
  evaluation.
- **80% of int4 peak is blocked on toolchain** (no async-copy/DSMEM/TMA on
  gfx1201/hipRTC) — parked, with the sparse 2:4 path (1138 TOPS) recorded as
  the future avenue.
