# CubbyLite

A dedicated, runnable mini-implementation of **[CubbyLLM](https://github.com/Grillcheese-AI/CubbyLLM)**'s
central bet, built on the **[grilly](https://github.com/Grillcheese-AI/grilly)** Vulkan backend and the
**EGGROLL backprop-free trainer** validated in-session.

**Central bet (CubbyLLM):** make a model's active weights a function of context rather than a fixed
stored state — the forgetting fix, the mechanism for automatic specialization, and the trick that
scales the output vocabulary. This repo implements the load-bearing subset and tests it:

| CubbyLLM pillar | CubbyLite implementation |
|---|---|
| Frozen router (online-learned routers forget themselves) | Multiplicative token-hash routing — zero routing parameters, allocation can never drift or collapse |
| Training follows the router's allocation | EGGROLL low-rank ES trains experts only on their assigned slice; fitness = pure task loss |
| Softmax-bypass output head | Cosine retrieval head against the tied embedding table |
| Spike-efficient compute | K/V as ternary spike trains; shared expert always on |
| Zero-Forgetting Stability Benchmark | Sequential correlated-domain training (plain → rot13 cipher), retention measured per domain |

## Architecture (≈300k params)

- 2 blocks × [spike self-attention (ternary K/V, causal) → frozen-hash top-1 of 7 routed experts + 1 shared expert]
- char-level LM, context 64, ~65-char vocab, grilly Vulkan GEMMs for all matmuls
- Training: EGGROLL ES — population 64, antithetic pairs, rank-8 factorized perturbations,
  fitness-standardized scale-free update, decayed step. **No gradients, no STE, no backward pass anywhere.**

## Results

### Zero-Forgetting Benchmark (`cubbylite/benchmark.py`)

Sequential training, 100 EGGROLL generations per phase, correlated domains
(rot13 = same alphabet, same language, shifted statistics):

| Router | CE_A init | after A | after B | ΔA (forgetting) | CE_B gain | Expert balance |
|---|---|---|---|---|---|---|
| **hash (frozen)** | 7.80 | **3.38** | 3.89 | **+0.514** | **0.682** | 0.05/0.32 |
| learned (quantile) | 7.80 | 3.47 | 3.99 | +0.526 | 0.633 | 0.11/0.17 |

### Findings (honest ones)

1. **The frozen router wins on everything it directly controls** — faster learning, better B
   acquisition, zero router parameters, zero balancing machinery. Consistent with CubbyLLM.
2. **But frozen routing alone does NOT prevent forgetting.** Both arms forget ≈equally (+0.51 vs
   +0.53). At this scale the interference runs through the **shared substrate the router does not
   partition**: the embedding table (which the retrieval head also reads), attention weights, and the
   shared experts. This *sharpens* CubbyLLM's own claim — their fix is not frozen routing alone but
   **context-conditioned parameter generation**; frozen routing is necessary, not sufficient.
3. **Hash routing is not load-balanced under skewed token frequencies** (0.05/0.32): frequent chars
   concentrate on one expert. Per-token-uniform ≠ per-instance-uniform. The learned quantile router
   balances better (0.11/0.17) at a small CE cost. A frequency-aware frozen hash (or quantile-corrected
   hash) is the obvious next variant.

## Run it

```
B:\git\grilly-venv\Scripts\python.exe cubbylite\benchmark.py    # ~5 min on RX 9070
```

## Roadmap (maps to CubbyLLM hypotheses)

- Route the embedding table and/or per-domain retrieval keys → attack the shared-substrate forgetting path
- Fully disjoint-vocabulary ablation (isolate expert-partition interference from substrate interference)
- Context-conditioned parameter generation: expert weights as a function of a frozen-router cluster embedding
- VSA binding head via grilly `BlockCodeOps` (pillar not implemented here)
- Scale: wider model, real tokenizer, EGGROLL rank/POP sweeps
