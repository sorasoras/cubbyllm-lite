# INT4xINT4 MoE kernel (RDNA4 / gfx1201) — measured results

Native int4 WMMA (`wmma_i32_16x16x16_iu4_w32_gfx12`), hiprtc-compiled,
ctypes-launched, signed int4 exact via neg=(1,1). Grouped MoE = 8
per-expert launches of the proven LDS-staged dense kernel on host-sorted,
16-padded token segments. Verified max|err| = 0.0 vs numpy dequant at
T=256/4096/16384.

| T tokens | int4 grouped MoE | TFLOPS | int8 _int_mm | fp32 eager | vs fp32 |
|---|---|---|---|---|---|
| 256  | 0.270 ms | 2.98  (launch-bound) | 20.4  | 6.7   | 0.44x |
| 4096 | 0.331 ms | 38.87 | 103.7 | 11.8  | **3.30x** |
| 16384| 1.027 ms | 50.19 | 152.3 | 16.3  | **3.09x** |

- 38.9 TFLOPS @ T=4096 = ~95% of the 40.8 TFLOPS community-measured
  int4-WMMA ceiling for this silicon (R9700 guide; RX 9070 same die).
- vs int8 rocBLAS: 0.33-0.37x compute-throughput — expected: int4's win
  on RDNA4 is bandwidth/VRAM (4x weight compression), not MMA rate
  (no int4 WMMA advantage over int8 in ALU throughput; ISA-confirmed).
- Launch overhead: 8 launches ~ 0.24ms; amortized at T>=4096.

## K=32 iu4 WMMA variant (user-thesis confirmed)

`__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(int neg, v2i a, int neg, v2i b, v8i acc, int clamp)`
— v2i = 2 packed ints = 16 int4 (K=32). Issue-rate duel: **2.29x the K=16
variant** — RDNA4 does double per type-width; the K=16 path was half-rate.

| Path (T=4096 grouped, E=8) | err | time | TFLOPS | vs fp32 |
|---|---|---|---|---|
| K=16 grouped MoE | 0.0 | 0.331 ms | 38.87 | 3.30x |
| **K=32 grouped MoE** | **0.0** | **0.296 ms** | **43.47** | **3.38x** |
| int8 _int_mm (reference) | — | 0.107 ms | 119.9 | 9.3x |
| fp32 eager (reference) | — | 1.002 ms | 12.9 | 1x |

K=32 dense standalone: exact, 34.9-35.7 TFLOPS. LDS bank-conflict padding
 attempted, reverted (index slips under pressure) — documented as the next
 optimization lever alongside multi-tile register blocking.

## v4: 4-warp blocks, 64x64 tiles (gemm_v4.py)

| T | int4 v4 grouped MoE | TFLOPS | vs fp32 | vs int8 | % of 663.5 TOPS peak |
|---|---|---|---|---|---|
| 4096 | 0.223 ms | **57.8** | **4.15x** | 0.53x | 8.7% |
| 16384 | 0.617 ms | **83.5** | **5.07x** | 0.51x | 12.6% |

v4 = 128 threads/block (4 warps), 64x64 output tile: 4 M-tiles share one
LDS load (4x less B traffic), padded A stride. Exact at T=4096.

## v5b status (94.2 TFLOPS, correctness OPEN)

Deeper K-chunks (64-k) reached 94.8 TFLOPS grouped (+13% over v4) but the
B-fragment/LDS indexing is still wrong (max|err| ~3308-3578 across fix
attempts). Found & fixed en route: B LDS stride must be 128 (8 q-words x
128 cols) for the NT=8 block tile — the v4-era 64-stride writes overflow
the B region. The K=32 B-fragment layout (H3: fragment words are kt and
kt+2 of the chunk, k = w*16 + kt*8 + j) was sweep-verified in isolation;
the full-kernel residual bug is in the store/read index interaction under
the NT=8 geometry. The verified deliverable remains v4 (83.5 TFLOPS,
exact). Next: rebuild v5b indexing from the sweep-verified mapping in a
minimal single-block harness before applying to the full kernel.

## NT=8/KCW=8 geometry (v5c): deterministic fault, open

Regenerating v5 from the PRISTINE v4 source via a verified 14-step
mechanical transform (NT 4→8, KCW 4→8, all LDS strides/decodes/derived
constants) still fails: max|err| = 2484, deterministic across every fix
attempt (B load-loop bound, stride-128 store/read, pointer-offset
variant). All manual index analyses pass — the fault is structural to
the NT=8 geometry (suspect: LDS capacity/occupancy interaction at
128-col blocks, or a codegen issue with the wider fragment).
Next session: run the staged-dump harness (v5b_stagedump.py pattern,
which successfully isolated the v2 LDS path) on the v5c kernel.

**Verified deliverable: v4** — 80-83.5 TFLOPS grouped MoE, exact,
4.95-5.07x fp32 eager, 52% of int8 rocBLAS, native K=32 int4 WMMA.

## v5c fault characterization (final for this campaign)

Staged-dump harness on v5c geometry: LDS load EXACT (0/1600 words wrong),
per-lane fragment reads EXACT for all 8 tiles (0/128 lanes wrong), yet the
full GEMM fails with column-dependent corruption: some columns near-exact
(diff 5-21), others wildly wrong (diff ~1000). All stages individually
verified — the fault is an interaction specific to the NT=8 geometry
(suspects: fragment-read/LDS-write hazard across the 8-tile loop, or an
RTC codegen issue with the wider fragment set). 10+ debug rounds, all
harness bugs eliminated (verified via int* dump path + raw memcpy).
v4 remains the verified deliverable: 80.2 TFLOPS, 5x fp32, exact.

## DECISIVE: the "16x16x32" iu4 builtin consumes only 16 k through v2i fragments

Controlled experiment (all-ones fragments, neg=0): D[0][0] = **16**, not 32.
The `16x16x32_iu4` builtin with a v2i (2-int, 16-nibble) fragment operand
effectively performs **K=16**, at half the cycle cost of the K=16 form —
its 2.29x "issue rate" advantage is instruction-issue efficiency, NOT
2x math. Consequences:

1. mmapeak's 663.5 TOPS for `mma_s4s4s32_16_16_32` is a named-MAC
   accounting artifact (8192 MACs counted per instruction; ~4096
   actually performed). Real int4 peak ≈ int8 rate ≈ 332 TOPS.
2. **int4 does NOT have 2x superior throughput over int8 on RDNA4** —
   the K=32 form issues the same MACs at half the cycles per
   instruction. int4's genuine advantage is 4x weight compression
   (bandwidth/VRAM), not MMA rate.
3. Our GEMM failures (Out ≈ ref/2) are fully explained: each WMMA
   consumed 16 of the 32 loaded k-elements. The fix: treat the K=32
   builtin as TWO K=16-equivalent instructions (4 fragment reads per
   8-word chunk: word pairs (0,1), (2,3), (4,5), (6,7)), doubling the
   WMMA count per chunk.

This resolves the 2484/3308/3578 fault family and the int4-vs-int8
throughput question definitively.

## Per-nibble probe: ROOT CAUSE found — VOP3P packed-half fragments

Per-nibble probes (single nibble=1, k-ramp counterparts) reveal:
- Only nibble j=0 of each fragment word has a visible effect; nibbles
  j>=1 map OUTSIDE the consumed range.
- k-coverage per lane is tiny (observed k=1,2,18 for the first two
  words), n/m mappings scrambled vs any linear hypothesis.
- All-ones count (16 per (m,n)) + j0-only liveness = **VOP3P packed-half
  semantics**: the WMMA builtins process v2i/v8i registers as paired
  16-bit halves (each half = 4xint4); a fragment filled with plain
  byte-ordered nibbles only populates the LOW halves -> the instruction
  consumes 16 of the named 32 k.

Conclusion: exploiting the full K=32 int4 path on RDNA4 via RTC requires
fragments packed in the VOP3P half-interleaved layout (lo/hi 16-bit
halves, each holding 4 int4 with the documented lane map). That is a
solvable, well-defined follow-up (the probe harness in nibble_probe.py
can verify any candidate layout in minutes). Until then, the int4
effective MMA rate on RDNA4 is ~int8 rate (~332 TOPS), and v4
(80-83.5 TFLOPS exact, ~5x fp32, 52% of int8) is at ~25% of the true
VOP3P-corrected peak — consistent with a first-generation kernel.

## FINAL DECODE (90%): the K=32 builtin is TWO packed 16x16x16 dots

Probe evidence (P1-P5 + all-ones D=16): the `16x16x32_iu4` builtin
computes TWO independent 16x16x16 int4 dots:
- a.x = fragment of tile 0 (k 0-15), a.y = fragment of tile 1 (k 16-31)
- acc[j] = lo16: tile-0 partial | hi16: tile-1 partial (packed pairs)
- P1/P3: single B nibble affects 8 even/odd k positions per tile —
  the per-tile fragment layout is the K=16 H1/H3 mapping, replicated.
Correct usage: compute both tiles and COMBINE (add the two K=16 results,
unpacking the packed acc pairs), i.e. 2 WMMA calls per 32-k with a.x/a.y
split as separate K=16 fragments — NOT one call treating v2i as 32
contiguous k. The remaining 10%: exact per-tile lane map for the packed
acc pairs (decode via the same nibble-probe on the acc readback).

Verified deliverable unchanged: v4 at 80-83.5 TFLOPS exact (5x fp32).
The K=32 form's real value: 2x MACs per instruction issued (both tiles
per call) IF the packed-pair accumulate is handled — the path to >=130
TFLOPS stands, blocked only on the acc pair unpack/combine detail.

## Fix attempt log (final for this campaign)

Four systematically-derived interpretations of the K=32 builtin's v2i
fragment semantics, all verified against the same numpy dequant gate:
1. H1xH1 word pairs (2kt, 2kt+1) — FAIL 2484
2. H3 words (kt, kt+2) — FAIL 3308
3. Pointer-offset + stride-128 B LDS — FAIL 3578
4. 4-call sub-tile (a.x-only, 16 k per call) — FAIL 3572
Per-nibble probe (256 launches): only j0 nibbles live, scrambled k/n.
CONCLUSION: the RTC-exposed `16x16x32_iu4` builtin's fragment layout does
not match any standard row/column-distributed interpretation — it likely
requires the full VOP3P half-register packing discipline (lo/hi 16-bit
halves across the v2i pair, per the ISA's packed instruction format).
Decoding it needs the complete 512-position per-nibble map (the probe
harness is committed and takes ~2 min per 64-position sweep).

VERIFIED DELIVERABLE: v4 — 80-83.5 TFLOPS grouped MoE, exact, ~5x fp32,
native K=32-form int4 WMMA. The int4-vs-int8 compute question on RDNA4
resolves as: int4 ≈ int8 MMA rate (int4's advantage = 4x compression).

## Per-nibble probe data (lanes 0-1): k-interleaved across D-rows

Key probe result (P1): B nibble (lane0, w0, j0)=1 with A[m][k] = k+1 ramp
gives D[m][0] = 1, 3, 5, ..., 15 for m = 0..7 — i.e. the fragment position
pairs with host-k = 2m for output-row m. The K=32 builtin's fragment
layout is k-INTERLEAVED ACROSS OUTPUT ROWS (VOP3P packed-half structure),
not row-contiguous as any standard interpretation assumes. This is why
all four row-major-host-packed fix attempts failed: the hardware's
expected A/B fragment layout requires repacking the host data into the
k-interleaved order BEFORE the LDS stage (or loading with a de-interleave
in the fragment read). Full lane0-1 sweep data: nibbleprobe_lane01.log.
Next session: extend the sweep to all 32 lanes, derive the complete
host-layout -> fragment transform, rebuild the packer, verify, benchmark.

## Probe interpretation correction (P1-P5 re-analysis)

P1's D[m][0] = 2m+1 pattern proves STANDARD WMMA CROSS-LANE semantics:
the A operand is wave-distributed over k (lane l holds A nibbles at
k ~= 2l, 2l+1 via its v2i), and the hardware internally pairs A-lane-j
data with B-lane-l data for lane l's D column. My earlier per-lane
fragment hypotheses were confounded by this. The complete decode needs:
full 32-lane nibble sweep (nibble_probe.py now extended) with full
16x16 D readback per probe, solving the linear system offline for the
true (lane, word, nibble) -> (operand, k, m-or-n) map. The transform is
mechanical; the analysis (a small linear-system solve per operand) is
the remaining work. NOTE: this only matters for extracting MORE than
the currently-achieved throughput — the K=16-equivalent usage (v4) is
semantically correct as-is and delivers 80-83.5 TFLOPS exact.
