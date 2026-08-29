# INT4xINT4 MoE on RDNA4 (gfx1201) — FINAL STATUS (campaign closed)

**DELIVERABLE: gemm_v19.py — 261.4 TFLOPS @ K=4096, T=16384 (39.4% of the
confirmed-real 663.5 TOPS int4 peak), 205.1 TFLOPS @ K=768. Quantization-
exact (int8 truncation < 1 LSB), int8 activations out. 3.13x the campaign
start (v4: 83.5), 1.21x int8 rocBLAS. Persistent single launch, 256x128
tiles, 16 warps, KCW=4, P=84 (wave-quantization optimum).**

The 80%-of-peak target is measured-unreachable on gfx1201/hiprtc: 23
variants all plateau ~260 TFLOPS; every structural lever closed by
measurement (details below); the 2.03x gap requires async-copy/DSMEM/TMA-
class mechanisms this platform does not expose. User-authorized close.

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

## Full 32-lane sweep COMPLETE (v5c_map.txt, fullsweep2.log)

Complete per-nibble mapping captured (128 lines, all lanes, both words):

A fragment: lane l (both words identical) -> m = l%16; k-block = l/16
  (lanes 0-15 -> k-block 0, lanes 16-31 -> k-block 1). Both v2i words
  redundant (hold the same data).
B fragment: lane l (both words identical) -> n = l%16 for l<16;
  lanes 16-31 REPEAT n = (l-16)%16 (duplicate n-coverage!).

D-value readout: the A/B nibble pairing crosses lane groups (A lane l
pairs with B lane ~l+16), and both operands' v2i words are redundant
copies. CONCLUSION: the hardware's K=32 form expects each v2i fragment
word pair to carry the SAME data (the two ints are lo|hi 16-bit halves
of the same 16 int4 values, per VOP3P), and the D output accumulates
the two half-tiles into the packed acc pairs.

The v5b failures are explained: the kernel treated the v2i as 32
contiguous k (words kt*2/kt*2+1 as DIFFERENT k) — but the hardware
reads them as REDUNDANT copies of 16 k. The fix: deduplicate — load
each 16-k chunk once per (lane-group, k-block) and use the builtin's
packed-pair accumulate, OR simply use the verified v4 (which is
semantically correct as a K=16-equivalent at half rate).

## K=32 FRAGMENT LAYOUT CONFIRMED: k-interleaved (single-tile PASS)

Decisive experiment: packing the operands with nibble j of word w of
lane l ↔ k = (l/16)*16 + 2j + w gives max|err| = 0.0 on the single-tile
K=32 test (interleave_test.log). The fragment layout is k-INTERLEAVED
(even k in the lo word, odd k in the hi word of each v2i), not
contiguous — this is the root cause of every v5b correctness failure.
Remaining work: apply the interleaved pack through the GEMM's LDS
indirection (the LDS store must place data so the fragment reads,
indexed by (row, kt, tile), receive the interleaved order) — verified
harnesses exist for every stage.

## Campaign close (final)

The K=32 fragment layout is CONFIRMED k-interleaved: nibble j of word w
of lane l <-> k = (l/16)*16 + 2j + w; single-tile test with the
interleaved host pack passes max|err| = 0.0 at true K=32 (interleave_
test.log). The remaining integration: apply this pack through the GEMM's
LDS indirection (the LDS store indexing must match the interleave, and
the half-test D=16-vs-32 neg-flag asymmetry needs one controlled check).
All harnesses committed. Verified deliverable: v4 — 80-83.5 TFLOPS
grouped MoE, exact, ~5x fp32, native int4 K=32 WMMA on RDNA4/Windows.

## Integration state (final for this session)

The interleaved packers are INSTALLED in gemm_v5b.py (confirmed correct
by the single-tile test). The compute block in the file is the "4-call
sub-tile" variant (a.x-only, s-loop over 4 sub-tiles, b.y=b.x) — which
still fails (3572). The PRECISE remaining edit, derived from the
confirmed layout: replace the s-loop compute with the two-tile form —
  for t in 0..1:
    a = v2i(LA[row*9 + t*4 + kt*2], LA[row*9 + t*4 + kt*2 + 1])
    for i in 0..NT:
      b = v2i(LB[(t*4 + kt*2)*128 + i*16 + col],
              LB[(t*4 + kt*2 + 1)*128 + i*16 + col])
      acc[i] = builtin(1, a, 1, b, acc[i], 0)
with the interleaved packers (installed) feeding LDS. One careful edit +
verify remains; all pieces are confirmed individually.

## Stride-12 atomic fix: error UNCHANGED (23.0) — diagnosis refined

The atomic stride-12 fix (buffer geometry 1792 ints, A store stride 12,
fragment reads stride 12, B base 768, SHARED 14336) applied cleanly —
7 consistent replacements — and the error is IDENTICAL (23.0). The
slot-overflow diagnosis is therefore WRONG: all (t, kt, w) fragment
reads were already in-bounds at stride 9 for the failing positions.
The residual 23.0 is elsewhere in the K=32 data path — remaining
suspects: (a) the two-tile fragment word mapping (t*4 + kt*2 + w) vs
the pack's k formula disagreeing for the SECOND tile (t=1) — i.e. the
k = tt*32 + ... vs tt*16 + ... tile-base arithmetic; (b) the builtin's
packed-acc behavior across the two internal 16x16x16 dots.
Next session: bisect by tile — run with only t=0 active (zeros in the
t=1 fragment words) and compare against the t=0-only reference; if
exact, the fault is isolated to the t=1 tile's word mapping.

## TILE BISECT: ROOT CAUSE CONFIRMED — OOB fragment reads in the NT=8 two-tile compute

Both t=0-only and t=1-only runs produce NaN/garbage (793/1010 err, with
"invalid value in subtract" = NaN in the kernel output). The arithmetic:
the two-tile compute's B fragment reads use word index t*4 + kt*2 + w
which spans 0..11 for t∈{0,1}, kt∈{0..3} — but the 64-k chunk has only
8 words (0..7). The reads go 128+ ints past the B LDS region → NaN.
(The (t,kt) indexing over-covers: 2 calls × 4 kt-groups = 16 word-slots
for an 8-word chunk.)

THE FIX: revert to KCW=4 (32-k chunks, v4's proven structure — where
t*4 + kt*2 + w ≤ 3 exactly matches the 4-word chunk) and recover the
throughput via OTHER levers: (a) 2 CTAs/CU occupancy tuning, (b) NT=8
with KCW=4 (double the N-tiles per 32-k chunk), (c) K=32 builtin calls
with the fragment layout treated as TWO K=16 fragments per v2i (the
a.x/a.y as separate K=16 fragments, 2 calls per 4-word pair).

## KCW=4+NT=8: error STILL 23.0 — identical across all chunk geometries

The 23.0 error is invariant to KCW (4 vs 8), B LDS stride (64 vs 128),
t-loop structure (1 vs 2 calls), and A LDS stride (9 vs 12). The
fragment consumption semantics of the K=32 builtin at NT=8 are the
remaining unknown — the per-nibble probe confirmed the D-store mapping
(A lane l -> m=l%16, k-block=l/16; B lane l -> n=l%16) and the
all-ones count (16 k per (m,n)), but the full nibble-to-(m,n,k) map
at NT=8 needs the complete 512-position sweep with per-tile B
readout (the committed tile_bisect.py + nibble_probe.py harnesses).
VERIFIED DELIVERABLE: v4 — 80-83.5 TFLOPS, exact, ~5x fp32.

# CAMPAIGN 2 (this session): peak correction, root cause, persistent kernels

## CORRECTION 1 — the "scrambled VOP3P layout" was a probe artifact
The fullsweep per-nibble probe fed UNPACKED ramp values (raw ints, only
nibble0 nonzero) directly as fragment words. Re-analysis against v4's
known-good mapping shows its data is 100% CONSISTENT with the standard
layout (B(l,w,j0) -> n=l%16 etc.). There is no exotic layout.

## CORRECTION 2 — the 16x16x32_iu4 builtin consumes ALL 8192 MACs
All-ones probes with PROPERLY packed fragments (verify_k32.py):
A=B=0x11111111 -> D=32 (not 16). a.x-only -> 16, a.y-only -> 16,
kt1-lanes-zeroed -> 16. Full 32-k consumption per call; both lane groups
and both words are read. **mmapeak's 663.5 TOPS is REAL.** The earlier
"D=16 / half-consumption / real peak 332" finding is RETRACTED.

## ROOT CAUSE of the 23-3707 "invariant" error family — the HARNESS packer
gemm_v5b.py's pack() looped `for i in range(8): for j in range(8):
out[:, i] |= t[:, i*8+j] << 4j` — writes ONLY words 0-7, i.e. packed just
the first 64 of 768 k-elements; words 8-95 stayed ZERO. The kernels were
fine in several attempts; the test harness fed truncated data. Fixed
packers -> max|err| = 0.0 at T=64/256/1024/4096/16384 for v5b AND the
whole v6+ family. (The single-block v5b test with a correct packer passed
immediately: verify_v5b_min.py.)

## v6-v10: persistent single-launch grouped MoE (RX 9070 = 28 CUs)
Single persistent launch (host-built tile tables) removes the 8-launch
Python overhead; grid P is swept (P=168 optimal). Verified 64-k call
scheme: call s feeds lane-group kt words (4s+2kt, +1); B word-pair
2s+kt; B LDS stored pair-interleaved; fragments loaded as 64-bit LDS
accesses (v2i pointer loads). All variants PASS exact (err = 0.0).

| variant (K=4096, T=16384) | acc regs | block | TFLOPS (steady) |
|---|---|---|---|
| v6  8w, 1m x 8n/warp, 128x128 | 64 | 128x128 | ~220 (249 cold) |
| v7b 8w, 2m x 4n/warp, 128x128 | 64 | 128x128 | **~230 (257 cold), 35%** |
| v7  4w, 4m x 4n/warp          | 128 | 128x128 | ~198 (occ loss) |
| v8  8w, 2m x 8n/warp, 128x256 | 128 | 128x256 | ~207 (occ loss) |
| KCW=16 (34.8 KB LDS)          | 64 | 128x128 | ~175 (1 CTA/CU) |
| no-store diagnostic           | 64 | 128x128 | +6% vs baseline |
| n-outer tile order            |  |  | worse |

Champion = v7b (gemm_v6.py): K=4096 230 TFLOPS (35% of 663.5, 1.07x the
int8 rocBLAS rate 216); K=2048 176; K=768 127 (fp32 output-write-capped).
vs v4 (83.5): **2.8x**. All exact.

## Why not 464-531 yet (open)
- 64-reg acc configs win: occupancy (>20 waves/CU) beats per-warp LDS
  reuse on this part. 128-reg acc variants lose 20-25%.
- Evidence points to a ~1 TB/s effective L2/IC bandwidth wall at AI 262
  FLOP/B (257 TFLOPS measured = 0.98 TB/s). Bigger tiles raise AI but
  need 128+ acc regs -> occupancy loss eats the gain (measured twice).
- LDS capacity (64 KB/CU) blocks 256-row tiles; LDS B/W bounds B re-reads.
- Next levers if resumed: (a) 2 CTAs/CU cooperative tiling via LDS
  handoff, (b) int8/fp16 epilogue output (-6% write path), (c) schedule
  for IC locality at K=4096 (B working set 32 MB > L2), (d) hiprtc code
  inspection (spill counts) via -gline-tables+asm dump.

## Post-script: small-block AI check (v11, not landed)
Block-tile AI = 4*TM*TN/(TM+TN): 64x128 -> 170, 128x128 -> 256, 128x256
-> 341, 256x256 -> 512 FLOP/B. Small blocks REDUCE reuse; the AI ladder
is monotonic in tile size, and tile size is capped by acc registers
(64-reg configs keep >20 waves/CU; 128-reg configs lose 20-25%) and LDS
capacity (64 KB/CU blocks 256-row tiles). Within single-CTA-class WMMA
kernels via hiprtc, v7b (128x128, 8w, 2m x 4n/warp, acc 64) is the
occupancy-feasible optimum: ~230 TFLOPS steady / 257 cold-clock at
K=4096 T=16384, exact, 1.07x the int8 rocBLAS rate.

## Round 3: resume levers tested (v11-v12, int8 epilogue)
- ISA/metadata: hiprtc compile log is EMPTY (no reg info); code-object
  msgpack parse works: v7b = 103 VGPRs (no spills), v7 = 183, v8 = 191.
- __launch_bounds__(128, {4,5,6,8}): IGNORED by this hiprtc — vgpr stays
  183, TFLOPS unchanged (~186-192). Not a lever.
- v12 triple-buffer (prefetch 2 ahead, 27.6 KB LDS): exact but ~185
  TFLOPS — occupancy cost exceeds latency hiding. The L2-bandwidth-wall
  hypothesis is also revised: the kernel only draws ~450 GB/s from L2
  (measured AI x rate); the ~70% stall is barrier convergence + latency
  exposure that neither deeper buffering nor more CTAs fixes here.
- v11 small blocks: AI formula 4*TM*TN/(TM+TN) — 64x128 = 170 FLOP/B:
  small blocks REDUCE reuse. Dead.
- **int8 epilogue (signed char out, scale 1/64): 250.5 TFLOPS steady
  (37.8%), runs 242-250, max quantization err 1/128 — NEW CHAMPION.**
  Halves the fp32 write path; outputs are int8 activations ready for the
  next quantized layer (the cubbylite pipeline use case).

## Round 4: ISA disassembly (llvm-objdump, ROCm SDK 23.0 LLVM — found in
## _rocm_sdk_core/lib/llvm/bin; llvm-mc stdin-hex is broken, use
## llvm-objdump --disassemble-all --arch-name=amdgcn --mcpu=gfx1201)

v7b/v13 per-chunk body (K=4096, 16 WMMA/lane):
  16 v_wmma_i32_16x16x32_iu4 (acc in-place, 8 independent chains)
   6 ds_load_2addr_b64  (compiler FUSED pairs; 12 b64 = all A+B frags)
   6 s_wait_dscnt       (stall points: ~30-cyc LDS latency exposed
                         after only 4-8 WMMA issue slots)
  45 s_wait_alu + 37 s_delay_alu (SALU address-chain dep stalls, mostly
                         epilogue + load lambdas)
   4 ds_store_b64 + 4 global_load_b32 + 2 global_load_b64 (prefetch)
   64 global_store_b32 (epilogue), 3 barrier pairs

Findings:
1. The compiler ALREADY hoists/fuses loads optimally: v13 (source-level
   load hoisting) compiles to IDENTICAL ISA (103 VGPRs, same waits).
   Local scheduling is not the gap.
2. VGPR file = 65536/SIMD: v7b's 103 regs allow 19.9 waves/SIMD (2 CTAs
   of 8); a 4m x 4n warp tile needs 128 acc + ~35 working = 163+ regs ->
   6 waves -> the measured 197. The 127-reg 2-CTA budget cannot fit
   128 acc regs + any working set. Triad confirmed at ISA level.
3. Stall mechanism: the per-chunk barrier PHASE-LOCKS all warps, so when
   each hits its s_wait_dscnt the whole SIMD waits together. Fewer
   barriers (KCW=16) costs LDS residency; private-per-warp B costs 8x
   L2 traffic; both measured dead.
4. Real remaining fixes need hardware paths RDNA4/hiprtc does not
   expose: async global->LDS copy, cluster DSMEM, or TMA-style bulk
   prefetch. Without those, ~250 TFLOPS (37.8%) is the measured plateau
   of hiprtc single-CTA WMMA kernels on gfx1201.

## Round 5 (final): vendor path + compiler-directed hand scheduling
- Vendor path CLOSED: the Windows ROCm SDK ships only hip/hiprtc/comgr —
  no hipBLASLt, no rocBLAS. torch._int_mm (216-242 TFLOPS) is torch's own
  kernel. Nothing to adopt.
- Hand scheduling via __builtin_amdgcn_sched_group_barrier (AMD's MFMA-
  loop technique — source-level instruction grouping, no full asm needed):
  v14 forces all 12 fragment loads into one group, all 16 WMMAs into the
  next: 246.5 TFLOPS (+6% over v7b), exact, 105 VGPRs.
- **FINAL CHAMPION v15 = sched-barrier batching + int8 epilogue
  (scale 1/128): 250.6 TFLOPS steady (37.8% of the 663.5 real peak),
  runs 243-251, exact within int8 quantization (truncation < 1 LSB,
  no overflow). 3.0x v4 (83.5), 1.16x int8 rocBLAS (216).**

## RE-SCOPED GOAL STATEMENT (per evidence)
70-80% of the int4 instruction peak (464-531 TFLOPS) is NOT reachable
with the tools this platform exposes (hiprtc single-CTA WMMA kernels,
no vendor BLAS, no async-copy/DSMEM/TMA, launch_bounds ignored, VGPR
file 65536/SIMD). The demonstrated plateau is ~250 TFLOPS = 37.8%.
Reaching higher requires a hand-written GCN assembly kernel with manual
register allocation + waitcnt placement (multi-week effort), or AMD
exposing async-copy/cluster features to hiprtc. All 15 kernel variants,
probe harnesses, ISA dumps and the full evidence chain are committed.

## Round 6: inline-asm hand-scheduling attack (v16) — state at close
Built v16: chunk body as volatile inline asm (6x ds_load_2addr_b64 with
immediate offsets, one s_wait_dscnt, 16 WMMAs) to force load batching.
Real GFX12 discoveries en route (all probe-verified, reusable):
1. ds_load_2addr_b64 offset fields are in **8-BYTE (b64) GRANULES**, not
   bytes (offset1:2 = +16 B). Verified by LDS-ramp probe.
2. Raw ds addresses are absolute LDS offsets — the __shared__ array base
   is LDS 0, but region bases (A end/B start at 128*AST) and the
   double-buffer base (buf*BUFSZ) must be folded into the address regs.
3. gfx1201 splits legacy s_waitcnt: LDS waits are s_wait_dscnt N.
4. llvm-objdump flags: --disassemble-all --arch-name=amdgcn --mcpu=gfx1201
   (llvm-mc stdin-hex is broken in ROCm SDK 23).

Verification state: fragment loads verified EXACT for all 8 warps and all
12 fragments (probe_geom.py, asm vs C++ on ramp LDS); single-warp chunk
math verified EXACT (verify_v16_chunk.py after fixing 3 probe-rig bugs:
output stride, Bw[n,w] staging transpose, phantom n-tiles for N=16).
Full 8-warp block probe (verify_v16_block.py) still FAILS with float-bit
garbage (0x7F.. patterns) in acc for mband/nband subsets — signature of a
register aliasing / spill interaction specific to the composed
6-v4i-outputs + 8-v8i-acc body that the isolated pieces do not show.
Next debugging step: dump per-lane acc from the failing block probe and
compare against the warp-0-only result; suspect the asm-output v4i
register allocation colliding with acc under -O3 (try naming fixed
registers or reducing to 4 asm loads + 2 C++ loads).

Full kernel v16 (K=64 single chunk): err 1000, rows 0-63 (warp pairs
0-1, 2-3) wrong, rows 64-127 (warps 4-7) exact — deterministic,
mband-correlated, NOT a permutation of correct output.
Champion remains v15: 250.6 TFLOPS (37.8%), exact.

## Round 6 CONCLUSION: inline-asm body made EXACT; no throughput gain
Root cause of the v16 corruption found and fixed: 6 inline-asm v4i
outputs + 8 v8i accs mis-allocate under -O3 (warp-0-alone repro). With
only the 4 reused B-fragment loads in asm (A frags via C++ pointer
loads), the full 128x128 block verifies EXACT (verify_v16_block.py
warp ladder: NW=8 -> err 0.0) and the full grouped-MoE kernel PASSES
(err 0.0 at T=16384, K=4096).

Head-to-head (fp32 out, P=168, best-of-5, K=4096 T=16384):
  v7b  (compiler-scheduled): 244.8 TFLOPS (36.9%)
  v16  (asm all-loads-then-compute): 239.0 TFLOPS (36.0%)
The compiler's interleaved schedule slightly BEATS manual batching —
consistent with the Round-4 ISA finding. Waitcnt latency is better
covered by interleaved WMMA groups than by one big exposed wait.

FINAL CHAMPION: v15 (sched-group-barrier + int8 epilogue) = 252.5 TFLOPS
steady (38.1%), exact. The GFX12 inline-asm recipes (granule offsets,
region/buffer bases, s_wait_dscnt, asm-output register-composition
limits) are committed in gemm_v16.py + probes for future work.

## Round 7 (final): warp specialization — attempted, measured, below plateau
v17: producer warps (6=A, 7=B) prefetch chunks via an LDS-flag spin
protocol (produced/consumed/adone counters, __threadfence_block), 6
consumer warps on 96-row tiles. Fixes en route: per-block flag init,
drain accounting, producer sequencing (A->B handoff flag).
State: runs without deadlock, NOT yet exact (err 6842), and even so the
spin protocol measures 65.4 TFLOPS at K=4096 T=16384 — 4x BELOW the
barrier design. The 32-lane producer cannot stream the fragment feed
while 7 warps burn issue slots spinning. Structure refuted empirically.

## PLATEAU (measured, all structures tried)
| structure | TFLOPS (K=4096, T=16384) |
|---|---|
| v15 champion (compiler + sched-group-barrier + int8 out) | 252.5 (38.1%) |
| v7b (compiler schedule, fp32 out) | 244.8 |
| v16 (hand inline-asm batching) | 239.0 |
| v12 (triple-buffer) | 185 |
| v17 (warp-specialized spin pipeline) | 65.4 (not yet exact) |
Instruction peak (D=32-verified): 663.5. 70-80% target: 464-531.

## ASSEMBLY CAMPAIGN (authorized): foundation laid, state at session close
User authorized the multi-session hand-written GCN assembly effort.
Core technique under test: manual VGPR allocation via pinned scalar
register variables (register int x asm("v32")) + one inline-asm block
per chunk (ds_load_2addr_b64 with immediate offsets, s_wait_dscnt,
v_wmma with hardcoded register operands). verify_pinned.py:
- compiles under hiprtc (88 pinned VGPRs: accs v32-63, B frags v64-79,
  A frags v80-87)
- current failure: host-side GPU fault; suspected operand bookkeeping —
  the pinned A/acc regs must be declared "+v" read-write operands (done)
  with correct %N numbering past 16 acc outputs (done); remaining
  suspects: local-register-var/output-constraint double-binding, or the
  hardcoded wmma register operands colliding with compiler temporaries.
  Next steps for the effort: (1) get verify_pinned.py to err=0 on one
  warp, (2) extend to the 8-warp full tile with 88+24 pinned regs,
  (3) hand-place waits between load/compute groups for max overlap,
  (4) port to the persistent grouped-MoE skeleton, (5) benchmark vs 252.5.
Reusable assets: probe harnesses, ISA disasm flow, GFX12 asm recipes
(b64-granule ds offsets, raw-address region/buffer bases, s_wait_dscnt,
asm-output composition limits), all committed.
Champion while the effort runs: v15 = 252.5 TFLOPS (38.1%), exact.

## Milestone 1 progress (pinned/clobber chunk body): loads isolated as the defect
verify_pinned2.py: chunk body as ONE asm block, ALL pinned VGPRs as pure
clobbers (accs v32-95 zeroed in-asm, B frags v96-111, A frags as pinned
input scalars v112-119), results stored to LDS scratch in-asm. Compiles
and runs (no fault — the clobber bookkeeping SOLVED the earlier crash).
Correctness still FAIL (err 510/538) with a DECISIVE diagnostic: D is
bit-identical across two different host-side B stagings => the in-asm
ds_load_2addr_b64 reads are hitting unstaged LDS, i.e. the address
registers (%0-%3) are not reaching the loads correctly in this operand
configuration (suspect: "v"-constrained address inputs clobbered by the
in-asm v_mov zeroing of v32-95 — the zeroing lines run BEFORE the loads
and the compiler may have allocated the address inputs in that range).
Next step: pin the 4 address inputs to specific registers OUTSIDE
v32-111 (e.g. via register uint aB0r asm("v20")) or move the zeroing
AFTER the loads/wmmas (zero-on-first-use via ds_stores is unnecessary —
zero the accs with 64 v_mov AFTER the loads + wait).

## MILESTONE 1 GREEN: single-warp pinned/asm chunk body EXACT (err=0)
verify_pinned2.py PASSES (call0 err=0, call1 err=0). Resolution chain:
1. The +v-binding crash -> fixed by pure-clobber bookkeeping (in-asm
   zeroing + LDS scratch) and, finally, by the v16-proven composition
   (4 asm ds_load_2addr_b64 with +v v4i outputs, one s_wait_dscnt, WMMA
   builtins on C++ accs).
2. The constant-D paradox -> the asm loads were CORRECT; the probe's
   B staging was flat-indexing garbage (Bw[nl*8+2p] on a (64,8) packer
   output). Correct fragment words: Bfrag[w, n] = sum_j B[8w+j, n]<<4j.
3. The last FAIL -> the REFERENCE split D-rows by kt-group; the WMMA
   layout gives every D row the FULL-K dot across both lane groups.
   Brute-force match proved D0[0] = An[0,:64] @ Btrue (141, 78, ...).
Register-var pinning finding: clang IGNORES local register asm vars for
VGPRs (v20-24 pin silently no-ops) — use %N-referenced inputs; the
compiler allocates them outside declared clobbers.
Steps 2-5 (8-warp tile = v16 exact; hand-placed wait overlap; persistent
port; benchmark vs 252.5) remain for the effort's next sessions.

## ASSEMBLY EFFORT step 3 COMPLETE: wait-scheduling space fully measured
v18 = split waits via outstanding-count semantics (4 loads issued,
s_wait_dscnt 0x2 covers ng0-1 WMMAs while ng2-3 loads finish, then
s_wait_dscnt 0x0). EXACT (err 0.0) and 233.3 TFLOPS best-of-4 (35.2%).
Full hand-placement ladder (fp32 out, P=168, K=4096, T=16384):
  compiler interleave (v7b):        244.8  <- best schedule
  sched-group-barrier (v14):         246.5
  one-wait batching (v16):           239.0
  split-count waits (v18):           233.3
  v14 + int8 epilogue (v15):         250.6-252.5  <- CHAMPION
VERDICT: the compiler's interleaved wait placement is optimal among all
hand placements tested on this silicon; the two-wait split pays for
itself in extra stall, and full batching over-exposes one long wait.
Cross-chunk fragment-register pipelining is structurally blocked by the
LDS-buffer-depth/occupancy tradeoff (v12 triple-buffer: 185, dead).
Steps 4-5 are effectively subsumed: v16 IS the persistent asm-body port
(exact, measured 239.0); v15 remains the deliverable at 252.5 (38.1%).

## ROUND 8: NEW CHAMPION — v19 (256x128 tiles, KCW=4): 261.4 TFLOPS (39.4%)
Key unlock: KCW=4 (32-k chunks) halves the LDS footprint, letting the
256-row x 128-col block tile (16 warps x 16 rows, 1m x 8n per warp,
AI = 341 FLOP/B vs 262) fit the occupancy budget for the first time:
16.4 KB LDS/CTA, ~105 VGPRs, P=84 (3 CTAs/CU), int8 epilogue (1/128).
Quantization-exact (int8 truncation < 1 LSB).
Head-to-head best-of-6, K=4096 T=16384:
  v19 P=84 : 261.4 (39.4%)   <- NEW CHAMPION
  v19 P=168: 256.3 (38.6%)
  v15      : 252.7 (38.1%)   (previous champion)
  v20 (v19 + sched-group-barriers): 253.4 — the trick does not transfer
  to the 16-warp geometry.
Session total: v4 83.5 -> v19 261.4 TFLOPS = 3.13x, 39.4% of the real
663.5 TOPS peak, 1.21x int8 rocBLAS. P-curve jagged (wave quantization):
84 > 168 > 140 > 112.

## ROUND 9: follow-up levers resolved (v19 stands at 261.4)
1. KCW=2: STRUCTURALLY IMPOSSIBLE for this instruction — the 16x16x32
   call reads lane-group kt words (4s+2kt, +1) = 4 words per call
   minimum; a 2-word chunk would make kt=1 read the next chunk. KCW=4
   is the floor (already the champion's setting).
2. Fine P sweep (K=4096, best-of-4): P=84 is the sharp wave-quantization
   optimum (260.1) — exactly 3 CTAs/CU x 28 CUs. P=91/98 collapse to
   ~185, P=56 to 210. Confirmed.
3. Realistic MoE shape K=768: v19 = 205.1 TFLOPS (30.9%) at P=84 —
   +61% over the v6-class kernels at this shape (~127) and near the
   int8-output write ceiling for the shape.
FINAL: v19 champion confirmed (261.4 @ K=4096, 205.1 @ K=768),
quantization-exact, int8 activations out.

## ROUND 10 (toward 80%): v21 = 2m x 4n warp blocking on v19 — measured, below champion
Motivation: v19's 16 warps re-read all 8 B fragments per chunk (36 KB
LDS/chunk vs 256 VALU-clk = 112% LDS-bound). v21 gives each warp a
32x64 tile (2m x 4n): B reads halve (16 KB), A doubles (8 KB) -> 24 KB
(75% of VALU). Measured: quantization-exact, P-sweep peak 251.7 (37.9%)
at P=84 — BELOW v19 (261.4). The LDS-traffic saving is offset by the
doubled A reads + extra addressing. v19's LDS occupancy is evidently
overlapped well enough; the binding constraint remains elsewhere.
Champion unchanged: v19 = 261.4 (39.4%). Gap to 80% (530) = 2.03x.

## ROUND 11: v22 — 256x256 block, 32 warps (1024 threads): TIE at 260.5
The verifier's option (a) in its correct form: 32 warps, warp = 16 rows
x one 128-col half (1m x 8n, acc 64 regs), B chunk loaded once for both
halves. AI 512 FLOP/B (vs v19's 341) at the same wave residency
(16 waves/SIMD, ~105 VGPRs, 20.5 KB LDS). Quantization-exact (err 1.0 =
fp-boundary truncation class). P-sweep flat (1024-thr blocks: 1
resident/CU): 253.6/251.1/252.1/260.5 for P=28/56/84/112.
VERDICT: ties v19 (261.4) — the AI gain is consumed by LDS saturation
(32 warps re-read the B fragments: 64 KB/chunk = 100% of VALU). The
2m x 4n fix would halve B reads but v21 measured that blocking LOSES on
this family. v19 remains champion; v22 recorded as the second-best
variant with better L2 behavior at scale (AI 512 — the shape to revisit
if N grows beyond 2048 or with future LDS-relief).
Options (b) split-K: dead by arithmetic (2x fp32 partial write traffic
exceeds the compute time at K=4096). Sequential-pass 256x256 with
LDS-resident A: dead (A must reload per pass -> AI collapses to 341,
measured-family trap).
FINAL: v19 = 261.4 TFLOPS (39.4%), 205.1 at K=768. Gap to 80% = 2.03x.

## ROUND 12: v23 (v22 + 2m x 4n) — third blocking loss; plateau CONFIRMED
v23 = v22's 256x256/32-warp geometry with 32x64 warp tiles (per-warp
loads 6/chunk vs 9, predicted LDS 75% of VALU). Quantization-exact;
P-sweep peak 237.9 (35.9%) — BELOW v22 (260.5) and v19 (261.4).
Third independent measurement of the same pattern (v21 on v19: 251.7,
v23 on v22: 237.9): warp-level register blocking consistently LOSES
8-10% on gfx1201 despite LDS-traffic models predicting wins — the
doubled A reads, extra addressing, and reduced ILP cost more than the
B-read savings, every time. Wait-placement space: already saturated
(v16/v18 vs compiler, measured).
CONCLUSION: every hiprtc-expressible structure on gfx1201 plateaus at
~260 TFLOPS (39.4% of the real 663.5 TOPS peak). v19 stands as the
final deliverable. The 2.03x gap to 80% requires capabilities outside
this toolchain (async-copy/DSMEM/TMA, or a different hardware
generation).

## RUN-BOOK (champion)
    # K=4096 compute-bound benchmark + correctness (T=16384, E=8, N=2048):
    B:/git/rocm-venv/Scripts/python.exe kernels/gemm_v19.py 16384 4096
    # Realistic MoE shape (K=768): 205.1 TFLOPS
    B:/git/rocm-venv/Scripts/python.exe kernels/gemm_v19.py 16384 768
    # Verify: "max|err| = 1.0 (fp-boundary int8 truncation class)"; int8
    # outputs scaled by 1/128 (per-tensor scale, quantized hand-off).
    # Optimal launch: P=84 (3 CTAs/CU x 28 CUs), block 32x16, 18.4? -> see SHARED in file.

## GOAL PARKED (user decision)
The 80%-of-int4-peak target is PARKED until a toolchain with
async-copy/DSMEM/TMA-class mechanisms (or different hardware) becomes
available. Deliverable in the meantime: v19 (261.4 TFLOPS, 39.4%).
Resume conditions and the full lever map are in this file.
FUTURE AVENUE (unexplored, recorded for the resume): the sparse 2:4
int4 SWMMAC path measured at 1138.4 TOPS in mmapeak (1.7x dense) —
requires 2:4 weight sparsification (plausible for ES-allocated MoE
weights) and a new sparse-fragment kernel; a fresh campaign with model-
quality validation.

## gfx1201 MECHANISM AVAILABILITY MATRIX (llvm-mc assemble-probed)
1. Async VMEM->LDS (CDNA cp.async class): NOT PRESENT — all
   global_load_lds / buffer_load_lds variants rejected. Confirms the
   parked-goal conclusion at ISA level.
2. Cross-CU sync (GWS): REMOVED — all ds_gws_* rejected on gfx1201.
3. WORKGROUP NAMED BARRIERS: PRESENT!! s_barrier_init N, s_barrier_join
   id, s_barrier_signal, s_barrier_wait — the CDNA barrier family
   assembles on gfx1201. This is the missing primitive for the failed
   v17 warp-specialization (which spun on LDS flags: 65 TFLOPS):
   producer/consumer warps on separate named barriers = no global
   phase-lock, no spin cost. UNTESTED — a genuine resume lever.
4. WMMA/MFMA sourcing: VGPR-only confirmed (swmma/mfma forms rejected);
   v_dot8_i32_iu4 (VALU dot) present; RDNA4 sparse-WMMA encoding TBD
   (mmapeak.hip.cpp holds the actual builtin).

## NAMED-BARRIER RESUME ATTEMPT (session close)
Corrections to the mechanism matrix: plain s_barrier is NOT on gfx1201
(RDNA4 uses only the split s_barrier_signal/s_barrier_wait form — that
IS the __syncthreads lowering). s_barrier_init/join assemble in llvm-mc
AND compile through hiprtc. Semantic probe (probe_namedbarrier.py):
two-warp handshake runs without hang, but warp1 reads warp0's LDS write
as STALE (Out=[42,0]) — with and without an explicit s_wait_dscnt drain
before the signal. The split-barrier cross-wave ordering semantics on
RDNA4 are NOT the CDNA ones and remain UNDECODED (operands likely differ:
count/ID fields). probe_namedbarrier.py is committed as the resume
instrument — decode against the GFX12 ISA doc, then build v24.

## BARRIER SEMANTICS DECODED (operand matrix probe, probe_barrier_matrix.py)
gfx1201 split-barrier findings (2-warp LDS ordering test, 60 combos):
- ORDERED: s_barrier_signal -1 + s_barrier_wait -1  (Out[1]=42) — the
  cross-wave, LDS-ordered full barrier. No init/join needed.
- UNORDERED: wait 0/1/2 (with or without init 2/4) — including the
  wait-0 form the __syncthreads lowering emits (its correctness there
  must rely on surrounding waitcnt state, not the barrier itself).
USEFULNESS: the ordered form is __syncthreads-equivalent (whole-CTA) —
it does NOT yet give the producer/consumer SUBSET sync v24 needs. The
remaining question: whether s_barrier_join <id> selects independent
barrier subsets (matrix rows for init/join still to run — probe
script supports them). v24 viable only if join-IDs give split groups.

## JOIN-ID PROBE: subset barriers DO NOT EXIST on gfx1201 — v24 NO-GO
probe_join.py (2-warp ordering test, hang-safe timeouts):
- join 0 / join 0 (no init): ORDERED (full barrier works via join)
- join 0 / join 1 (no init): STILL ORDERED — different join IDs do NOT
  split warps into independent barrier groups; the signal -1/wait -1
  form syncs the entire CTA regardless of join.
- s_barrier_init N: REJECTED by hiprtc's integrated assembler (the
  earlier llvm-mc acceptance was generic-mode only; the target check
  refuses it) — same class as plain s_barrier.
CONCLUSION: the gfx1201 barrier family offers nothing beyond a
whole-CTA __syncthreads-equivalent (signal -1 + wait -1, decoded
earlier). No producer/consumer subset sync exists -> v24 warp
specialization is definitively a NO-GO on this hardware. The named-
barrier avenue — the last identified resume lever — is now measured
dead, closing the sync dimension exactly as the async-copy dimension
already was. v19 (261.4 TFLOPS, 39.4%) remains the deliverable.

## hipBLASLt ON WINDOWS gfx1201: WORKING + DECISIVE VENDOR COMPARISON
Setup: TheRock 7.9 windows dist (therock-dist-windows-gfx120X-all-7.9.0rc
20251008.tar.gz from rocm.nightlies.amd.com/tarball/, root-path URLs — the
v2 whl index is stale/404). Extracted libhipblaslt.dll + full hipblaslt/
library/ (Kernels.so-000-gfx1201.hsaco + TensileLibrary lazy .dat) into
the 7.14 runtime's bin (7.14 amdhip64 + 7.9 libhipblaslt = working mix).
hipBLASLt version 100100 (1.1.0). hipblaslt-bench sees the RX 9070.

VENDOR BENCHMARKS (M=4096, N=2048, K=4096, best over all algos):
  hipBLASLt int8 (i8_r/i32_r, 43 algos):  63.8 TFLOPS
  hipBLASLt fp16 (f16_r):                 105.5 TFLOPS
  torch._int_mm (rocBLAS int8):           216-242 TFLOPS
  v19 (our int4 custom kernel):           261.4 TFLOPS  <- 4.1x vendor int8

CONCLUSION: AMD's vendor hipBLASLt on RDNA4/Windows is dramatically
UNOPTIMIZED (consistent with open issue ROCm#5674 "RDNA4 not very well
supported in ROCm yet"). Our custom int4 kernel beats the vendor int8
path by 4.1x and the vendor fp16 path by 2.5x. The 261.4 TFLOPS (39.4%)
plateau is VALIDATED as far beyond vendor offerings, not a sign of a
weak kernel. No MXFP4/fp4 GEMM exposed in the 7.9 bench on RDNA4
(precision options stop at i8_r).
