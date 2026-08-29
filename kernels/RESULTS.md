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
