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
| 16384 | (see commit log) | | | | |

v4 = 128 threads/block (4 warps), 64x64 output tile: 4 M-tiles share one
LDS load (4x less B traffic), padded A stride. Exact at T=4096.
