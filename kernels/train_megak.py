"""THE ASSEMBLED MEGAKERNEL: all four validated mechanisms in one launch.

  (1) lane stacking   — 32 lanes x 2048 rows (1 primal + 31 tangents) share
                        the same int4 weights (M=65536 GEMM);
  (2) epilogue fusion — int8-quantize fused into the GEMM epilogue;
  (3) CTA roles       — G gemm CTAs (persistent, contiguous tile slabs —
                        the Test-4 locality fix) + X memory CTAs running an
                        Adam-style update sweep on the PREVIOUS phase's
                        weight buffers (ping-pong: no race with the GEMM's
                        weights);
  (4) batched passes  — one big sweep per launch, no per-step launches.

Arms:
  baseline : v4 GEMM (fp32 out) + separate quant8 kernel + separate adam
             kernel — three serial launches.
  assembled: ONE train_megak launch (splits 160g+56m and 112g+112m).
Correctness: assembled int8 output must be BITWISE identical to the
baseline pipeline's; the Adam sweep must match a torch reference.
"""
import sys
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, ctypes, time
import numpy as np
import gemm_v4 as G4
import overlap_survivors as OS
HIP = G4.HIP

SRC = r"""
typedef unsigned int uint32_t;
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
#define NT 4
extern "C" __global__ void train_megak(
        const uint32_t* __restrict__ Ap, const uint32_t* __restrict__ Bt,
        const float* __restrict__ scale, signed char* __restrict__ Out,
        int M, int N, int Kw, int TilesN, int G_CTAS,
        float* w, float* m, float* v, const float* g,
        float lr, float b1, float b2, float eps, int n) {
    if (G_CTAS > 0 && (int)blockIdx.x < G_CTAS) {
        // ---- GEMM role: persistent, CONTIGUOUS tile slab per CTA ----
        extern __shared__ int lds[];
        int lane = threadIdx.x & 31;
        int warp = threadIdx.y & 3;
        int col = lane & 15, kt = lane >> 4;
        int total = (M / 64) * TilesN;
        int chunk = (total + G_CTAS - 1) / G_CTAS;
        int t0 = blockIdx.x * chunk;
        int t1 = t0 + chunk; if (t1 > total) t1 = total;
        auto load = [&](int kw, int buf, int mb, int n0) {
            int* LA = lds + buf * 576;
            int* LB = LA + 320;
            for (int w = threadIdx.y * 32 + lane; w < 64 * 5; w += 128) {
                int r = w / 5, q = w % 5;
                LA[r * 5 + q] = (q < 4) ? Ap[(mb + r) * Kw + kw + q] : 0;
            }
            for (int w = threadIdx.y * 32 + lane; w < 4 * 64; w += 128) {
                int q = w >> 6, nl = w & 63;
                LB[q * 64 + nl] = Bt[(kw + q) * N + n0 + nl];
            }
        };
        for (int tile = t0; tile < t1; ++tile) {
            int n0 = (tile % TilesN) * 64;
            int mb = (tile / TilesN) * 64;
            v8i acc[NT];
            for (int i = 0; i < NT; ++i) acc[i] = {};
            load(0, 0, mb, n0);
            __syncthreads();
            for (int kw = 0; kw < Kw; kw += 4) {
                int* LA = lds + (kw / 4 & 1) * 576;
                int* LB = LA + 320;
                int row_local = warp * 16 + col;
                v2i a; a.x = LA[row_local * 5 + kt * 2]; a.y = LA[row_local * 5 + kt * 2 + 1];
                for (int i = 0; i < NT; ++i) {
                    v2i b; b.x = LB[(kt * 2) * 64 + i * 16 + col];
                    b.y = LB[(kt * 2 + 1) * 64 + i * 16 + col];
                    acc[i] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, a, 1, b, acc[i], 0);
                }
                __syncthreads();
                if (kw + 4 < Kw) load(kw + 4, ((kw / 4) + 1) & 1, mb, n0);
                __syncthreads();
            }
            int rbase = (lane >> 4) * 8;
            for (int i = 0; i < NT; ++i)
                for (int j = 0; j < 8; ++j) {
                    float x = (float)acc[i][j] * scale[0];
                    Out[(mb + warp * 16 + rbase + j) * N + n0 + i * 16 + col] =
                        (signed char)(int)(x > 127.f ? 127.f : (x < -127.f ? -127.f : x));
                }
        }
    } else {
        // ---- memory role: Adam-style update on the PREVIOUS phase's
        //      weight buffers (ping-pong — no race with the GEMM) ----
        int tid = threadIdx.y * 32 + threadIdx.x;
        int g0 = (G_CTAS > 0) ? G_CTAS : 0;
        int mc = blockIdx.x - g0;
        int MC = gridDim.x - g0;
        for (int i = mc * 128 + tid; i < n; i += MC * 128) {
            float mi = b1 * m[i] + (1.0f - b1) * g[i];
            float vi = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
            m[i] = mi;
            v[i] = vi;
            w[i] -= lr * mi / (sqrtf(vi) + eps);
        }
    }
}
extern "C" __global__ void adamk(float* w, float* m, float* v, const float* g,
                                 float lr, float b1, float b2, float eps, int n) {
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = i0; i < n; i += stride) {
        float mi = b1 * m[i] + (1.0f - b1) * g[i];
        float vi = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
        m[i] = mi;
        v[i] = vi;
        w[i] -= lr * mi / (sqrtf(vi) + eps);
    }
}
"""

if __name__ == "__main__":
    torch.manual_seed(0)
    LANES, B = 32, 2048                      # 1 primal + 31 tangent lanes
    M = LANES * B                             # 65536 stacked rows
    N, K = 2048, 4096
    Kw = K // 8
    scale = torch.full((1,), 1.0 / 128.0, device="cuda")
    print(f"assembled config: {LANES} lanes x {B} rows = M={M}, N={N}, K={K}",
          flush=True)

    A = torch.randint(-8, 8, (M, K), device="cuda", dtype=torch.int32)
    Bm = torch.randint(-8, 8, (N, K), device="cuda", dtype=torch.int32)
    Ap = G4.pack(A).contiguous()
    Bt = G4.pack_transposed(Bm).contiguous()
    Outf = torch.empty((M, N), device="cuda", dtype=torch.float32)
    O8a = torch.empty((M, N), device="cuda", dtype=torch.int8)   # baseline
    O8b = torch.empty((M, N), device="cuda", dtype=torch.int8)   # assembled

    na = 128 * 1024 * 1024                    # Adam arrays: 4 x 512MB
    LR, B1, B2, EPS = 1e-3, 0.9, 0.999, 1e-8
    st = dict(device="cuda", dtype=torch.float32)
    w0 = torch.randn(na, **st); m0 = torch.randn(na, **st) * 0.1
    v0 = torch.rand(na, **st) * 0.1; gg = torch.randn(na, **st) * 0.01
    W = {k: t.clone() for k, t in dict(w=w0, m=m0, v=v0).items()}

    gfn = G4.compile_src(G4.SRC, "v4asm")
    qfn = OS.compile_fn(OS.Q8_SRC, "q8asm", "gemm_i4_q8")
    cfn = OS.compile_fn(OS.Q8_SRC, "quant8asm", "quant8")
    tkn = OS.compile_fn(SRC, "trainmegak", "train_megak")
    akn = OS.compile_fn(SRC, "adamasm", "adamk")

    def baseline():
        OS.launch(gfn, (N // 64, M // 64, 1), (32, 4, 1), G4.SHARED,
                  [Ap, Bt, scale, Outf, M, N, Kw])
        OS.launch(cfn, (512, 1, 1), (256, 1, 1), 0, [O8a, Outf, 1.0, M * N])
        OS.launch(akn, (512, 1, 1), (256, 1, 1), 0,
                  [W['w'], W['m'], W['v'], gg, LR, B1, B2, EPS, na])

    def assembled(G, X):
        OS.launch(tkn, (G + X, 1, 1), (32, 4, 1), G4.SHARED,
                  [Ap, Bt, scale, O8b, M, N, Kw, N // 64, G,
                   W['w'], W['m'], W['v'], gg, LR, B1, B2, EPS, na])

    # ---------- correctness ----------
    OS.sync(); baseline(); OS.sync()
    for k, src in dict(w=w0, m=m0, v=v0).items():   # reset for the assembled run
        W[k].copy_(src)
    assembled(160, 56); OS.sync()
    same_q = bool(torch.equal(O8a, O8b))
    mi = B1 * m0 + (1 - B1) * gg
    vi = B2 * v0 + (1 - B2) * gg * gg
    wref = w0 - LR * mi / (vi.sqrt() + EPS)
    ok_w = (torch.allclose(W['m'], mi, atol=1e-5)
            and torch.allclose(W['v'], vi, atol=1e-5)
            and torch.allclose(W['w'], wref, atol=1e-5))
    print(f"int8 output bitwise identical to baseline pipeline : {same_q}", flush=True)
    print(f"Adam sweep matches torch reference (m, v, w)       : {ok_w}", flush=True)
    assert same_q and ok_w, "CORRECTNESS FAILED"

    # ---------- timing ----------
    n_it = 3

    def timed(f):
        OS.sync()
        t0 = time.perf_counter()
        for _ in range(n_it):
            f()
        OS.sync()
        return (time.perf_counter() - t0) / n_it

    T_base = timed(baseline)
    print(f"baseline (3 launches: gemm+quant+adam): {T_base*1e3:6.2f} ms", flush=True)
    for G, X in ((448, 112), (560, 56), (392, 168)):
        T_g = timed(lambda: OS.launch(
            tkn, (G, 1, 1), (32, 4, 1), G4.SHARED,
            [Ap, Bt, scale, O8b, M, N, Kw, N // 64, G,
             W['w'], W['m'], W['v'], gg, LR, B1, B2, EPS, 0]))
        T_m = timed(lambda: OS.launch(
            tkn, (X, 1, 1), (32, 4, 1), G4.SHARED,
            [Ap, Bt, scale, O8b, M, N, Kw, N // 64, 0,
             W['w'], W['m'], W['v'], gg, LR, B1, B2, EPS, na]))
        T_asm = timed(lambda: assembled(G, X))
        print(f"assembled {G}g+{X}m: gemm-alone {T_g*1e3:6.2f} ms  "
              f"adam-alone {T_m*1e3:6.2f} ms  BOTH {T_asm*1e3:6.2f} ms  "
              f"-> vs baseline {T_base/T_asm:.2f}x   "
              f"(adam {(1 - (T_asm - T_g) / max(T_m, 1e-9)) * 100:.0f}% hidden)",
              flush=True)
    tf = 2 * M * N * K / 1e12
    print(f"(GEMM work alone = {tf/1e-3:.1f} TFLOP per launch; "
          f"adam traffic = {6 * na * 4 / 1e9:.1f} GB per launch)", flush=True)

    # ---------- V2: raster + role hybrid — the GEMM stays in its fastest
    # form (hardware interleaves many short raster blocks; the persistent
    # form under-provisions latency hiding or over-runs LDS residency),
    # memory-role blocks appended to the SAME grid squat their slots while
    # GEMM blocks churn around them.
    SRC2 = SRC.replace("train_megak", "train_megak_raster").replace(
        "        int chunk = (total + G_CTAS - 1) / G_CTAS;\n"
        "        int t0 = blockIdx.x * chunk;\n"
        "        int t1 = t0 + chunk; if (t1 > total) t1 = total;\n",
        "        int t0 = blockIdx.x;\n"
        "        int t1 = (t0 + 1 < total) ? t0 + 1 : total;\n"
        "        if (t0 >= total) t1 = t0;\n").replace("G_CTAS", "G_BLOCKS")
    tk2 = OS.compile_fn(SRC2, "trainmegak2", "train_megak_raster")
    GB = (M // 64) * (N // 64)

    def assembled2(X):
        OS.launch(tk2, (GB + X, 1, 1), (32, 4, 1), G4.SHARED,
                  [Ap, Bt, scale, O8b, M, N, Kw, N // 64, GB,
                   W['w'], W['m'], W['v'], gg, LR, B1, B2, EPS, na])

    for k, src in dict(w=w0, m=m0, v=v0).items():
        W[k].copy_(src)
    assembled2(168); OS.sync()
    ok2 = (bool(torch.equal(O8a, O8b))
           and torch.allclose(W['m'], mi, atol=1e-5)
           and torch.allclose(W['v'], vi, atol=1e-5)
           and torch.allclose(W['w'], wref, atol=1e-5))
    print(f"V2 raster+role correctness (bitwise int8 + adam): {ok2}", flush=True)
    assert ok2, "V2 CORRECTNESS FAILED"

    T_g2 = timed(lambda: OS.launch(
        tk2, (GB, 1, 1), (32, 4, 1), G4.SHARED,
        [Ap, Bt, scale, O8b, M, N, Kw, N // 64, GB,
         W['w'], W['m'], W['v'], gg, LR, B1, B2, EPS, 0]))
    print(f"V2 raster GEMM alone (fused int8 epilogue): {T_g2*1e3:6.2f} ms "
          f"({tf / T_g2:6.1f} TFLOPS)", flush=True)
    # ---------- V3: memory blocks FIRST in the grid — they dispatch at
    # kernel start, squat their CTA/LDS slots for the whole launch, and the
    # GEMM raster churns through the remaining slots behind them.
    SRC3 = r"""
typedef unsigned int uint32_t;
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
#define NT 4
extern "C" __global__ void train_megak3(
        const uint32_t* __restrict__ Ap, const uint32_t* __restrict__ Bt,
        const float* __restrict__ scale, signed char* __restrict__ Out,
        int M, int N, int Kw, int TilesN, int X_MEM,
        float* w, float* m, float* v, const float* g,
        float lr, float b1, float b2, float eps, int n) {
    if (X_MEM > 0 && (int)blockIdx.x < X_MEM) {
        // ---- memory role (first X_MEM blocks; long-lived, grid-stride) ----
        int tid = threadIdx.y * 32 + threadIdx.x;
        for (int i = blockIdx.x * 128 + tid; i < n; i += X_MEM * 128) {
            float mi = b1 * m[i] + (1.0f - b1) * g[i];
            float vi = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
            m[i] = mi;
            v[i] = vi;
            w[i] -= lr * mi / (sqrtf(vi) + eps);
        }
    } else {
        // ---- GEMM role: one 64x64 tile per block, raster-churned ----
        extern __shared__ int lds[];
        int tile = blockIdx.x - X_MEM;
        int n0 = (tile % TilesN) * 64;
        int mb = (tile / TilesN) * 64;
        int lane = threadIdx.x & 31;
        int warp = threadIdx.y & 3;
        int col = lane & 15, kt = lane >> 4;
        v8i acc[NT];
        for (int i = 0; i < NT; ++i) acc[i] = {};
        auto load = [&](int kw, int buf) {
            int* LA = lds + buf * 576;
            int* LB = LA + 320;
            for (int w2 = threadIdx.y * 32 + lane; w2 < 64 * 5; w2 += 128) {
                int r = w2 / 5, q = w2 % 5;
                LA[r * 5 + q] = (q < 4) ? Ap[(mb + r) * Kw + kw + q] : 0;
            }
            for (int w2 = threadIdx.y * 32 + lane; w2 < 4 * 64; w2 += 128) {
                int q = w2 >> 6, nl = w2 & 63;
                LB[q * 64 + nl] = Bt[(kw + q) * N + n0 + nl];
            }
        };
        load(0, 0);
        __syncthreads();
        for (int kw = 0; kw < Kw; kw += 4) {
            int* LA = lds + (kw / 4 & 1) * 576;
            int* LB = LA + 320;
            int row_local = warp * 16 + col;
            v2i a; a.x = LA[row_local * 5 + kt * 2]; a.y = LA[row_local * 5 + kt * 2 + 1];
            for (int i = 0; i < NT; ++i) {
                v2i b; b.x = LB[(kt * 2) * 64 + i * 16 + col];
                b.y = LB[(kt * 2 + 1) * 64 + i * 16 + col];
                acc[i] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, a, 1, b, acc[i], 0);
            }
            __syncthreads();
            if (kw + 4 < Kw) load(kw + 4, ((kw / 4) + 1) & 1);
            __syncthreads();
        }
        int rbase = (lane >> 4) * 8;
        for (int i = 0; i < NT; ++i)
            for (int j = 0; j < 8; ++j) {
                float x = (float)acc[i][j] * scale[0];
                Out[(mb + warp * 16 + rbase + j) * N + n0 + i * 16 + col] =
                    (signed char)(int)(x > 127.f ? 127.f : (x < -127.f ? -127.f : x));
            }
    }
}
"""
    tk3 = OS.compile_fn(SRC3, "trainmegak3", "train_megak3")

    def assembled3(X):
        OS.launch(tk3, (X + GB, 1, 1), (32, 4, 1), G4.SHARED,
                  [Ap, Bt, scale, O8b, M, N, Kw, N // 64, X,
                   W['w'], W['m'], W['v'], gg, LR, B1, B2, EPS, na])

    for k, src in dict(w=w0, m=m0, v=v0).items():
        W[k].copy_(src)
    assembled3(168); OS.sync()
    ok3 = (bool(torch.equal(O8a, O8b))
           and torch.allclose(W['m'], mi, atol=1e-5)
           and torch.allclose(W['v'], vi, atol=1e-5)
           and torch.allclose(W['w'], wref, atol=1e-5))
    print(f"V3 mem-first correctness (bitwise int8 + adam): {ok3}", flush=True)
    assert ok3, "V3 CORRECTNESS FAILED"

    for X in (112, 168, 336):
        T_m3 = timed(lambda: OS.launch(
            tk3, (X, 1, 1), (32, 4, 1), G4.SHARED,
            [Ap, Bt, scale, O8b, M, N, Kw, N // 64, X,
             W['w'], W['m'], W['v'], gg, LR, B1, B2, EPS, na]))
        T_a3 = timed(lambda: assembled3(X))
        hid = (1 - (T_a3 - T_g2) / max(T_m3, 1e-9)) * 100
        print(f"V3 assembled {X:3d}m-first: BOTH {T_a3*1e3:6.2f} ms  "
              f"-> vs baseline {T_base/T_a3:.2f}x   (adam {hid:.0f}% hidden, "
              f"adam-alone {T_m3*1e3:.1f} ms, gemm-alone {T_g2*1e3:.1f} ms)",
              flush=True)
