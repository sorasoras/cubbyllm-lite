"""Overlap-survivors probe: test the four design answers that survived the
stream-serialization verdict (kernels/stream_overlap.py).

TEST 1 (lane stacking, the (1+P)-lane mechanism): v4 GEMM TFLOPS vs M —
  does stacking more rows against the same weights raise efficiency?
TEST 2 (batched passes): one 17-rep memory sweep vs 17 separate launches.
TEST 3 (epilogue fusion): int8-quantize fused into the GEMM epilogue vs
  GEMM(fp32 out) + separate quant kernel — outputs compared bitwise.
TEST 4 (megakernel CTA-roles): ONE launch, first G_CTAS CTAs run persistent
  GEMM tiles, the rest run the memory sweep — vs each alone. Decides
  in-kernel co-residency (the only overlap path left on this stack).
"""
import sys
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, ctypes, time
import numpy as np
import gemm_v4 as G4
HIP = G4.HIP
RTC = G4.RTC

Q8_SRC = r"""
typedef unsigned int uint32_t;
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
#define NT 4
extern "C" __global__ void gemm_i4_q8(const uint32_t* __restrict__ Ap,
                                      const uint32_t* __restrict__ Bt,
                                      const float* __restrict__ scale,
                                      signed char* __restrict__ Out,
                                      int M, int N, int Kw) {
    extern __shared__ int lds[];
    int n0 = blockIdx.x * 64;
    int mb = blockIdx.y * 64;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.y & 3;
    int col = lane & 15, kt = lane >> 4;
    v8i acc[NT];
    for (int i = 0; i < NT; ++i) acc[i] = {};
    auto load = [&](int kw, int buf) {
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
            float v = (float)acc[i][j] * scale[0];
            Out[(mb + warp * 16 + rbase + j) * N + n0 + i * 16 + col] =
                (signed char)(int)(v > 127.f ? 127.f : (v < -127.f ? -127.f : v));
        }
}
extern "C" __global__ void quant8(signed char* out, const float* x, float s, int n) {
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = i0; i < n; i += stride) {
        float v = x[i] * s;
        out[i] = (signed char)(int)(v > 127.f ? 127.f : (v < -127.f ? -127.f : v));
    }
}
"""

MEGA_SRC = r"""
typedef unsigned int uint32_t;
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
#define NT 4
extern "C" __global__ void megak(const uint32_t* __restrict__ Ap,
                                  const uint32_t* __restrict__ Bt,
                                  const float* __restrict__ scale,
                                  float* __restrict__ Out,
                                  int M, int N, int Kw, int TilesN, int G_CTAS,
                                  float* mo, const float* ma, const float* mb,
                                  int n, int reps) {
    if (G_CTAS > 0 && (int)blockIdx.x < G_CTAS) {
        // GEMM role: persistent loop over 64x64 output tiles
        extern __shared__ int lds[];
        int lane = threadIdx.x & 31;
        int warp = threadIdx.y & 3;
        int col = lane & 15, kt = lane >> 4;
        int total = (M / 64) * TilesN;
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
        for (int tile = blockIdx.x; tile < total; tile += G_CTAS) {
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
                for (int j = 0; j < 8; ++j)
                    Out[(mb + warp * 16 + rbase + j) * N + n0 + i * 16 + col] =
                        (float)acc[i][j] * scale[0];
        }
    } else {
        // memory role: Adam-like sweep, grid-stride over the memory CTAs
        int tid = threadIdx.y * 32 + threadIdx.x;
        int g0 = (G_CTAS > 0) ? G_CTAS : 0;
        int mc = blockIdx.x - g0;
        int MC = gridDim.x - g0;
        int i0 = mc * 128 + tid;
        int stride = MC * 128;
        for (int r = 0; r < reps; ++r)
            for (int i = i0; i < n; i += stride)
                mo[i] = ma[i] + mb[i];
    }
}
"""


def compile_fn(src, tag, name):
    buf = ctypes.create_string_buffer(src.encode())
    prog = ctypes.c_void_p()
    assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p),
                                   tag.encode(), 0, None, None) == 0
    opts = (ctypes.c_char_p * 2)(b"--offload-arch=gfx1201", b"-O3")
    assert RTC.hiprtcCompileProgram(prog, 2, opts) == 0, f"compile failed {tag}"
    csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
    code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
    mod = ctypes.c_void_p()
    assert HIP.hipModuleLoadData(ctypes.byref(mod), code) == 0
    fn = ctypes.c_void_p()
    assert HIP.hipModuleGetFunction(ctypes.byref(fn), mod, name.encode()) == 0
    return fn


def launch(fn, grid, block, shared, args, stream=None):
    storage = []
    for t in args:
        if torch.is_tensor(t):
            storage.append(ctypes.c_void_p(t.data_ptr()))
        elif isinstance(t, float):
            storage.append(ctypes.c_float(t))
        else:
            storage.append(ctypes.c_int32(t))
    ptrs = (ctypes.c_void_p * len(storage))(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p)
                                              for b in storage])
    st = HIP.hipModuleLaunchKernel(fn, grid[0], grid[1], grid[2], block[0], block[1],
                                    block[2], shared, stream, ptrs, None)
    assert st == 0, f"launch {st}"


def sync():
    assert HIP.hipDeviceSynchronize() == 0


def timed(f):
    sync()
    t0 = time.perf_counter()
    f()
    sync()
    return time.perf_counter() - t0


if __name__ == "__main__":
    torch.manual_seed(0)
    N, K = 2048, 4096
    Kw = K // 8
    scale = torch.full((1,), 1.0 / 128.0, device="cuda")
    Bm = torch.randint(-8, 8, (N, K), device="cuda", dtype=torch.int32)
    Bt = G4.pack_transposed(Bm).contiguous()
    gfn = G4.compile_src(G4.SRC, "v4s")
    qfn = compile_fn(Q8_SRC, "q8", "gemm_i4_q8")
    cfn = compile_fn(Q8_SRC, "quant8", "quant8")
    mfn = compile_fn(MEGA_SRC, "megak", "megak")

    print("=== TEST 1: lane stacking — v4 TFLOPS vs M (N=2048, K=4096) ===", flush=True)
    for M in (2048, 4096, 8192, 16384, 32768, 65536):
        A = torch.randint(-8, 8, (M, K), device="cuda", dtype=torch.int32)
        Ap = G4.pack(A).contiguous()
        Out = torch.empty((M, N), device="cuda", dtype=torch.float32)
        n_it = max(3, 16384 * 10 // M)

        def run():
            launch(gfn, (N // 64, M // 64, 1), (32, 4, 1), G4.SHARED,
                   [Ap, Bt, scale, Out, M, N, Kw])
        t = timed(lambda: [run() for _ in range(n_it)]) / n_it
        print(f"  M={M:6d}: {t*1e3:6.2f} ms  {2*M*N*K/1e12/t:6.1f} TFLOPS", flush=True)

    print("=== TEST 2: batched passes — 1x17 reps vs 17 launches ===", flush=True)
    Nf = 64 * 1024 * 1024
    a = torch.randn(Nf, device="cuda")
    b = torch.randn(Nf, device="cuda")
    o = torch.empty(Nf, device="cuda")
    sw = compile_fn(r"""
extern "C" __global__ void memsweep(float* out, const float* a, const float* b,
                                     float s, int n, int reps) {
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int r = 0; r < reps; ++r)
        for (int i = i0; i < n; i += stride)
            out[i] = a[i] * s + b[i];
}
""", "memsweep2", "memsweep")
    t_one = timed(lambda: launch(sw, (512, 1, 1), (256, 1, 1), 0, [o, a, b, 1.0, Nf, 17]))
    t_many = timed(lambda: [launch(sw, (512, 1, 1), (256, 1, 1), 0, [o, a, b, 1.0, Nf, 1])
                            for _ in range(17)])
    print(f"  one 17-rep launch : {t_one*1e3:6.1f} ms", flush=True)
    print(f"  17 launches       : {t_many*1e3:6.1f} ms   (overhead {1e3*(t_many-t_one):.2f} ms)",
          flush=True)

    print("=== TEST 3: epilogue fusion — int8 quant fused vs separate kernel ===", flush=True)
    M = 16384
    A = torch.randint(-8, 8, (M, K), device="cuda", dtype=torch.int32)
    Ap = G4.pack(A).contiguous()
    Out = torch.empty((M, N), device="cuda", dtype=torch.float32)
    O8a = torch.empty((M, N), device="cuda", dtype=torch.int8)
    O8b = torch.empty((M, N), device="cuda", dtype=torch.int8)
    ne = M * N

    def sep():
        launch(gfn, (N // 64, M // 64, 1), (32, 4, 1), G4.SHARED,
               [Ap, Bt, scale, Out, M, N, Kw])
        launch(cfn, (512, 1, 1), (256, 1, 1), 0, [O8a, Out, 1.0, ne])

    def fus():
        launch(qfn, (N // 64, M // 64, 1), (32, 4, 1), G4.SHARED,
               [Ap, Bt, scale, O8b, M, N, Kw])

    sep(); fus(); sync()
    same = bool(torch.equal(O8a, O8b))
    print(f"  fused vs separate outputs identical: {same}", flush=True)
    n_it = 20
    t_sep = timed(lambda: [sep() for _ in range(n_it)]) / n_it
    t_fus = timed(lambda: [fus() for _ in range(n_it)]) / n_it
    print(f"  separate (gemm+quant): {t_sep*1e3:6.2f} ms", flush=True)
    print(f"  fused (int8 epilogue) : {t_fus*1e3:6.2f} ms   "
          f"gain {t_sep/t_fus:.2f}x", flush=True)

    print("=== TEST 4: megakernel CTA-roles — gemm CTAs + memory CTAs, one launch ===",
          flush=True)
    Out2 = torch.empty((M, N), device="cuda", dtype=torch.float32)
    reps = 17

    def mega(g_ctas, x_ctas):
        launch(mfn, (g_ctas + x_ctas, 1, 1), (32, 4, 1), G4.SHARED,
               [Ap, Bt, scale, Out2, M, N, Kw, N // 64, g_ctas,
                o, a, b, Nf, reps])

    # correctness spot-check of the persistent GEMM role (tile 0,0)
    mega(4, 0); sync()
    ref = (A[:64].cpu().numpy().astype(np.float32)
           @ Bm[:64].cpu().numpy().astype(np.float32).T) / 128.0
    err = float(np.abs(Out2[:64, :64].cpu().numpy() - ref).max())
    print(f"  megakernel GEMM-role spot-check max|err| = {err}  "
          f"{'PASS' if err < 1e-3 else 'FAIL'}", flush=True)

    for G, X in ((160, 56), (112, 112)):
        T_g = timed(lambda: mega(G, 0))
        T_m = timed(lambda: mega(0, X))
        T_b = timed(lambda: mega(G, X))
        print(f"  split {G}g+{X}m: gemm-alone {T_g*1e3:6.1f} ms  mem-alone "
              f"{T_m*1e3:6.1f} ms  BOTH {T_b*1e3:6.1f} ms  -> "
              f"overlap factor {(T_g + T_m) / T_b:.2f}x "
              f"(serial {T_g + T_m - T_b:.2f} ms hidden)", flush=True)
