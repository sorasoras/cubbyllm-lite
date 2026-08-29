"""Stage-isolated v5b debug: dump LDS contents AND per-lane fragment reads.

Stage 1 kernel: v5b's exact load, then write ALL of LDS to Out.
Stage 2 kernel: v5b's exact load + fragment reads, then write each lane's
a.x, a.y, b.x, b.y (as the WMMA would consume them) to Out.
Both compared against host-computed expectations. Single block, tiny config.
"""
import sys
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, ctypes
import numpy as np
import wmma_gemm_v2 as W
HIP = W.HIP
RTC = W.RTC
HIP.hipDeviceSynchronize.restype = ctypes.c_int
HIP.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

HEAD = r"""
typedef unsigned int uint32_t;
#define NT 4
#define KCW 8
"""

STAGE1 = HEAD + r"""
extern "C" __global__ void s1(const uint32_t* __restrict__ Ap,
                              const uint32_t* __restrict__ Wp,
                              const int* __restrict__ blk_expert,
                              const int* __restrict__ blk_mbase,
                              const float* __restrict__ scale,
                              int* __restrict__ Out,
                              int N, int Kw) {
    extern __shared__ int lds[];
    int n0 = 0;
    int e = blk_expert[blockIdx.y];
    int mb = blk_mbase[blockIdx.y];
    auto load = [&](int kw, int buf) {
        int* LA = lds + buf * 1152;
        int* LB = LA + 576;
        for (int w = threadIdx.y * 32 + threadIdx.x; w < 64 * 9; w += 128) {
            int r = w / 9, q = w % 9;
            LA[r * 9 + q] = (q < 8) ? Ap[(mb + r) * Kw + kw + q] : 0;
        }
        for (int w = threadIdx.y * 32 + threadIdx.x; w < KCW * 64; w += 128) {
            int q = w >> 6, nl = w & 63;
            LB[q * 64 + nl] = Wp[((long)e * N + n0 + nl) * Kw + kw + q];
        }
    };
    load(0, 0);
    __syncthreads();
    int* L = lds;
    for (int w = threadIdx.y * 32 + threadIdx.x; w < 1088; w += 128)
        Out[w] = L[w];
}
"""

STAGE2 = HEAD + r"""
extern "C" __global__ void s2(const uint32_t* __restrict__ Ap,
                              const uint32_t* __restrict__ Wp,
                              const int* __restrict__ blk_expert,
                              const int* __restrict__ blk_mbase,
                              const float* __restrict__ scale,
                              int* __restrict__ Out,
                              int N, int Kw) {
    extern __shared__ int lds[];
    int n0 = 0;
    int e = blk_expert[blockIdx.y];
    int mb = blk_mbase[blockIdx.y];
    auto load = [&](int kw, int buf) {
        int* LA = lds + buf * 1152;
        int* LB = LA + 576;
        for (int w = threadIdx.y * 32 + threadIdx.x; w < 64 * 9; w += 128) {
            int r = w / 9, q = w % 9;
            LA[r * 9 + q] = (q < 8) ? Ap[(mb + r) * Kw + kw + q] : 0;
        }
        for (int w = threadIdx.y * 32 + threadIdx.x; w < KCW * 64; w += 128) {
            int q = w >> 6, nl = w & 63;
            LB[q * 64 + nl] = Wp[((long)e * N + n0 + nl) * Kw + kw + q];
        }
    };
    load(0, 0);
    __syncthreads();
    int* LA = lds;
    int* LB = LA + 576;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.y & 3;
    int col = lane & 15, kt = lane >> 4;
    int row_local = warp * 16 + col;
    long long o = ((long long)blockIdx.y * 4 + warp) * 128;
    // per-lane fragment words exactly as v5b's compute reads them
    int base = lane * 4;
    Out[o + base + 0] = LA[row_local * 9 + kt * 2];         // a.x
    Out[o + base + 1] = LA[row_local * 9 + kt * 2 + 1];     // a.y
    Out[o + base + 2] = LB[(kt * 2) * 64 + col];            // b.x (i=0)
    Out[o + base + 3] = LB[(kt * 2 + 1) * 64 + col];        // b.y (i=0)
}
"""

def run_stage(src, tag, out_words, Ap, Wp, be, bm, N, Kw):
    buf = ctypes.create_string_buffer(src.encode())
    prog = ctypes.c_void_p()
    assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p),
                                   tag.encode(), 0, None, None) == 0
    opts = (ctypes.c_char_p * 4)(b"--offload-arch=gfx1201", b"-O3", b"-DNEG_A=1", b"-DNEG_B=1")
    assert RTC.hiprtcCompileProgram(prog, 4, opts) == 0, f"compile failed {tag}"
    csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
    code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
    m2 = ctypes.c_void_p(); HIP.hipModuleLoadData(ctypes.byref(m2), code)
    kern = ctypes.c_void_p()
    name = b"s1" if tag.startswith("s1") else b"s2"
    HIP.hipModuleGetFunction(ctypes.byref(kern), m2, name)
    Out = torch.zeros(out_words, device="cuda", dtype=torch.int32)
    args = [Ap, Wp, be, bm, torch.ones(1, device="cuda"), Out, N, Kw]
    storage = [ctypes.c_void_p(t.data_ptr()) if torch.is_tensor(t) else ctypes.c_int32(t)
               for t in args]
    ptrs = (ctypes.c_void_p * len(storage))(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in storage])
    st = HIP.hipModuleLaunchKernel(kern, 1, 1, 1, 32, 4, 1, 2 * 1152 * 4, None, ptrs, None)
    torch.cuda.synchronize()
    assert st == 0, f"launch {st}"
    host = np.empty(out_words, dtype=np.int32)
    HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out.data_ptr()),
                  out_words * 4, 2)
    return host

if __name__ == "__main__":
    torch.manual_seed(0)
    E, N, K = 2, 64, 64
    Kw = K // 8
    A = torch.randint(1, 8, (64, K), device="cuda", dtype=torch.int32)   # 1 block of 64 rows, expert 0
    Ap = W.pack(A).contiguous()                                          # (64, 8)
    Wp = W.pack(torch.randint(1, 8, (E * N, K), device="cuda", dtype=torch.int32)).contiguous()
    be = torch.tensor([0], device="cuda", dtype=torch.int32)
    bm = torch.tensor([0], device="cuda", dtype=torch.int32)

    # ---- stage 1: LDS contents (strided A layout) ----
    lds = run_stage(STAGE1, "s1_dump", 1088, Ap, Wp, be, bm, N, Kw)
    A_np = Ap.cpu().numpy().reshape(64, Kw)          # (64 rows, 8 words)
    Wp_np = Wp.cpu().numpy().reshape(E * N, Kw)      # (128 weight-rows, 8 words)
    a_bad = b_bad = 0
    for w in range(576):
        r, q = w // 9, w % 9
        exp = int(A_np[r, q]) if q < 8 else 0
        if lds[w] != exp: a_bad += 1
    for w in range(512):
        q, nl = w >> 6, w & 63
        exp = int(Wp_np[nl, q])
        if lds[576 + w] != exp: b_bad += 1
    print(f"stage1 LDS: A words wrong {a_bad}/576, B words wrong {b_bad}/512", flush=True)

    # ---- stage 2: fragment reads (kernel indices vs host formulas) ----
    fr = run_stage(STAGE2, "s2_dump", 512, Ap, Wp, be, bm, N, Kw).reshape(4, 32, 4)
    fr_bad = 0
    for w in range(4):
        for l in range(32):
            col, kt = l & 15, l >> 4
            row = w * 16 + col
            ea = [int(A_np[row, kt * 2]), int(A_np[row, kt * 2 + 1])]
            eb = [int(Wp_np[col, kt * 2]), int(Wp_np[col, kt * 2 + 1])]
            ga = [fr[w, l, 0].item(), fr[w, l, 1].item()]
            gb = [fr[w, l, 2].item(), fr[w, l, 3].item()]
            if ga != ea or gb != eb:
                fr_bad += 1
                if fr_bad <= 4:
                    # locate the mis-read words in the A LDS region
                    locs = []
                    flat = A_np.reshape(-1)
                    for gi, gv in enumerate(flat):
                        if gv == ga[0]: locs.append(("A_flat", gi, gi // 9, gi % 9))
                    print(f"  frag mismatch w{w} l{l}: a {ga} vs {ea} | b ok={gb == eb} | "
                          f"ga[0] found at {locs[:3]}", flush=True)
    print(f"stage2 fragments: {fr_bad}/128 lanes wrong", flush=True)
