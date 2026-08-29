"""Stage-isolated dump for the v5c (NT=8, KCW=8) kernel: LDS contents + fragments."""
import sys
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, ctypes
import numpy as np
import wmma_gemm_v2 as W
HIP = W.HIP
RTC = W.RTC
HIP.hipDeviceSynchronize.restype = ctypes.c_int
HIP.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

HEAD = "typedef unsigned int uint32_t;\n#define NT 8\n#define KCW 8\n"

STAGE1 = HEAD + r"""
extern "C" __global__ void s1(const uint32_t* __restrict__ Ap,
                              const uint32_t* __restrict__ Bt,
                              const int* __restrict__ be,
                              const int* __restrict__ bm,
                              const float* __restrict__ scale,
                              int* __restrict__ Out,
                              int N, int Kw) {
    extern __shared__ int lds[];
    int n0 = 0;
    int e = be[blockIdx.y];
    int mb = bm[blockIdx.y];
    auto load = [&](int kw, int buf) {
        int* LA = lds + buf * 1600;
        int* LB = LA + 576;
        for (int w = threadIdx.y * 32 + threadIdx.x; w < 64 * 9; w += 128) {
            int r = w / 9, q = w % 9;
            LA[r * 9 + q] = (q < 8) ? Ap[(mb + r) * Kw + kw + q] : 0;
        }
        for (int w = threadIdx.y * 32 + threadIdx.x; w < KCW * 128; w += 128) {
            int q = w >> 7, nl = w & 127;
            LB[q * 128 + nl] = Bt[(kw + q) * N + n0 + nl];
        }
    };
    load(0, 0);
    __syncthreads();
    for (int w = threadIdx.y * 32 + threadIdx.x; w < 1600; w += 128)
        Out[w] = lds[w];
}
"""

STAGE2 = HEAD + r"""
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
extern "C" __global__ void s2(const uint32_t* __restrict__ Ap,
                              const uint32_t* __restrict__ Bt,
                              const int* __restrict__ be,
                              const int* __restrict__ bm,
                              const float* __restrict__ scale,
                              int* __restrict__ Out,
                              int N, int Kw) {
    extern __shared__ int lds[];
    int n0 = 0;
    int e = be[blockIdx.y];
    int mb = bm[blockIdx.y];
    auto load = [&](int kw, int buf) {
        int* LA = lds + buf * 1600;
        int* LB = LA + 576;
        for (int w = threadIdx.y * 32 + threadIdx.x; w < 64 * 9; w += 128) {
            int r = w / 9, q = w % 9;
            LA[r * 9 + q] = (q < 8) ? Ap[(mb + r) * Kw + kw + q] : 0;
        }
        for (int w = threadIdx.y * 32 + threadIdx.x; w < KCW * 128; w += 128) {
            int q = w >> 7, nl = w & 127;
            LB[q * 128 + nl] = Bt[(kw + q) * N + n0 + nl];
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
    long long o = ((long long)blockIdx.y * 4 + warp) * 1024;
    int base = lane * 32;
    for (int i = 0; i < NT; ++i) {
        v2i a; a.x = LA[row_local * 9 + kt * 2]; a.y = LA[row_local * 9 + kt * 2 + 1];
        v2i b; b.x = LB[(kt * 2) * 128 + i * 16 + col];
        b.y = LB[(kt * 2 + 1) * 128 + i * 16 + col];
        Out[o + base + i * 4 + 0] = a.x;
        Out[o + base + i * 4 + 1] = a.y;
        Out[o + base + i * 4 + 2] = b.x;
        Out[o + base + i * 4 + 3] = b.y;
    }
}
"""

def run_stage(src, tag, out_words, Ap, Bt, be, bm, N, Kw):
    buf = ctypes.create_string_buffer(src.encode())
    prog = ctypes.c_void_p()
    assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p),
                                   tag.encode(), 0, None, None) == 0
    opts = (ctypes.c_char_p * 4)(b"--offload-arch=gfx1201", b"-O3", b"-DNEG_A=1", b"-DNEG_B=1")
    stc = RTC.hiprtcCompileProgram(prog, 4, opts)
    if stc != 0:
        szl = ctypes.c_size_t(); RTC.hiprtcGetProgramLogSize(prog, ctypes.byref(szl))
        lg = ctypes.create_string_buffer(szl.value + 1); RTC.hiprtcGetProgramLog(prog, lg)
        print(f"COMPILE FAIL {tag}:", lg.value.decode(errors='replace')[-800:], flush=True)
        raise SystemExit(1)
    csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
    code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
    m2 = ctypes.c_void_p(); HIP.hipModuleLoadData(ctypes.byref(m2), code)
    kern = ctypes.c_void_p()
    HIP.hipModuleGetFunction(ctypes.byref(kern), m2, (b"s1" if tag.startswith("s1") else b"s2"))
    Out = torch.zeros(out_words, device="cuda", dtype=torch.int32)
    args = [Ap, Bt, be, bm, torch.ones(1, device="cuda"), Out, N, Kw]
    storage = [ctypes.c_void_p(t.data_ptr()) if torch.is_tensor(t) else ctypes.c_int32(t)
               for t in args]
    ptrs = (ctypes.c_void_p * len(storage))(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in storage])
    st = HIP.hipModuleLaunchKernel(kern, 1, 1, 1, 32, 4, 1, 2 * 1600 * 4, None, ptrs, None)
    torch.cuda.synchronize()
    assert st == 0, f"launch {st}"
    host = np.empty(out_words, dtype=np.int32)
    HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out.data_ptr()),
                  out_words * 4, 2)
    return host

if __name__ == "__main__":
    torch.manual_seed(0)
    E, N, K = 2, 128, 64
    Kw = K // 8
    A = torch.randint(1, 8, (64, K), device="cuda", dtype=torch.int32)
    Ap = W.pack(A).contiguous()                                   # (64, 8)
    W0 = torch.randint(1, 8, (N, K), device="cuda", dtype=torch.int32)
    Bt = W.pack_transposed(W0).contiguous()                       # (8, 128)
    be = torch.tensor([0], device="cuda", dtype=torch.int32)
    bm = torch.tensor([0], device="cuda", dtype=torch.int32)

    # ---- stage 1: LDS contents ----
    lds = run_stage(STAGE1, "s1_v5c", 1600, Ap, Bt, be, bm, N, Kw)
    A_np = Ap.cpu().numpy().reshape(64, 8)
    Bt_np = Bt.cpu().numpy().reshape(8, 128)
    a_bad = b_bad = 0
    for w in range(576):
        r, q = w // 9, w % 9
        exp = int(A_np[r, q]) if q < 8 else 0
        if lds[w] != exp: a_bad += 1
    for w in range(1024):
        q, nl = w >> 7, w & 127
        if lds[576 + w] != int(Bt_np[q, nl]): b_bad += 1
    print(f"stage1 LDS: A words wrong {a_bad}/576, B words wrong {b_bad}/1024", flush=True)

    # ---- stage 2: fragments (all 8 tiles) ----
    fr = run_stage(STAGE2, "s2_v5c", 4096, Ap, Bt, be, bm, N, Kw).reshape(4, 32, 32)
    Bt_flat = Bt.cpu().numpy().reshape(-1)
    fr_bad = 0
    for w in range(4):
        for l in range(32):
            col, kt = l & 15, l >> 4
            row = w * 16 + col
            for i in range(8):
                ea = [int(A_np[row, kt * 2]), int(A_np[row, kt * 2 + 1])]
                eb = [int(Bt_flat[(kt * 2) * 128 + i * 16 + col]),
                      int(Bt_flat[(kt * 2 + 1) * 128 + i * 16 + col])]
                g = fr[w, l, i * 4: i * 4 + 4].tolist()
                if g != ea + eb:
                    fr_bad += 1
                    if fr_bad <= 4:
                        print(f"  tile{i} warp{w} lane{l}: a {g[:2]} vs {ea} | b {g[2:]} vs {eb}", flush=True)
    print(f"stage2 fragments: {fr_bad}/128 lanes wrong", flush=True)
