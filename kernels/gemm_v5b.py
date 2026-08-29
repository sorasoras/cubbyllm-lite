"""v5b: v4 + KCW=8 (64-k chunks). Minimal delta from the proven v4.

Block: 128 threads (4 warps, wave32). Warp w computes the 16x64 output
sub-tile at M-row-block w; all warps share one LDS load of the 64x64 A
tile and the 4x64 B k-chunk (padded A stride 5 to kill bank conflicts).
K=32 iu4 WMMA (663.5 TOPS instruction peak per mmapeak/BENCHMARK.md).
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

SRC = r"""
typedef unsigned int uint32_t;
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
#define NT 4
#define KCW 8
#define WARPS 4
// A: (M, Kw) packed; Bt: (Kw, N) packed-transposed
// grid: (N/64, M/64), block: 32 x 4 (lane, warp)
extern "C" __global__ void gemm_i4_v4(const uint32_t* __restrict__ Ap,
                                      const uint32_t* __restrict__ Bt,
                                      const float* __restrict__ scale,
                                      float* __restrict__ Out,
                                      int M, int N, int Kw) {
    extern __shared__ int lds[];   // 2 bufs x (64*9 A-pad + 8*64 B) ints
    int n0 = blockIdx.x * 64;
    int mb = blockIdx.y * 64;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.y & 3;
    int col = lane & 15, kt = lane >> 4;
    v8i acc[NT];
    for (int i = 0; i < NT; ++i) acc[i] = {};

    auto load = [&](int kw, int buf) {         // kw in words; chunk = 8 words (64 k)
        int* LA = lds + buf * 1600;
        int* LB = LA + 576;
        for (int w = threadIdx.y * 32 + lane; w < 64 * 9; w += 128) {
            int r = w / 9, q = w % 9;
            LA[r * 9 + q] = (q < 8) ? Ap[(mb + r) * Kw + kw + q] : 0;
        }
        for (int w = threadIdx.y * 32 + lane; w < KCW * 128; w += 128) {
            int q = w >> 7, nl = w & 127;
            LB[q * 128 + nl] = Bt[(kw + q) * N + n0 + nl];
        }
    };
    float oacc[NT][8];
    for (int i = 0; i < NT; ++i)
        for (int j = 0; j < 8; ++j) oacc[i][j] = 0.0f;
    for (int kw = 0; kw < Kw; kw += KCW) {
        load(kw, 0);
        __syncthreads();
        int* LA = lds;
        int* LB = LA + 576;
        int row_local = warp * 16 + col;
        // KCW=8 words = 64 k = 4 K=16 sub-tiles; the builtin consumes a.x
        // (16 k per wave via the lane-group split) -> 4 calls per chunk
        // two K=32 calls per 64-k chunk; interleaved fragments (confirmed layout)
        for (int t = 0; t < 2; ++t) {
            v2i a; a.x = LA[row_local * 9 + t * 4 + kt * 2];
            a.y = LA[row_local * 9 + t * 4 + kt * 2 + 1];
            for (int i = 0; i < NT; ++i) {
                v2i b; b.x = LB[(t * 4 + kt * 2) * 128 + i * 16 + col];
                b.y = LB[(t * 4 + kt * 2 + 1) * 128 + i * 16 + col];
                v8i r = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, a, 1, b, v8i{}, 0);
                for (int j = 0; j < 8; ++j) {
                    int lo = (r[j] << 16) >> 16;   // sign-extend lo16
                    int hi = r[j] >> 16;           // sign-extend hi16
                    oacc[i][j] += (float)lo + (float)hi;
                }
            }
        }
        __syncthreads();
    }
    int rbase = (lane >> 4) * 8;
    for (int i = 0; i < NT; ++i)
        for (int j = 0; j < 8; ++j)
            Out[(mb + warp * 16 + rbase + j) * N + n0 + i * 16 + col] =
                oacc[i][j] * scale[0];
}
"""

def compile_src(src, tag):
    buf = ctypes.create_string_buffer(src.encode())
    prog = ctypes.c_void_p()
    assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p),
                                   tag.encode(), 0, None, None) == 0
    opts = (ctypes.c_char_p * 4)(b"--offload-arch=gfx1201", b"-O3", b"-DNEG_A=1", b"-DNEG_B=1")
    assert RTC.hiprtcCompileProgram(prog, 4, opts) == 0, f"compile failed {tag}"
    csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
    code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
    m2 = ctypes.c_void_p()
    assert HIP.hipModuleLoadData(ctypes.byref(m2), code) == 0
    fn = ctypes.c_void_p()
    assert HIP.hipModuleGetFunction(ctypes.byref(fn), m2, b"gemm_i4_v4") == 0
    return fn

def pack(t):
    """VOP3P k-interleaved pack (confirmed layout):
    k = c*64 + tt*32 + kt*16 + 2j + w  <->  packed word c*8 + tt*4 + kt*2 + w, nibble j."""
    R, K = t.shape
    out = torch.zeros((R, K // 8), device=t.device, dtype=torch.int64)
    for c in range(K // 64):
        for tt in range(2):
            for kt in range(2):
                for w in range(2):
                    word = c * 8 + tt * 4 + kt * 2 + w
                    for j in range(8):
                        k = c * 64 + tt * 32 + kt * 16 + 2 * j + w
                        out[:, word] |= (t[:, k].long() & 0xF) << (4 * j)
    return out.to(torch.int32).contiguous()

def pack_transposed(t):
    """Same interleaved order, transposed: out[word, n] = t[n, k(word, nib)]."""
    N, K = t.shape
    out = torch.zeros((K // 8, N), device=t.device, dtype=torch.int64)
    for c in range(K // 64):
        for tt in range(2):
            for kt in range(2):
                for w in range(2):
                    word = c * 8 + tt * 4 + kt * 2 + w
                    for j in range(8):
                        k = c * 64 + tt * 32 + kt * 16 + 2 * j + w
                        out[word, :] |= ((t[:, k].t().long() & 0xF) << (4 * j))
    return out.to(torch.int32).contiguous()

SHARED = 2 * 1600 * 4
def launch(fn, grid, args_list, shared=SHARED):
    torch.cuda.synchronize()
    storage = [ctypes.c_void_p(t.data_ptr()) if torch.is_tensor(t) else ctypes.c_int32(t)
               for t in args_list]
    ptrs = (ctypes.c_void_p * len(storage))(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in storage])
    st = HIP.hipModuleLaunchKernel(fn, grid[0], grid[1], grid[2], 32, 4, 1, shared, None, ptrs, None)
    torch.cuda.synchronize()
    assert st == 0, f"launch {st}"

def bench(fnc, n=50):
    fnc(); torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(n): fnc()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n

if __name__ == "__main__":
    torch.manual_seed(0)
    fn = compile_src(SRC, "v4")
    scale = torch.ones(1, device="cuda")
    E, N, K = 8, 2048, 768
    Kw = K // 8
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 4096

    # pad tokens to 64-row blocks per expert for the v4 tile
    assign = torch.randint(0, E, (T,))
    order = torch.argsort(assign, stable=True)
    counts = torch.bincount(assign, minlength=E)
    segs64 = ((counts + 63) // 64) * 64           # 64-padded segments
    M_pad = int(segs64.sum())
    A_tok = torch.zeros(M_pad, K, device="cuda", dtype=torch.int32)
    seg_base = []
    pos = 0
    for eid in range(E):
        idx = order[assign == eid]; n_e = idx.numel()
        A_tok[pos:pos+n_e] = torch.randint(-8, 8, (n_e, K), device="cuda", dtype=torch.int32)
        seg_base.append((pos, int(segs64[eid]), n_e))
        pos += int(segs64[eid])
    W_all = torch.randint(-8, 8, (E * N, K), device="cuda", dtype=torch.int32)
    Ap = pack(A_tok).contiguous()
    Bt_e = [pack_transposed(W_all[eid*N:(eid+1)*N]).contiguous() for eid in range(E)]
    Out = torch.empty((M_pad, N), device="cuda", dtype=torch.float32)
    host = np.empty(M_pad * N, dtype=np.float32)

    def moe_launch():
        for eid in range(E):
            base, rows64, _ = seg_base[eid]
            args = [Ap[base:base+rows64], Bt_e[eid], scale,
                    Out[base:base+rows64], rows64, N, Kw]
            storage = [ctypes.c_void_p(t.data_ptr()) if torch.is_tensor(t) else ctypes.c_int32(t)
                       for t in args]
            ptrs = (ctypes.c_void_p * len(storage))(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in storage])
            st = HIP.hipModuleLaunchKernel(fn, N // 64, rows64 // 64, 1, 32, 4, 1, SHARED, None, ptrs, None)
            assert st == 0

    moe_launch()
    s1 = HIP.hipDeviceSynchronize()
    s2 = HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out.data_ptr()),
                       M_pad * N * 4, 2)
    Out_np = host.reshape(M_pad, N)
    A_np = A_tok.cpu().numpy().astype(np.float32)
    W_np = W_all.cpu().numpy().astype(np.float32)
    err = 0.0
    for eid in range(E):
        base, rows64, n_e = seg_base[eid]
        ref = A_np[base:base+n_e] @ W_np[eid*N:(eid+1)*N].T
        err = max(err, np.abs(Out_np[base:base+n_e] - ref).max())
    print(f"v5b-sync grouped MoE (T={T}): sync={s1} memcpy={s2}  max|err| = {err:.1f}  {'PASS' if err == 0 else 'FAIL'}", flush=True)

    gflop = 2 * T * K * N / 1e9
    t_moe = bench(moe_launch)
    A_q = A_tok[:T].to(torch.int8).contiguous()
    W_q = W_all[:N].to(torch.int8).contiguous()
    t_i8 = bench(lambda: torch._int_mm(A_q, W_q.t()))
    t_f32 = bench(lambda: A_tok[:T].float() @ W_all[:N].float().T)
    print(f"v5b int4 grouped-MoE : {t_moe:6.3f} ms  {gflop/t_moe:6.1f} TFLOPS")
    print(f"int8 _int_mm        : {t_i8:6.3f} ms  {gflop/t_i8:6.1f} TFLOPS")
    print(f"fp32 eager          : {t_f32:6.3f} ms  {gflop/t_f32:6.1f} TFLOPS")
    print(f"v5b vs fp32: {t_f32/t_moe:.2f}x | vs int8: {t_i8/t_moe:.2f}x | % of 663.5 TOPS peak: {gflop/t_moe/663.5*100:.1f}%")
