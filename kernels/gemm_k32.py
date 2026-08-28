"""v3: INT4xINT4 GEMM on the K=32 iu4 WMMA variant (2.29x K=16 issue rate).

Fragments: A row-distributed v2i (16 int4 = 2 words per lane), B column-
distributed v2i; acc v8i. Same verified mapping as v2, chunk = 4 packed
words (32 k-elements) per WMMA instruction.
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
extern "C" __global__ void gemm_i4_k32(const uint32_t* __restrict__ Ap,
                                       const uint32_t* __restrict__ Bt,
                                       const float* __restrict__ scale,
                                       float* __restrict__ Out,
                                       int M, int N, int Kw) {
    extern __shared__ int lds[];              // 2 bufs x (64 A + 256 B) ints
    int n0 = blockIdx.x * (16 * NT);
    int mb = blockIdx.y * 16;
    int lane = threadIdx.x & 31;
    int col = lane & 15, kt = lane >> 4;
    v8i acc[NT];
    for (int i = 0; i < NT; ++i) acc[i] = {};
    auto load = [&](int kw, int buf) {
        int* L = lds + buf * 320;
        for (int w = lane; w < 64; w += 32)               // A: 16 rows x 4 words
            L[w] = Ap[(mb + w / 4) * Kw + kw + w % 4];
        for (int w = lane; w < 16 * NT * 4; w += 32) {    // B: NT x 16 rows x 4 words
            int q = w >> 6, rem = w & 63, i = rem >> 4, r = rem & 15;
            L[64 + w] = Bt[(kw + q) * N + n0 + i * 16 + r];
        }
    };
    load(0, 0);
    __syncthreads();
    for (int kw = 0; kw < Kw; kw += 4) {
        int* L = lds + (kw / 4 & 1) * 320;
        v2i a; a.x = L[col * 4 + kt * 2]; a.y = L[col * 4 + kt * 2 + 1];
        for (int i = 0; i < NT; ++i) {
            v2i b; b.x = L[64 + (kt * 2) * 64 + i * 16 + col];
            b.y = L[64 + (kt * 2 + 1) * 64 + i * 16 + col];
            acc[i] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, a, 1, b, acc[i], 0);
        }
        __syncthreads();
        if (kw + 4 < Kw) load(kw + 4, ((kw / 4) + 1) & 1);
        __syncthreads();
    }
    int rbase = (lane >> 4) * 8;
    for (int i = 0; i < NT; ++i)
        for (int j = 0; j < 8; ++j)
            Out[(mb + rbase + j) * N + n0 + i * 16 + col] = (float)acc[i][j] * scale[0];
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
    assert HIP.hipModuleGetFunction(ctypes.byref(fn), m2, b"gemm_i4_k32") == 0
    return fn

def pack(t):
    R, K = t.shape
    out = torch.zeros((R, K // 8), device=t.device, dtype=torch.int64)
    for i in range(8):
        out |= (t[:, i::8].long() & 0xF) << (4 * i)
    return out.to(torch.int32).contiguous()

def pack_transposed(t):
    N, K = t.shape
    out = torch.zeros((K // 8, N), device=t.device, dtype=torch.int64)
    for i in range(8):
        out |= (t[:, i::8].t().long() & 0xF) << (4 * i)
    return out.to(torch.int32).contiguous()

def launch(fn, grid, args_list, shared=2 * 320 * 4):
    torch.cuda.synchronize()
    storage = [ctypes.c_void_p(t.data_ptr()) if torch.is_tensor(t) else ctypes.c_int32(t)
               for t in args_list]
    ptrs = (ctypes.c_void_p * len(storage))(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in storage])
    st = HIP.hipModuleLaunchKernel(fn, grid[0], grid[1], grid[2], 32, 1, 1, shared, None, ptrs, None)
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
    fn = compile_src(SRC, "k32")
    scale = torch.ones(1, device="cuda")

    # correctness
    M, N, K = 256, 2048, 768
    Kw = K // 8
    A = torch.randint(-8, 8, (M, K), device="cuda", dtype=torch.int32)
    B = torch.randint(-8, 8, (N, K), device="cuda", dtype=torch.int32)
    Ap, Bt = pack(A), pack_transposed(B)
    Out = torch.empty((M, N), device="cuda", dtype=torch.float32)
    launch(fn, (N // 64, M // 16, 1), [Ap, Bt, scale, Out, M, N, Kw])
    ref = A.float() @ B.float().T
    e = (Out - ref).abs().max().item()
    print(f"[K=32 dense signed] max|err| = {e:.1f}  {'PASS' if e == 0 else 'FAIL'}", flush=True)

    # benchmark + grouped MoE at T=4096
    E, T = 8, 4096
    Mb = T
    Ab = torch.randint(-8, 8, (Mb, K), device="cuda", dtype=torch.int32)
    Bb = torch.randint(-8, 8, (N, K), device="cuda", dtype=torch.int32)
    Apb, Btb = pack(Ab), pack_transposed(Bb)
    Outb = torch.empty((Mb, N), device="cuda", dtype=torch.float32)
    gflop = 2 * Mb * K * N / 1e9
    t_k32 = bench(lambda: launch(fn, (N // 64, Mb // 16, 1), [Apb, Btb, scale, Outb, Mb, N, Kw]))
    A_q = Ab.to(torch.int8).contiguous()
    B_q = Bb.to(torch.int8).contiguous()
    t_i8 = bench(lambda: torch._int_mm(A_q, B_q.t()))
    t_f32 = bench(lambda: Ab.float() @ Bb.float().T)
    print(f"=== dense M={Mb}, K={K}, N={N} ({gflop:.1f} GFLOP) ===")
    print(f"int4 K=32 WMMA : {t_k32:6.3f} ms  {gflop/t_k32:6.1f} TFLOPS")
    print(f"int8 _int_mm   : {t_i8:6.3f} ms  {gflop/t_i8:6.1f} TFLOPS")
    print(f"fp32 eager     : {t_f32:6.3f} ms  {gflop/t_f32:6.1f} TFLOPS")
    print(f"int4 K=32 vs fp32: {t_f32/t_k32:.2f}x | vs int8: {t_i8/t_k32:.2f}x", flush=True)
