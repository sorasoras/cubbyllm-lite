"""Full INT4xINT4 GEMM on RDNA4 via native iu4 WMMA. Correctness + benchmark.

Discovered fragment mapping (verified empirically, 16x16x16, wave32):
  A (MxK): lane l, slot j -> A[l%16][(l/16)*8 + j]        (row-distributed)
  B (KxN): lane l, slot j -> B[(l/16)*8 + j][l%16]        (column-distributed)
  D (MxN): lane l, slot j -> D[(l/16)*8 + j][l%16]        (column-distributed)
Weights stored (N,K) packed int4: word w of row r covers k = 8w..8w+7,
nibble j = element 8w+j. A fragment for K-tile t = word (2t + l/16) of row m.
Signedness: compile-time NEG_A/NEG_B sweep; fallback = +8 offset trick.
"""
import ctypes
import numpy as np
import torch

HIP = ctypes.CDLL(r"B:\git\rocm-venv\Lib\site-packages\_rocm_sdk_core\bin\amdhip64_7.dll")
RTC = ctypes.CDLL(r"B:\git\rocm-venv\Lib\site-packages\_rocm_sdk_core\bin\hiprtc0714.dll")
RTC.hiprtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p,
                                    ctypes.c_char_p, ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_char_p)]
RTC.hiprtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
RTC.hiprtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
RTC.hiprtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
RTC.hiprtcGetCodeSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
RTC.hiprtcGetCode.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
HIP.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
HIP.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
HIP.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]

SRC = r"""
typedef unsigned int uint32_t;
typedef int v8i __attribute__((ext_vector_type(8)));
#define NT 4
extern "C" __global__ void gemm_i4(const uint32_t* __restrict__ Ap,
                                   const uint32_t* __restrict__ Bp,
                                   const float* __restrict__ scale,
                                   float* __restrict__ Out,
                                   int M, int N, int Kw) {
    int n_tile0 = blockIdx.x * NT, m_tile = blockIdx.y;
    int lane = threadIdx.x & 31;
    int m = m_tile * 16 + (lane & 15);
    int kt = lane >> 4;
    v8i acc[NT];
    for (int i = 0; i < NT; ++i) acc[i] = {};
    for (int kw = 0; kw < Kw; kw += 2) {
        int a = Ap[m * Kw + kw + kt];                 // A word reused across NT
        for (int i = 0; i < NT; ++i) {
            int b = Bp[(n_tile0 + i) * 16 + (lane & 15)] * 0 + // placeholder
                    Bp[((n_tile0 + i) * 16 + (lane & 15)) * Kw + kw + kt];
            acc[i] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(
                NEG_A, a, NEG_B, b, acc[i], 0);
        }
    }
    int col = lane & 15, rbase = (lane >> 4) * 8;
    for (int i = 0; i < NT; ++i)
        for (int j = 0; j < 8; ++j)
            Out[(m_tile * 16 + rbase + j) * N + (n_tile0 + i) * 16 + col] =
                (float)acc[i][j] * scale[0];
}
"""

def compile_variant(neg_a, neg_b):
    src = ctypes.create_string_buffer(SRC.encode())
    prog = ctypes.c_void_p()
    assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(src, ctypes.c_char_p),
                                   b"gemm.hip", 0, None, None) == 0
    opts = (ctypes.c_char_p * 4)(b"--offload-arch=gfx1201", b"-O3",
                                 f"-DNEG_A={neg_a}".encode(), f"-DNEG_B={neg_b}".encode())
    st = RTC.hiprtcCompileProgram(prog, 4, opts)
    if st != 0:
        sz = ctypes.c_size_t(); RTC.hiprtcGetProgramLogSize(prog, ctypes.byref(sz))
        log = ctypes.create_string_buffer(sz.value + 1); RTC.hiprtcGetProgramLog(prog, log)
        raise RuntimeError(log.value.decode(errors="replace")[-1500:])
    csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
    code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
    mod = ctypes.c_void_p()
    assert HIP.hipModuleLoadData(ctypes.byref(mod), code) == 0
    fn = ctypes.c_void_p()
    assert HIP.hipModuleGetFunction(ctypes.byref(fn), mod, b"gemm_i4") == 0
    return fn

def pack(t):
    """(R, K) int tensor with values in nibble range -> (R, K/8) int32 packed words."""
    R, K = t.shape
    out = torch.zeros((R, K // 8), device=t.device, dtype=torch.int64)
    for i in range(8):
        out |= (t[:, i::8].long() & 0xF) << (4 * i)
    return out.to(torch.int32).contiguous()

def gemm_i4(fn, A_packed, B_packed, scale):
    M, Kw = A_packed.shape
    N = B_packed.shape[0]
    Out = torch.empty((M, N), device="cuda", dtype=torch.float32)
    storage = [ctypes.c_void_p(A_packed.data_ptr()), ctypes.c_void_p(B_packed.data_ptr()),
               ctypes.c_void_p(scale.data_ptr()), ctypes.c_void_p(Out.data_ptr()),
               ctypes.c_int32(M), ctypes.c_int32(N), ctypes.c_int32(Kw)]
    ptrs = (ctypes.c_void_p * 7)(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in storage])
    st = HIP.hipModuleLaunchKernel(fn, (N // 16) // 4, (M // 16), 1, 32, 1, 1, 0, None, ptrs, None)
    torch.cuda.synchronize()
    assert st == 0, f"launch {st}"
    return Out

# ---------------- correctness: unsigned mapping sanity ----------------
torch.manual_seed(0)
M, N, K = 256, 2048, 768
A_u = torch.randint(0, 16, (M, K), device="cuda", dtype=torch.int32)
B_u = torch.randint(0, 16, (N, K), device="cuda", dtype=torch.int32)
fn00 = compile_variant(0, 0)
scale = torch.ones(1, device="cuda")
out_u = gemm_i4(fn00, pack(A_u), pack(B_u), scale)
ref_u = A_u.float() @ B_u.float().T
err = (out_u - ref_u).abs().max().item()
print(f"[unsigned 0..15, neg=0,0]  max|err| = {err:.1f}  {'PASS' if err == 0 else 'FAIL'}", flush=True)

# ---------------- signedness sweep ----------------
A_s = torch.randint(-8, 8, (M, K), device="cuda", dtype=torch.int32)
B_s = torch.randint(-8, 8, (N, K), device="cuda", dtype=torch.int32)
ref_s = A_s.float() @ B_s.float().T
signed_fn = None
for na, nb in [(0, 0), (1, 1), (1, 0), (0, 1)]:
    fn = compile_variant(na, nb) if (na, nb) != (0, 0) else fn00
    out_s = gemm_i4(fn, pack(A_s), pack(B_s), scale)
    e = (out_s - ref_s).abs().max().item()
    print(f"[signed nibbles, neg=({na},{nb})]  max|err| = {e:.1f}  {'PASS' if e == 0 else 'FAIL'}", flush=True)
    if e == 0:
        signed_fn = (fn, na, nb)
        break

# fallback: offset trick (values +8, two corrections)
if signed_fn is None:
    A_of, B_of = A_s + 8, B_s + 8
    out_of = gemm_i4(fn00, pack(A_of), pack(B_of), scale)
    corr = out_of - 8 * A_s.float().sum(1, keepdim=True) - 8 * B_s.float().sum(1)[None, :] - 64 * K
    e = (corr - ref_s).abs().max().item()
    print(f"[offset +8 + corrections]  max|err| = {e:.1f}  {'PASS' if e == 0 else 'FAIL'}", flush=True)
    signed_fn = (fn00, "offset")

# ---------------- benchmark ----------------
def bench(fn_call, n=50):
    fn_call(); torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(n): fn_call()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n

print("\n=== benchmark: M=4096, K=768, N=2048 (16.8 GFLOP) ===", flush=True)
Mb = 4096
Ab = torch.randint(-8, 8, (Mb, K), device="cuda", dtype=torch.int8)
Bb = torch.randint(-8, 8, (N, K), device="cuda", dtype=torch.int8)
Ab32 = Ab.to(torch.int32); Bb32 = Bb.to(torch.int32)
gflop = 2 * Mb * K * N / 1e9

t_fp32 = bench(lambda: Ab.float() @ Bb.float().T)
t_i8 = bench(lambda: torch._int_mm(Ab, Bb.t()))
fn_use = signed_fn[0]
Ap_b, Bp_b = pack(Ab32), pack(Bb32)
t_i4 = bench(lambda: gemm_i4(fn_use, Ap_b, Bp_b, scale))

print(f"fp32 eager : {t_fp32:8.3f} ms  {gflop/t_fp32:7.1f} TFLOPS")
print(f"int8 _int_mm: {t_i8:8.3f} ms  {gflop/t_i8:7.1f} TFLOPS")
print(f"int4 WMMA  : {t_i4:8.3f} ms  {gflop/t_i4:7.1f} TFLOPS")
print(f"speedup vs fp32: {t_fp32/t_i4:.2f}x | vs int8: {t_i8/t_i4:.2f}x")
