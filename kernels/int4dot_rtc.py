"""Native INT4xINT4 GEMM on RX 9070 via V_DOT8_I32_IU4, compiled with hiprtc.

Device kernel compiled at runtime by hiprtc (RTC mode sidesteps the
VS18/clang-23 header conflict entirely); code object loaded through
amdhip64_7.dll via ctypes; buffers are torch GPU tensors.

Kernel: GEMM with A = ternary spike activations packed SIGNED int4,
B = weights packed UNSIGNED int4 (s_w + 8). V_DOT8_I32_IU4 = 8 native
int4 MACs per instruction. Correction: true = raw - 8*rowsum(A).
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

extern "C" __global__ void gemm_i4_iu(const uint32_t* __restrict__ A,
                                      const uint32_t* __restrict__ Bp,
                                      const float* __restrict__ rowcorr,
                                      float* __restrict__ Out,
                                      int M, int N, int Kw) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    int acc = 0;
    const uint32_t* arow = A + (long)m * Kw;
    const uint32_t* brow = Bp + (long)n * Kw;
    for (int kw = 0; kw < Kw; ++kw) {
        int a = arow[kw], b = brow[kw];
        asm volatile("v_dot8_i32_iu4 %0, %1, %2, %0" : "+v"(acc) : "v"(a), "v"(b));
    }
    Out[(long)m * N + n] = (float)acc - rowcorr[m];
}
"""

# ---- compile with hiprtc ----
src = ctypes.create_string_buffer(SRC.encode())
prog = ctypes.c_void_p()
st = RTC.hiprtcCreateProgram(ctypes.byref(prog),
                             ctypes.cast(src, ctypes.c_char_p),
                             b"gemm_i4_iu.hip", 0, None, None)
assert st == 0, f"hiprtcCreateProgram {st}"
opts = (ctypes.c_char_p * 2)(b"--offload-arch=gfx1201", b"-O3")
st = RTC.hiprtcCompileProgram(prog, 2, opts)
if st != 0:
    sz = ctypes.c_size_t()
    RTC.hiprtcGetProgramLogSize(prog, ctypes.byref(sz))
    log = ctypes.create_string_buffer(sz.value + 1)
    RTC.hiprtcGetProgramLog(prog, log)
    print("COMPILE LOG:", log.value.decode(errors="replace")[-2500:])
    raise SystemExit(1)
csz = ctypes.c_size_t()
RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
code = ctypes.create_string_buffer(csz.value)
RTC.hiprtcGetCode(prog, code)
print(f"kernel compiled: {csz.value/1e3:.1f} KB", flush=True)

# ---- load module ----
mod = ctypes.c_void_p()
assert HIP.hipModuleLoadData(ctypes.byref(mod), code) == 0
fn = ctypes.c_void_p()
HIP.hipModuleGetFunction(ctypes.byref(fn), mod, b"gemm_i4_iu")

# ---- torch buffers ----
torch.manual_seed(0)
M, N, K = 256, 2048, 768
Kw = K // 8
dev = "cuda"

A = torch.randint(-1, 2, (M, K), device=dev, dtype=torch.int8)       # ternary spikes
Ws = torch.randint(-8, 8, (N, K), device=dev, dtype=torch.int8)      # signed int4 weights
rowcorr = (A.float().sum(1) * 8).contiguous()                        # 8*rowsum(A)

def pack(t):   # value v -> nibble k%8 of word k//8
    out = torch.zeros((t.shape[0], Kw), device=dev, dtype=torch.int64)
    for i in range(8):
        out |= (t[:, i::8].long() & 0xF) << (4 * i)
    return out

Ap = pack(A).to(torch.int32)             # signed nibbles (ternary fits)
Bp = pack(Ws + 8).to(torch.int32)        # unsigned nibbles (s_w + 8)
Out = torch.empty((M, N), device=dev, dtype=torch.float32)

# ---- launch via ctypes ----
storage = [ctypes.c_void_p(Ap.data_ptr()), ctypes.c_void_p(Bp.data_ptr()),
           ctypes.c_void_p(rowcorr.data_ptr()), ctypes.c_void_p(Out.data_ptr()),
           ctypes.c_int32(M), ctypes.c_int32(N), ctypes.c_int32(Kw)]
ptrs = (ctypes.c_void_p * len(storage))(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p)
                                          for b in storage])
blk, grd = (32, 8, 1), ((N + 31) // 32, (M + 7) // 8, 1)
st = HIP.hipModuleLaunchKernel(fn, *grd, *blk, 0, None, ptrs, None)
torch.cuda.synchronize()
assert st == 0, f"launch {st}"

# ---- correctness vs torch reference ----
ref = A.float() @ Ws.float().T
err = (Out - ref).abs().max().item()
print(f"correctness: max|err| = {err:.1f}  {'PASS' if err < 0.5 else 'FAIL'}", flush=True)

# ---- benchmark ----
start = torch.cuda.Event(True); end = torch.cuda.Event(True)
start.record()
for _ in range(200):
    HIP.hipModuleLaunchKernel(fn, *grd, *blk, 0, None, ptrs, None)
end.record(); torch.cuda.synchronize()
print(f"native int4 GEMM (M={M},K={K},N={N}): {start.elapsed_time(end)/200:.3f} ms/call "
      f"(fp32 0.13, int8 0.03)", flush=True)
