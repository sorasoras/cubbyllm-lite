import sys
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, ctypes
import numpy as np
import wmma_gemm_v2 as W
HIP = W.HIP; RTC = W.RTC
HIP.hipDeviceSynchronize.restype = ctypes.c_int
HIP.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

# Probe: 2 warps. Warp0 writes x=42 then signal+wait; warp1 signal+wait then
# reads x. If signal/wait orders across warps (arrive-then-wait semantics),
# warp1 sees 42. Hang => wrong semantics for this use.
SRC = r"""
typedef unsigned int uint;
extern "C" __global__ void probe(int* Out) {
    __shared__ int x;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.y & 1;
    if (threadIdx.x == 0 && threadIdx.y == 0) x = 0;
    __syncthreads();
    if (warp == 0) {
        if (lane == 0) x = 42;
        __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
        __asm__ volatile("s_barrier_signal -1" ::: "memory");
        __asm__ volatile("s_barrier_wait 0" ::: "memory");
    } else {
        __asm__ volatile("s_barrier_signal -1" ::: "memory");
        __asm__ volatile("s_barrier_wait 0" ::: "memory");
        if (lane == 0) Out[1] = x;
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) Out[0] = x;
}
"""
buf = ctypes.create_string_buffer(SRC.encode())
prog = ctypes.c_void_p()
assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p), b"nb", 0, None, None) == 0
opts = (ctypes.c_char_p * 4)(b"--offload-arch=gfx1201", b"-O3", b"-DNEG_A=1", b"-DNEG_B=1")
assert RTC.hiprtcCompileProgram(prog, 4, opts) == 0, "compile"
csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
m2 = ctypes.c_void_p(); assert HIP.hipModuleLoadData(ctypes.byref(m2), code) == 0
fn = ctypes.c_void_p(); assert HIP.hipModuleGetFunction(ctypes.byref(fn), m2, b"probe") == 0
Out = torch.zeros(4, device="cuda", dtype=torch.int32)
st = [ctypes.c_void_p(Out.data_ptr())]
p = (ctypes.c_void_p * 1)(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in st])
r = HIP.hipModuleLaunchKernel(fn, 1, 1, 1, 32, 2, 1, 64, None, p, None)
HIP.hipDeviceSynchronize()
host = np.zeros(4, dtype=np.int32)
HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out.data_ptr()), 16, 2)
print(f"named-barrier probe: launch={r}, Out={host.tolist()} (expect [42,42] if cross-warp ordered)")
