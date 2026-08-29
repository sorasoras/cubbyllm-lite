import sys
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, ctypes
import numpy as np
import wmma_gemm_v2 as W
HIP = W.HIP; RTC = W.RTC
HIP.hipDeviceSynchronize.restype = ctypes.c_int
HIP.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

# args: init_str join_warp0 join_warp1   (e.g. "s_barrier_init 2" "0" "1")
init_str = sys.argv[1] if len(sys.argv) > 1 else "none"
j0 = sys.argv[2] if len(sys.argv) > 2 else "0"
j1 = sys.argv[3] if len(sys.argv) > 3 else "0"

TMPL = r"""
typedef unsigned int uint;
extern "C" __global__ void probe(int* Out) {
    __shared__ int x;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.y & 1;
    if (threadIdx.x == 0 && threadIdx.y == 0) x = 0;
    __asm__ volatile("s_barrier_signal -1" ::: "memory");   // init-clear? use sync first
    __asm__ volatile("s_barrier_wait -1" ::: "memory");
    if (threadIdx.x == 0 && threadIdx.y == 0) x = 0;
    __asm__ volatile("s_barrier_signal -1" ::: "memory");
    __asm__ volatile("s_barrier_wait -1" ::: "memory");
    INIT
    if (warp == 0) {
        __asm__ volatile("s_barrier_join J0" ::: "memory");
        if (lane == 0) x = 42;
        __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
        __asm__ volatile("s_barrier_signal -1" ::: "memory");
        __asm__ volatile("s_barrier_wait -1" ::: "memory");
    } else {
        __asm__ volatile("s_barrier_join J1" ::: "memory");
        __asm__ volatile("s_barrier_signal -1" ::: "memory");
        __asm__ volatile("s_barrier_wait -1" ::: "memory");
        if (lane == 0) Out[1] = x;
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) Out[0] = 99;
}
"""
src = TMPL.replace("INIT", (f'__asm__ volatile("{init_str}" ::: "memory");' if init_str != "none" else ""))
src = src.replace("J0", j0).replace("J1", j1)

buf = ctypes.create_string_buffer(src.encode())
prog = ctypes.c_void_p()
assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p), b"pj", 0, None, None) == 0
opts = (ctypes.c_char_p * 4)(b"--offload-arch=gfx1201", b"-O3", b"-DNEG_A=1", b"-DNEG_B=1")
assert RTC.hiprtcCompileProgram(prog, 4, opts) == 0, "asm-reject"
csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
m2 = ctypes.c_void_p(); HIP.hipModuleLoadData(ctypes.byref(m2), code)
fn = ctypes.c_void_p(); HIP.hipModuleGetFunction(ctypes.byref(fn), m2, b"probe")
Out = torch.zeros(4, device="cuda", dtype=torch.int32)
st = [ctypes.c_void_p(Out.data_ptr())]
p = (ctypes.c_void_p * 1)(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in st])
HIP.hipModuleLaunchKernel(fn, 1, 1, 1, 32, 2, 1, 64, None, p, None)
HIP.hipDeviceSynchronize()
host = np.zeros(4, dtype=np.int32)
HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out.data_ptr()), 16, 2)
print(f"init={init_str} join0={j0} join1={j1} -> Out={host.tolist()}"
      f"{' [ORDERED]' if host[1] == 42 else ' [independent/unordered]'}", flush=True)
