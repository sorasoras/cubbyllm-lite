import sys
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, ctypes
import numpy as np
import wmma_gemm_v2 as W
HIP = W.HIP; RTC = W.RTC
HIP.hipDeviceSynchronize.restype = ctypes.c_int
HIP.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

TMPL = r"""
typedef unsigned int uint;
extern "C" __global__ void probe(int* Out) {
    __shared__ int x;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.y & 1;
    if (threadIdx.x == 0 && threadIdx.y == 0) x = 0;
    __syncthreads();
    INIT__
    if (warp == 0) {
        if (lane == 0) x = 42;
        __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
        SIGNAL__
        WAIT__
    } else {
        SIGNAL__
        WAIT__
        if (lane == 0) Out[1] = x;
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) Out[0] = 99;
}
"""

def build(init, sig, wait):
    src = TMPL
    src = src.replace("INIT__", init if init else "")
    src = src.replace("SIGNAL__", f'__asm__ volatile("s_barrier_signal {sig}" ::: "memory");' if sig is not None else "")
    src = src.replace("WAIT__", f'__asm__ volatile("s_barrier_wait {wait}" ::: "memory");' if wait is not None else "")
    return src

def compile_run(src, timeout_s=20):
    buf = ctypes.create_string_buffer(src.encode())
    prog = ctypes.c_void_p()
    if RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p), b"bm", 0, None, None) != 0:
        return "create-fail", None
    opts = (ctypes.c_char_p * 4)(b"--offload-arch=gfx1201", b"-O3", b"-DNEG_A=1", b"-DNEG_B=1")
    if RTC.hiprtcCompileProgram(prog, 4, opts) != 0:
        return "asm-reject", None
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
    return "ok", host.tolist()

print(f"{'init':>22} {'signal':>8} {'wait':>6}  result", flush=True)
results = []
inits = [None, 's_barrier_init 2', 's_barrier_init 4']
sigs = [-1, 0, 1, 2, 32]
waits = [0, 1, 2, -1]
for init in inits:
    for sig in sigs:
        for wait in waits:
            src = build(init, sig, wait)
            status, out = compile_run(src)
            ordered = (status == "ok" and out is not None and out[1] == 42)
            results.append((init, sig, wait, status, out, ordered))
            flag = " <== ORDERED!" if ordered else ""
            print(f"{str(init):>22} {sig:>8} {wait:>6}  {status:10} {out}{flag}", flush=True)
# control: plain __syncthreads handshake
src_ctrl = TMPL.replace("INIT__", "").replace("SIGNAL__", "__syncthreads();").replace("WAIT__", "")
status, out = compile_run(src_ctrl)
print(f"{'control __syncthreads':>22} {'--':>8} {'--':>6}  {status:10} {out}", flush=True)
wins = [r for r in results if r[5]]
print(f"\nORDERED combos: {len(wins)} / {len(results)}")
for r in wins:
    print(f"  init={r[0]} signal={r[1]} wait={r[2]}")
