import sys
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, ctypes
import numpy as np
import wmma_gemm_v2 as W
HIP = W.HIP; RTC = W.RTC
HIP.hipDeviceSynchronize.restype = ctypes.c_int
HIP.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

SRC = r"""
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
extern "C" __global__ void probe(const int* In, int* Out) {
    extern __shared__ int lds[];
    // fill LDS: lds[i] = 1000 + i (per-lane strided fill)
    for (int i = threadIdx.x; i < 256; i += blockDim.x) lds[i] = 1000 + i;
    __syncthreads();
    if (threadIdx.x != 0) return;
    unsigned int a = (unsigned int)(64 * 4);   // base = int index 64
    v4i f;
    __asm__ volatile("ds_load_2addr_b64 %0, %1 offset0:0 offset1:16"
                     : "=v"(f) : "v"(a));
    __asm__ volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    Out[0] = f.x; Out[1] = f.y; Out[2] = f.z; Out[3] = f.w;
}
"""
buf = ctypes.create_string_buffer(SRC.encode())
prog = ctypes.c_void_p()
assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p), b"ds2a", 0, None, None) == 0
opts = (ctypes.c_char_p * 4)(b"--offload-arch=gfx1201", b"-O3", b"-DNEG_A=1", b"-DNEG_B=1")
assert RTC.hiprtcCompileProgram(prog, 4, opts) == 0
csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
m2 = ctypes.c_void_p(); assert HIP.hipModuleLoadData(ctypes.byref(m2), code) == 0
fn = ctypes.c_void_p(); assert HIP.hipModuleGetFunction(ctypes.byref(fn), m2, b"probe") == 0

In = torch.zeros(1, device="cuda", dtype=torch.int32)
Out = torch.zeros(4, device="cuda", dtype=torch.int32)
args = [In, Out]
st = [ctypes.c_void_p(t.data_ptr()) for t in args]
p = (ctypes.c_void_p * 2)(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in st])
st2 = HIP.hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 1024, None, p, None)
HIP.hipDeviceSynchronize()
host = np.zeros(4, dtype=np.int32)
HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out.data_ptr()), 16, 2)
print(f"launch rc={st2}; ds_load_2addr_b64(base=idx 64, off 0/16) -> {host.tolist()}")
print(f"expected: [1064, 1065] (b64 @ 64) and [1080, 1081] (b64 @ 68) if dst[0:1]=off0, dst[2:3]=off1")
