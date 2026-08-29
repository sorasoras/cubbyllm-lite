
import sys
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, ctypes
import numpy as np
import wmma_gemm_v2 as W
HIP = W.HIP
RTC = W.RTC
HIP.hipDeviceSynchronize.restype = ctypes.c_int
HIP.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]

SRC = r"""
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
extern "C" __global__ void duel16(float* out, int iters) {
    int lane = threadIdx.x & 31;
    int a = (int)0x13579bdf ^ lane, b = (int)0x2468ace0 | lane;
    v8i acc = {};
    for (int i = 0; i < iters; ++i) {
        v2i av; av.x = a ^ i; av.y = a ^ (i + 7);
        v2i bv; bv.x = b ^ i; bv.y = b ^ (i + 3);
        acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(1, av.x, 1, bv.x, acc, 0);
    }
    out[lane] = (float)(acc[0] + acc[7]);
}
extern "C" __global__ void duel32(float* out, int iters) {
    int lane = threadIdx.x & 31;
    v2i a; a.x = (int)0x13579bdf ^ lane; a.y = (int)0x0f1e2d3c + lane;
    v2i b; b.x = (int)0x2468ace0 | lane; b.y = (int)0x50617a89 - lane;
    v8i acc = {};
    for (int i = 0; i < iters; ++i) {
        v2i av; av.x = a.x ^ i; av.y = a.y ^ (i + 7);
        v2i bv; bv.x = b.x ^ i; bv.y = b.y ^ (i + 3);
        acc = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, av, 1, bv, acc, 0);
    }
    out[lane] = (float)(acc[0] + acc[7]);
}
"""
mod = W.compile_src(SRC, "duel2")
fn16 = W.get_fn(mod, "duel16")
fn32 = W.get_fn(mod, "duel32")
out = torch.zeros(64, device="cuda", dtype=torch.float32)
iters = 20000
storage = [ctypes.c_void_p(out.data_ptr()), ctypes.c_int32(iters)]
ptrs = (ctypes.c_void_p * 2)(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in storage])

def launch16():
    HIP.hipModuleLaunchKernel(fn16, 1, 1, 1, 32, 1, 1, 0, None, ptrs, None)
def launch32():
    HIP.hipModuleLaunchKernel(fn32, 1, 1, 1, 32, 1, 1, 0, None, ptrs, None)

launch16(); torch.cuda.synchronize()
launch32(); torch.cuda.synchronize()

def bench(fn, n=30):
    fn(); torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n

t16 = bench(launch16); t32 = bench(launch32)
macs16 = 16 * 16 * 16 * iters
macs32 = 16 * 16 * 32 * iters
print("K=16: %8.3f ms  %9.1f GMAC/s per wave" % (t16, macs16 / t16 / 1e9))
print("K=32: %8.3f ms  %9.1f GMAC/s per wave" % (t32, macs32 / t32 / 1e9))
print("K=32 / K=16 throughput ratio: %.2fx" % ((macs32 / t32) / (macs16 / t16)))
