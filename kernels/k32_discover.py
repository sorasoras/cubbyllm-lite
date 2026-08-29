"""Discover K=32 iu4 WMMA fragment mapping: single tile, hypothesis sweep."""
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
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
extern "C" __global__ void t(const int* __restrict__ Af,
                             const int* __restrict__ Bf,
                             int* __restrict__ Out) {
    int lane = threadIdx.x & 31;
    v2i a; a.x = Af[lane * 2]; a.y = Af[lane * 2 + 1];
    v2i b; b.x = Bf[lane * 2]; b.y = Bf[lane * 2 + 1];
    v8i acc = {};
    acc = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, a, 1, b, acc, 0);
    int col = lane % 16, rbase = (lane / 16) * 8;
    for (int j = 0; j < 8; ++j)
        Out[(rbase + j) * 16 + col] = acc[j];
}
"""
mod = W.compile_src(SRC, "k32d")
fn = W.get_fn(mod, "t")

torch.manual_seed(0)
A = torch.randint(0, 8, (16, 32), device="cuda", dtype=torch.int32)
B = torch.randint(0, 8, (32, 16), device="cuda", dtype=torch.int32)
ref = (A.float() @ B.float()).cpu()

def frag(t, mode, is_b=False):
    """Pack operand tile into lane fragments. A: (16,32) [m,k]; B: (32,16) [k,n]."""
    out = torch.zeros(64, dtype=torch.int32, device="cuda")
    for l in range(32):
        for w in range(2):
            v = 0
            for j in range(8):
                if mode == "H1":       # contiguous 16: word w covers k = (l/16)*16 + w*8 + j
                    k = (l // 16) * 16 + w * 8 + j
                    val = t[k, l % 16] if is_b else t[l % 16, k] if k < 32 else 0
                elif mode == "H2":     # per-word 8: word w covers k = (l/16)*8 + w*8 + j
                    k = (l // 16) * 8 + w * 8 + j
                    val = (t[k, l % 16] if is_b else t[l % 16, k]) if k < 32 else 0
                elif mode == "H3":     # k = w*16 + (l/16)*8 + j
                    k = w * 16 + (l // 16) * 8 + j
                    val = (t[k, l % 16] if is_b else t[l % 16, k]) if k < 32 else 0
                v |= (val.to(torch.int64) & 0xF) << (4 * j)
            out[l * 2 + w] = v
    return out

print("sweep A_map x B_map for K=32 fragment layout (values 0..7, neg=1):")
for am in ("H1", "H2", "H3"):
    for bm in ("H1", "H2", "H3"):
        Af, Bf = frag(A, am), frag(B, bm, is_b=True)
        Out = torch.zeros(256, dtype=torch.int32, device="cuda")
        args = [Af, Bf, Out]
        storage = [ctypes.c_void_p(t.data_ptr()) for t in args]
        ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in storage])
        HIP.hipModuleLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, None, ptrs, None)
        torch.cuda.synchronize()
        got = Out.view(16, 16).cpu()
        ok = torch.equal(got, ref)
        print(f"  A={am} B={bm}: {'PASS' if ok else 'FAIL (' + str(int((got != ref).sum())) + '/256 wrong)'}", flush=True)
