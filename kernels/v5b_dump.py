"""Dump v5b LDS after load; compare vs host-computed expected words."""
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
#define NT 4
#define KCW 8
extern "C" __global__ void dump(const uint32_t* __restrict__ Ap,
                                const uint32_t* __restrict__ Bt,
                                int* __restrict__ Out,
                                int N, int Kw) {
    extern __shared__ int lds[];
    int n0 = 0;
    int mb = 0;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.y & 3;
    auto load = [&](int kw, int buf) {
        int* LA = lds + buf * 1152;
        int* LB = LA + 576;
        for (int w = threadIdx.y * 32 + lane; w < 64 * 9; w += 128) {
            int r = w / 9, q = w % 9;
            LA[r * 9 + q] = (q < 8) ? Ap[(mb + r) * Kw + kw + q] : 0;
        }
        for (int w = threadIdx.y * 32 + lane; w < KCW * 64; w += 128) {
            int q = w >> 6, nl = w & 63;
            LB[q * 64 + nl] = Bt[(kw + q) * N + n0 + nl];
        }
    };
    load(0, 0);
    __syncthreads();
    // dump A words 0..63 (packed layout) and B words 0..511
    for (int w = threadIdx.y * 32 + lane; w < 64; w += 128)
        Out[w] = lds[w];
    for (int w = threadIdx.y * 32 + lane; w < 512; w += 128)
        Out[64 + w] = lds[576 + w];
}
"""
mod = W.compile_src(SRC, "v5bd")
fn = W.get_fn(mod, "dump")

N, K = 64, 64
Kw = K // 8
M = 64
A = torch.randint(1, 8, (M, K), device="cuda", dtype=torch.int32)
Bt = W.pack_transposed(torch.randint(1, 8, (N, K), device="cuda", dtype=torch.int32)).contiguous()
# host-expected Ap words
Ap_host = torch.zeros(M * Kw, dtype=torch.int64)
for r in range(M):
    for q in range(8):
        Ap_host[r * Kw + q] = W.pack(A[r:r+1]).view(-1)[q].item()
Ap = Ap_host.to(torch.int32).view(M, Kw).contiguous()
Out = torch.zeros(64 + 512, device="cuda", dtype=torch.int32)

W.launch1(fn, (1, 4, 1), [Ap, Bt, Out, N, Kw], shared=2 * 1152 * 4)
s = HIP.hipDeviceSynchronize()
got = Out.cpu().numpy()
a_bad = sum(1 for w in range(64) if got[w] != Ap_host[w].item())
print(f"A LDS words wrong: {a_bad}/64", flush=True)
bt = Bt.cpu().numpy().reshape(-1)
b_bad = sum(1 for w in range(512) if got[64+w] != bt[w])
print(f"B LDS words wrong: {b_bad}/512", flush=True)
if a_bad:
    for w in range(64):
        if got[w] != Ap_host[w].item():
            print(f"  A[{w}] got {got[w]:#x} exp {Ap_host[w].item():#x}", flush=True)
