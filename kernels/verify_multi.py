"""Isolated 64-k multi-call probe: 2 wmma calls per 64-k chunk, numpy-checked.

Single-call semantics (verified): lane (kt=lane>>4, col=lane&15) holds
A[m=col][k = 16kt .. 16kt+15]; a.x nibble j = k 16kt+j, a.y = 16kt+8+j;
full 32-k consumption per call.

Scheme A (v5b committed):  call s: words (4kt + 2s, +1)
Scheme B (derived):        call s: words (4s + 2kt, +1)
Both give disjoint 64-k coverage on paper; this probe finds which (if
either) the hardware actually computes correctly.
"""
import sys
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, ctypes
import numpy as np
import wmma_gemm_v2 as W
HIP = W.HIP
HIP.hipDeviceSynchronize.restype = ctypes.c_int
HIP.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

SRC = r"""
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
// Frag arrays: 128 ints = 32 lanes x 2 words x 2 calls: f[(c*2 + w) + lane*4]
extern "C" __global__ void probe2(const int* Af, const int* Bf, int* Out, int scheme) {
    int lane = threadIdx.x & 31;
    int col = lane % 16, rbase = (lane / 16) * 8;
    for (int j = 0; j < 8; ++j) Out[(rbase + j) * 16 + col] = 0;
    for (int s = 0; s < 2; ++s) {
        int w0 = (scheme == 0) ? (4 * (lane / 16) + 2 * s)   /* A: per call s */
                               : (4 * s + 2 * (lane / 16));
        v2i a; a.x = Af[lane * 4 + s * 2]; a.y = Af[lane * 4 + s * 2 + 1];
        v2i b; b.x = Bf[lane * 4 + s * 2]; b.y = Bf[lane * 4 + s * 2 + 1];
        v8i acc = {};
        acc = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(0, a, 0, b, acc, 0);
        for (int j = 0; j < 8; ++j) Out[(rbase + j) * 16 + col] += acc[j];
    }
}
"""
mod = W.compile_src(SRC, "probe2")
fn = W.get_fn(mod, "probe2")
Out = torch.zeros(256, device="cuda", dtype=torch.int32)

def launch(Af, Bf, scheme):
    args = [Af, Bf, Out, scheme]
    st = [ctypes.c_void_p(t.data_ptr()) if torch.is_tensor(t) else ctypes.c_int32(t) for t in args]
    p = (ctypes.c_void_p * 4)(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in st])
    Out.zero_()
    HIP.hipModuleLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, None, p, None)
    torch.cuda.synchronize()
    host = np.empty(256, dtype=np.int32)
    HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out.data_ptr()), 1024, 2)
    return host.reshape(16, 16)

torch.manual_seed(0)
K64 = 64
# Host int4 data, unsigned 0..15 for the neg=0 probe
A = torch.randint(0, 16, (16, K64), device="cuda", dtype=torch.int32)
B = torch.randint(0, 16, (K64, 16), device="cuda", dtype=torch.int32)
ref = (A.cpu().numpy().astype(np.int64) @ B.cpu().numpy().astype(np.int64))

def pack_words(t):
    """(R, 64) int4 -> (R, 8) words: word q nibble j = t[r, 8q+j]."""
    R = t.shape[0]
    out = torch.zeros((R, 8), device="cuda", dtype=torch.int64)
    for i in range(8):
        out |= (t[:, i::8].long() & 0xF) << (4 * i)
    return out.to(torch.int32)

Aw = pack_words(A)   # (16, 8) words; row m, word q = k 8q..8q+7
Bw = pack_words(B.t().contiguous())  # (16, 8): row n, word q = k 8q..8q+7

def build_frags(scheme):
    """Fragment arrays for 2 calls; f[lane*4 + s*2 + w] = chunk word per scheme."""
    Af = torch.zeros(128, device="cuda", dtype=torch.int32)
    Bf = torch.zeros(128, device="cuda", dtype=torch.int32)
    for lane in range(32):
        kt, col = lane >> 4, lane & 15
        for s in range(2):
            w0 = (4 * kt + 2 * s) if scheme == 0 else (4 * s + 2 * kt)
            Af[lane * 4 + s * 2 + 0] = Aw[col, w0]
            Af[lane * 4 + s * 2 + 1] = Aw[col, w0 + 1]
            Bf[lane * 4 + s * 2 + 0] = Bw[col, w0]
            Bf[lane * 4 + s * 2 + 1] = Bw[col, w0 + 1]
    return Af, Bf

for scheme, name in [(0, "A: v5b committed (4kt+2s)"), (1, "B: derived (4s+2kt)")]:
    Af, Bf = build_frags(scheme)
    D = launch(Af, Bf, scheme)
    err = np.abs(D.astype(np.int64) - ref).max()
    print(f"scheme {name}: max|err| = {err}  {'PASS' if err == 0 else 'FAIL'}", flush=True)
    if err != 0 and scheme == 1:
        print("  D[0:2,0:2] =", D[:2, :2].tolist(), " ref:", ref[:2, :2].tolist())
