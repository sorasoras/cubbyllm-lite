"""Per-nibble position probe for the K=32 iu4 WMMA builtin.

For each candidate nibble position (lane 0-1, word 0-1, nibble 0-7):
  - B-probe: that nibble=1 in B (all else 0), A = two k-ramps.
    D[0][n] nonzero at exactly one column -> n-mapping; the VALUE via
    ramp_lo gives k%16, ramp_hi gives which 16-block.
  - A-probe: symmetric for the A operand (m-mapping + k).
Sweeps 64 B positions and 64 A positions (2 ramps each) = 256 launches.
Output: the complete observed fragment -> (operand-index, k, n-or-m) map.
"""
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
extern "C" __global__ void probe(const int* Af, const int* Bf, int* Out) {
    int lane = threadIdx.x & 31;
    v2i a; a.x = Af[lane * 2]; a.y = Af[lane * 2 + 1];
    v2i b; b.x = Bf[lane * 2]; b.y = Bf[lane * 2 + 1];
    v8i acc = {};
    acc = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(0, a, 0, b, acc, 0);
    int col = lane % 16, rbase = (lane / 16) * 8;
    for (int j = 0; j < 8; ++j) Out[(rbase + j) * 16 + col] = acc[j];
}
"""
mod = W.compile_src(SRC, "nibbleprobe")
fn = W.get_fn(mod, "probe")
Out = torch.zeros(256, device="cuda", dtype=torch.int32)
storage = [ctypes.c_void_p(Out.data_ptr())]
ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.byref(storage[0]), ctypes.c_void_p),
                               ctypes.c_void_p(0), ctypes.c_void_p(0)])

def launch(Af, Bf):
    args = [Af, Bf, Out]
    st = [ctypes.c_void_p(t.data_ptr()) for t in args]
    p = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in st])
    HIP.hipModuleLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, None, p, None)
    torch.cuda.synchronize()
    host = np.empty(256, dtype=np.int32)
    HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out.data_ptr()), 1024, 2)
    return host.reshape(16, 16)

K = 32
ramp_lo = torch.zeros(16, K, device="cuda", dtype=torch.int32)
ramp_hi = torch.zeros(16, K, device="cuda", dtype=torch.int32)
for k in range(K):
    ramp_lo[:, k] = k % 16
    ramp_hi[:, k] = (k // 16) + 1
B_ramp_lo = torch.zeros(K, 16, device="cuda", dtype=torch.int32)
B_ramp_hi = torch.zeros(K, 16, device="cuda", dtype=torch.int32)
for k in range(K):
    B_ramp_lo[k, :] = k % 16
    B_ramp_hi[k, :] = (k // 16) + 1     # 1 for k<16, 2 for k>=16

def zero_frag():
    return torch.zeros(64, dtype=torch.int32, device="cuda")

def set_nibble(frag, l, w, j):
    f = frag.clone()
    f[l * 2 + w] |= (1 << (4 * j))
    return f

# --- B sweep: for each (lane 0-1, word 0-1, nibble 0-7), with A = ramps ---
print("B-nibble mapping (lane, word, nib) -> (n, k_lo, k_hi):", flush=True)
b_map = {}
for l in range(2):
    for w in range(2):
        for j in range(8):
            bf = set_nibble(zero_frag(), l, w, j)
            z = zero_frag()
            D1 = launch(ramp_lo, bf)
            D2 = launch(ramp_hi, bf)
            nz1 = np.argwhere(D1 != 0)
            nz2 = np.argwhere(D2 != 0)
            if len(nz1):
                r, n = int(nz1[0][0]), int(nz1[0][1])
                klo = int(D1[r, n]) - 0
                khi = int(D2[r, n])
                b_map[(l, w, j)] = (n, klo, khi - 1)
            else:
                b_map[(l, w, j)] = None
# compact print: lane0 only + dedupe
for l in range(2):
    for w in range(2):
        cells = []
        for j in range(8):
            m = b_map[(l, w, j)]
            cells.append(f"j{j}:n{m[0]},k{m[1] if m[1] is not None else '?'},{m[2]}" if m else f"j{j}:None")
        print(f"  B lane{l} w{w}: " + " | ".join(cells), flush=True)

# --- A sweep: for each (lane 0-1, word 0-1, nibble 0-7), with B = ramps ---
print("A-nibble mapping (lane, word, nib) -> (m, k_lo, k_hi):", flush=True)
a_map = {}
for l in range(2):
    for w in range(2):
        for j in range(8):
            af = set_nibble(zero_frag(), l, w, j)
            D1 = launch(af, B_ramp_lo)
            D2 = launch(af, B_ramp_hi)
            nz1 = np.argwhere(D1 != 0)
            if len(nz1):
                r, n = int(nz1[0][0]), int(nz1[0][1])
                klo = int(D1[r, n])
                khi = int(D2[r, n])
                a_map[(l, w, j)] = (r, klo, khi - 1)
            else:
                a_map[(l, w, j)] = None
for l in range(2):
    for w in range(2):
        cells = []
        for j in range(8):
            m = a_map[(l, w, j)]
            cells.append(f"j{j}:m{m[0]},k{m[1] if m[1] is not None else '?'},{m[2]}" if m else f"j{j}:None")
        print(f"  A lane{l} w{w}: " + " | ".join(cells), flush=True)
