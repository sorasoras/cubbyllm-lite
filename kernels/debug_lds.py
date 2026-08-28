"""Isolate v2 LDS bug: dump the exact fragments gemm_i4 reads, compare to host expectations."""
import sys
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch
import wmma_gemm_v2 as W

SRC_DUMP = r"""
typedef unsigned int uint32_t;
#define NT 4
extern "C" __global__ void dump(const uint32_t* __restrict__ Ap,
                                const uint32_t* __restrict__ Bt,
                                int* __restrict__ Out,
                                int M, int N, int Kw) {
    extern __shared__ int lds[];
    int n0 = blockIdx.x * (16 * NT);
    int mb = blockIdx.y * 16;
    int lane = threadIdx.x & 31;
    int col = lane & 15, kt = lane >> 4;
    auto load = [&](int kw, int buf) {
        int* L = lds + buf * 160;
        for (int w = lane; w < 32; w += 32)
            L[w] = Ap[(mb + w / 2) * Kw + kw + w % 2];
        for (int w = lane; w < 16 * NT * 2; w += 32) {
            int q = w >> 6, rem = w & 63, i = rem >> 4, r = rem & 15;
            L[32 + w] = Bt[(kw + q) * N + n0 + i * 16 + r];
        }
    };
    load(0, 0);
    __syncthreads();
    // dump: per lane: A frag word + NT B frag words (after ONE chunk load, kw=0)
    Out[lane * (NT + 1) + 0] = (int)(lane < 32 ? lds[col * 2 + kt] : 0);
    for (int i = 0; i < NT; ++i)
        Out[lane * (NT + 1) + 1 + i] = lds[32 + kt * 64 + i * 16 + col];
}
"""

mod = W.compile_src(SRC_DUMP, "dump")
fn = W.get_fn(mod, "dump")

# small config: M=16 (one m-block), N=64 (one n-block = 4 tiles), K=16 (one chunk, Kw=2)
M, N, K = 16, 64, 16
A = torch.randint(1, 8, (M, K), device="cuda", dtype=torch.int32)
B = torch.randint(1, 8, (N, K), device="cuda", dtype=torch.int32)
Ap = W.pack(A)                  # (16, 2) int32
Bt = W.pack_transposed(B)       # (2, 64) int32
Out = torch.zeros(32 * 5, device="cuda", dtype=torch.int32)
W.launch1(fn, (1, 1, 1), [Ap, Bt, Out, M, N, K])

# host expectations (single chunk kw=0)
exp_a = [Ap.view(-1)[(l & 15) * 2 + (l >> 4)].item() for l in range(32)]  # A frag: row col, word kt
exp_b = []
for l in range(32):
    kt, col = l >> 4, l & 15
    exp_b.append([Bt.view(-1)[kt * N + i * 16 + col].item() for i in range(4)])

got = Out.cpu().view(32, 5)
ok = True
for l in range(32):
    col, kt = l & 15, l >> 4
    ea = Ap.view(-1)[col * 2 + kt].item()
    if got[l, 0].item() != ea:
        print(f"A MISMATCH lane {l}: got {got[l,0].item()} exp {ea}")
        ok = False
    for i in range(4):
        eb = Bt.view(-1)[kt * N + i * 16 + col].item()
        if got[l, 1 + i].item() != eb:
            print(f"B MISMATCH lane {l} tile {i}: got {got[l,1+i].item()} exp {eb}")
            ok = False
print("FRAGMENT DUMP:", "PASS — load/decode path is correct, bug is elsewhere" if ok else "FAIL — load/decode wrong")
