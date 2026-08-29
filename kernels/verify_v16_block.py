import sys
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, ctypes
import numpy as np
import wmma_gemm_v2 as W
HIP = W.HIP; RTC = W.RTC
HIP.hipDeviceSynchronize.restype = ctypes.c_int
HIP.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

# Full 128x128 block tile, 8 warps, ONE 64-k chunk, v16 asm body, numpy ref.
SRC = r"""
typedef unsigned int uint;
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
#define AST 10
#define BST 256
extern "C" __global__ void probe(const int* Aw, const int* Bw, int* Out) {
    extern __shared__ int lds[];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.y & 7;
    int col = lane & 15, kt = lane >> 4;
    int mrow0 = (warp >> 1) * 32 + col;
    int nb = (warp & 1) * 128 + col * 2;
    for (int i = threadIdx.y * 32 + lane; i < 128 * AST; i += 256) {
        int r = i / AST, q = i % AST;
        if (q < 8) lds[r * AST + q] = Aw[r * 8 + q];
    }
    for (int i = threadIdx.y * 32 + lane; i < 4 * BST; i += 256) {
        int p = i >> 8, nl = i & 255;
        if (nl < 128) {
            lds[128 * AST + p * BST + nl * 2]     = Bw[nl * 8 + 2 * p];
            lds[128 * AST + p * BST + nl * 2 + 1] = Bw[nl * 8 + 2 * p + 1];
        }
    }
    __syncthreads();
    uint bufOff = 0;
    uint aA0 = bufOff + (uint)(mrow0 * AST + 2 * kt) * 4;
    uint aA1 = bufOff + (uint)((mrow0 + 16) * AST + 2 * kt) * 4;
    uint aB0 = bufOff + (uint)(128 * AST + kt * BST + nb) * 4;
    uint aB1 = bufOff + (uint)(128 * AST + (kt + 2) * BST + nb) * 4;
    uint aB2 = aB0 + 256, aB3 = aB1 + 256;
    // A frags via C++ loads (single-use per call); only the reused B frags via asm
    v4i fA0, fA1, fB0, fB1, fB2, fB3;
    fA0.x = lds[mrow0 * AST + 2 * kt];
    fA0.y = lds[mrow0 * AST + 2 * kt + 1];
    fA0.z = lds[mrow0 * AST + 2 * kt + 4];
    fA0.w = lds[mrow0 * AST + 2 * kt + 5];
    fA1.x = lds[(mrow0 + 16) * AST + 2 * kt];
    fA1.y = lds[(mrow0 + 16) * AST + 2 * kt + 1];
    fA1.z = lds[(mrow0 + 16) * AST + 2 * kt + 4];
    fA1.w = lds[(mrow0 + 16) * AST + 2 * kt + 5];
    __asm__ volatile("ds_load_2addr_b64 %0, %1 offset0:0 offset1:16" : "=v"(fB0) : "v"(aB0));
    __asm__ volatile("ds_load_2addr_b64 %0, %1 offset0:0 offset1:16" : "=v"(fB1) : "v"(aB2));
    __asm__ volatile("ds_load_2addr_b64 %0, %1 offset0:0 offset1:16" : "=v"(fB2) : "v"(aB1));
    __asm__ volatile("ds_load_2addr_b64 %0, %1 offset0:0 offset1:16" : "=v"(fB3) : "v"(aB3));
    __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
    v2i a0 = fA0.xy, a1 = fA1.xy;
    const v2i b00 = fB0.xy, b01 = fB0.zw, b02 = fB1.xy, b03 = fB1.zw;
    const v2i b10 = fB2.xy, b11 = fB2.zw, b12 = fB3.xy, b13 = fB3.zw;
    v8i acc[8];
    for (int i = 0; i < 8; ++i) acc[i] = {};
    #pragma unroll
    for (int ng = 0; ng < 4; ++ng) {
        const v2i b = (ng == 0) ? b00 : (ng == 1) ? b01 : (ng == 2) ? b02 : b03;
        acc[0 * 4 + ng] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, a0, 1, b, acc[0 * 4 + ng], 0);
        acc[1 * 4 + ng] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, a1, 1, b, acc[1 * 4 + ng], 0);
    }
    #pragma unroll
    for (int ng = 0; ng < 4; ++ng) {
        const v2i b = (ng == 0) ? b10 : (ng == 1) ? b11 : (ng == 2) ? b12 : b13;
        acc[0 * 4 + ng] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, fA0.zw, 1, b, acc[0 * 4 + ng], 0);
        acc[1 * 4 + ng] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, fA1.zw, 1, b, acc[1 * 4 + ng], 0);
    }
    int rbase = (lane >> 4) * 8;
    for (int mg = 0; mg < 2; ++mg)
        for (int ng = 0; ng < 4; ++ng)
            for (int j = 0; j < 8; ++j)
                Out[(int)((warp >> 1) * 32 + mg * 16 + rbase + j) * 128
                    + (warp & 1) * 64 + ng * 16 + col] = acc[mg * 4 + ng][j];
}
"""
buf = ctypes.create_string_buffer(SRC.encode())
prog = ctypes.c_void_p()
assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p), b"v16b", 0, None, None) == 0
opts = (ctypes.c_char_p * 4)(b"--offload-arch=gfx1201", b"-O3", b"-DNEG_A=1", b"-DNEG_B=1")
assert RTC.hiprtcCompileProgram(prog, 4, opts) == 0
csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
m2 = ctypes.c_void_p(); assert HIP.hipModuleLoadData(ctypes.byref(m2), code) == 0
fn = ctypes.c_void_p(); assert HIP.hipModuleGetFunction(ctypes.byref(fn), m2, b"probe") == 0

torch.manual_seed(0)
K = 64
A = torch.randint(-8, 8, (128, K), device="cuda", dtype=torch.int32)
B = torch.randint(-8, 8, (K, 128), device="cuda", dtype=torch.int32)
def pack_rows(t):
    out = torch.zeros((t.shape[0], 8), device="cuda", dtype=torch.int64)
    for i in range(8):
        out |= (t[:, i::8].long() & 0xF) << (4 * i)
    return out.to(torch.int32).contiguous()
Aw = pack_rows(A)                     # (128, 8)
Bw = pack_rows(B.t().contiguous())    # (128, 8): Bw[n, w] = B[k=8w+j][n]
Out = torch.zeros(128 * 128, device="cuda", dtype=torch.int32)
args = [Aw, Bw, Out]
st = [ctypes.c_void_p(t.data_ptr()) for t in args]
p = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in st])
NW = int(sys.argv[1]) if len(sys.argv) > 1 else 8
HIP.hipModuleLaunchKernel(fn, 1, 1, 1, 32, NW, 1, (128 * 10 + 4 * 256) * 4, None, p, None)
HIP.hipDeviceSynchronize()
host = np.zeros(128 * 128, dtype=np.int32)
HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out.data_ptr()), 128 * 128 * 4, 2)
D = host.reshape(128, 128)
def unpack_signed(w):
    wh = w.cpu().numpy().astype(np.uint32)
    out = np.zeros((wh.shape[0], 64), dtype=np.int32)
    for i in range(8):
        out[:, i::8] = (wh >> (4 * i)) & 0xF
    return np.where(out >= 8, out - 16, out)
An = unpack_signed(Aw)
Bn = unpack_signed(Bw)
Btrue = Bn.T   # Btrue[k][n]
ref = An @ Btrue
mrows = ((NW + 1) // 2) * 32
sub = D[:mrows]
refsub = ref[:mrows]
err = np.abs(sub.astype(np.int64) - refsub)
print(f"warps={NW}: covered {mrows}x128  max|err| = {err.max()}  {'PASS' if err.max() == 0 else 'FAIL'}")
if err.max():
    for mb in range(mrows // 32):
        for nbd in range(2):
            e = err[mb*32:(mb+1)*32, nbd*64:(nbd+1)*64].max()
            print(f"  mb{mb} nband{nbd}: {e}")
    bad = np.argwhere(err > 0)
    r, c = bad[0]
    print(f"  first bad: row {r} (mg {(r%32)//16}, in-16 {r%16}, kt {((r%32)%16)//8}), col {c} (ng {c//16}, in-16 {c%16})")
