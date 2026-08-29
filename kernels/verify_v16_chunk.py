import sys
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, ctypes
import numpy as np
import wmma_gemm_v2 as W
HIP = W.HIP; RTC = W.RTC
HIP.hipDeviceSynchronize.restype = ctypes.c_int
HIP.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

# One warp, one 64-k chunk, v16 asm-scheduled body vs numpy. 8 output n-tiles? keep 4 like v7b (NT=4 cols = 64).
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
    int col = lane & 15, kt = lane >> 4;
    int mrow0 = col;                    // single warp: rows 0..15 (call rows +16 unused -> use row+8 variant below)
    // stage: A words (16 rows x 8 words) at row*AST; B 8 pair-words x 16 cols at 128*AST + p*BST + c*2
    for (int i = lane; i < 16 * AST; i += 32) {
        int r = i / AST, q = i % AST;
        if (q < 8) lds[r * AST + q] = Aw[r * 8 + q];
    }
    for (int i = lane; i < 4 * BST; i += 32) {
        int p = i >> 8, nl = i & 255;
        if (nl < 16) {
            lds[128 * AST + p * BST + nl * 2]     = Bw[nl * 8 + 2 * p];      // Bw[n, w]
            lds[128 * AST + p * BST + nl * 2 + 1] = Bw[nl * 8 + 2 * p + 1];
        }
    }
    __syncthreads();
    int nb = col * 2;
    uint aA0 = (uint)(mrow0 * AST + 2 * kt) * 4;
    uint aB0 = (uint)(128 * AST + kt * BST + nb) * 4;
    uint aB2 = aB0 + 256;
    v4i fA0, fB0, fB1;
#ifdef USE_CPP_LOADS
    fA0 = *(const v4i*)(lds + mrow0 * AST + 2 * kt);   // WRONG layout on purpose? no:
#endif
    fA0.x = *(const int*)(lds + mrow0 * AST + 2 * kt);
    fA0.y = *(const int*)(lds + mrow0 * AST + 2 * kt + 1);
    fA0.z = *(const int*)(lds + mrow0 * AST + 2 * kt + 4);
    fA0.w = *(const int*)(lds + mrow0 * AST + 2 * kt + 5);
    fB0.x = *(const int*)(lds + 128 * AST + kt * BST + nb);
    fB0.y = *(const int*)(lds + 128 * AST + kt * BST + nb + 1);
    fB0.z = *(const int*)(lds + 128 * AST + kt * BST + nb + 32);
    fB0.w = *(const int*)(lds + 128 * AST + kt * BST + nb + 33);
    fB1.x = *(const int*)(lds + 128 * AST + kt * BST + nb + 64);
    fB1.y = *(const int*)(lds + 128 * AST + kt * BST + nb + 65);
    fB1.z = *(const int*)(lds + 128 * AST + kt * BST + nb + 96);
    fB1.w = *(const int*)(lds + 128 * AST + kt * BST + nb + 97);
    __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
    v2i a0 = fA0.xy;
    const v2i b00 = fB0.xy, b01 = fB0.zw, b02 = fB1.xy, b03 = fB1.zw;
    v8i acc[4] = {};
    {
        const v2i b = b00;   // N=16 -> single n-tile
        acc[0] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, a0, 1, b, acc[0], 0);
    }
    // also call 1 (words 2kt+4, pair-row kt+2)
    v2i a1 = fA0.zw;
    uint aB1 = (uint)(128 * AST + (kt + 2) * BST + nb) * 4;
    v4i fB2;
    fB2.x = *(const int*)(lds + 128 * AST + (kt + 2) * BST + nb);
    fB2.y = *(const int*)(lds + 128 * AST + (kt + 2) * BST + nb + 1);
    fB2.z = *(const int*)(lds + 128 * AST + (kt + 2) * BST + nb + 32);
    fB2.w = *(const int*)(lds + 128 * AST + (kt + 2) * BST + nb + 33);
    __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
    const v2i b10 = fB2.xy, b11 = fB2.zw;
    {
        const v2i b = b10;
        acc[0] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, a1, 1, b, acc[0], 0);
    }
    int rbase = (lane >> 4) * 8;
    for (int j = 0; j < 8; ++j)
        Out[(rbase + j) * 16 + col] = acc[0][j];
}
"""
buf = ctypes.create_string_buffer(SRC.encode())
prog = ctypes.c_void_p()
assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p), b"v16c", 0, None, None) == 0
opts = (ctypes.c_char_p * 4)(b"--offload-arch=gfx1201", b"-O3", b"-DNEG_A=1", b"-DNEG_B=1")
assert RTC.hiprtcCompileProgram(prog, 4, opts) == 0
csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
m2 = ctypes.c_void_p(); assert HIP.hipModuleLoadData(ctypes.byref(m2), code) == 0
fn = ctypes.c_void_p(); assert HIP.hipModuleGetFunction(ctypes.byref(fn), m2, b"probe") == 0

torch.manual_seed(0)
# int4 signed data via nibbles, neg=1: use -8..7
A = torch.randint(-8, 8, (16, 64), device="cuda", dtype=torch.int32)
B = torch.randint(-8, 8, (64, 16), device="cuda", dtype=torch.int32)
def pack_rows(t):   # (R, 64) -> (R, 8) words
    out = torch.zeros((t.shape[0], 8), device="cuda", dtype=torch.int64)
    for i in range(8):
        out |= (t[:, i::8].long() & 0xF) << (4 * i)
    return out.to(torch.int32).contiguous()
Aw = pack_rows(A)                       # (16, 8)
Bw = pack_rows(B.t().contiguous())      # (16, 8) per pair-row p: words 2p,2p+1 x 16 cols
Out = torch.zeros(256, device="cuda", dtype=torch.int32)
args = [Aw, Bw, Out]
st = [ctypes.c_void_p(t.data_ptr()) for t in args]
p = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in st])
HIP.hipModuleLaunchKernel(fn, 1, 1, 1, 32, 1, 1, (128 * 10 + 4 * 256) * 4, None, p, None)
HIP.hipDeviceSynchronize()
host = np.zeros(256, dtype=np.int32)
HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out.data_ptr()), 1024, 2)
D = host.reshape(16, 16)
# reference: D[m][n] = sum_k A[m][k]*B[k][n] with signed int4; call0 covers k 0-15+32-47? -- emulate the v16 call set:
# call0: A words (2kt,2kt+1) per lane-group kt -> k 0-15; B pair-row kt -> k 0-15. call1: A words (2kt+4,+5) -> k 16-31; B pair-row kt+2 -> k 32-47.
def unpack_words(w):  # (R,8) words -> (R, 64) nibbles signed
    R = w.shape[0]
    out = np.zeros((R, 64), dtype=np.int32)
    wh = w.cpu().numpy().astype(np.uint32)
    for i in range(8):
        out[:, i::8] = (wh[:, :] >> (4 * i)) & 0xF
    return np.where(out >= 8, out - 16, out)
An = unpack_words(Aw); Bn = unpack_words(Bw)   # Bn[p, c*?]: Bn row p = word p, nibble j = B[k=8p+j][c]
# B_true[k][n] = Bn[k//8, n] with nibble k%8 -> rebuild
Btrue = Bn.T.copy()   # Btrue[k][n] = Bn[n][k]
# call0 contributes k 0-15 with B rows 0-15; call1: A k 16-31 (words 2kt+4 -> k=8*(2kt+4)+j), B pair-row kt+2 -> rows 32-47 -> k 32-47
ref = np.zeros((16, 16), dtype=np.int64)
ref += An[:, 0:32] @ Btrue[0:32]        # call0: k 0-31 (kt0->0-15, kt1->16-31)
ref_call1 = An[:, 32:64] @ Btrue[32:64] # call1: k 32-63
ref += ref_call1                        # call1 covers all 16 cols (single n-tile)
err = np.abs(D.astype(np.int64) - ref).max()
print(f"v16 chunk probe: max|err| = {err}  {'PASS' if err == 0 else 'FAIL'}")
if err:
    print("D[0:2,:4] =", D[:2, :4].tolist())
    print("ref[0:2,:4] =", ref[:2, :4].tolist())
