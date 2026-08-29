import sys
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, ctypes
import numpy as np
import wmma_gemm_v2 as W
HIP = W.HIP; RTC = W.RTC
HIP.hipDeviceSynchronize.restype = ctypes.c_int
HIP.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

# Milestone 1: one warp, one 64-k chunk, chunk body as ONE asm block with
# manually allocated VGPRs, all declared as clobbers (no +v binding):
#   accs v32-95 (8 x v8i), B frags v96-111, A frags via pinned input scalars
#   v112-119, addresses as %0-%3, scratch base %4.
# Zero + compute + store-to-LDS-scratch all inside the asm block.
# v16-proven composition: 4 asm ds_load_2addr_b64 (+v v4i outputs), one
# s_wait_dscnt, then 16 WMMA builtins on C++ accs (no clobber lists).
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
    int mrow0 = col;
    for (int i = lane; i < 24 * AST; i += 32) {
        int r = i / AST, q = i % AST;
        if (q < 8) lds[r * AST + q] = Aw[r * 8 + q];
    }
    for (int i = lane; i < 4 * BST; i += 32) {
        int p = i >> 8, nl = i & 255;
        if (nl < 64) {
            lds[128 * AST + p * BST + nl * 2]     = Bw[(2 * p) * 64 + nl];
            lds[128 * AST + p * BST + nl * 2 + 1] = Bw[(2 * p + 1) * 64 + nl];
        }
    }
    __syncthreads();
    uint aB0 = (uint)(128 * AST + kt * BST + col * 2) * 4;
    uint aB2 = aB0 + 256, aB1 = (uint)(128 * AST + (kt + 2) * BST + col * 2) * 4, aB3 = aB1 + 256;
    v4i fB0, fB1, fB2, fB3;
    __asm__ volatile("ds_load_2addr_b64 %0, %1 offset0:0 offset1:16" : "=v"(fB0) : "v"(aB0));
    __asm__ volatile("ds_load_2addr_b64 %0, %1 offset0:0 offset1:16" : "=v"(fB1) : "v"(aB2));
    __asm__ volatile("ds_load_2addr_b64 %0, %1 offset0:0 offset1:16" : "=v"(fB2) : "v"(aB1));
    __asm__ volatile("ds_load_2addr_b64 %0, %1 offset0:0 offset1:16" : "=v"(fB3) : "v"(aB3));
    __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
    v2i a00 = *(const v2i*)(lds + mrow0 * AST + 2 * kt);
    v2i a01 = *(const v2i*)(lds + mrow0 * AST + 2 * kt + 4);
    v2i a10 = *(const v2i*)(lds + (mrow0 + 8) * AST + 2 * kt);
    v2i a11 = *(const v2i*)(lds + (mrow0 + 8) * AST + 2 * kt + 4);
    const v2i b00 = fB0.xy, b01 = fB0.zw, b02 = fB1.xy, b03 = fB1.zw;
    const v2i b10 = fB2.xy, b11 = fB2.zw, b12 = fB3.xy, b13 = fB3.zw;
    v8i acc[8];
    for (int i = 0; i < 8; ++i) acc[i] = {};
    #pragma unroll
    for (int ng = 0; ng < 4; ++ng) {
        const v2i b = (ng == 0) ? b00 : (ng == 1) ? b01 : (ng == 2) ? b02 : b03;
        acc[0 * 4 + ng] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, a00, 1, b, acc[0 * 4 + ng], 0);
        acc[1 * 4 + ng] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, a10, 1, b, acc[1 * 4 + ng], 0);
    }
    #pragma unroll
    for (int ng = 0; ng < 4; ++ng) {
        const v2i b = (ng == 0) ? b10 : (ng == 1) ? b11 : (ng == 2) ? b12 : b13;
        acc[0 * 4 + ng] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, a01, 1, b, acc[0 * 4 + ng], 0);
        acc[1 * 4 + ng] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, a11, 1, b, acc[1 * 4 + ng], 0);
    }
    int rbase = (lane >> 4) * 8;
    for (int mg = 0; mg < 2; ++mg)
        for (int ng = 0; ng < 4; ++ng)
            for (int j = 0; j < 8; ++j)
                Out[(int)((mg * 16 + 8 * kt + j) * 64 + ng * 16 + col)] = acc[mg * 4 + ng][j];
}
"""

buf = ctypes.create_string_buffer(SRC.encode())
prog = ctypes.c_void_p()
assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p), b"pin2", 0, None, None) == 0
opts = (ctypes.c_char_p * 4)(b"--offload-arch=gfx1201", b"-O3", b"-DNEG_A=1", b"-DNEG_B=1")
assert RTC.hiprtcCompileProgram(prog, 4, opts) == 0
csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
m2 = ctypes.c_void_p(); assert HIP.hipModuleLoadData(ctypes.byref(m2), code) == 0
fn = ctypes.c_void_p(); assert HIP.hipModuleGetFunction(ctypes.byref(fn), m2, b"probe") == 0

torch.manual_seed(0)
A = torch.randint(-8, 8, (24, 64), device="cuda", dtype=torch.int32)
B = torch.randint(-8, 8, (64, 64), device="cuda", dtype=torch.int32)
def pack_rows(t):
    out = torch.zeros((t.shape[0], 8), device="cuda", dtype=torch.int64)
    for i in range(8):
        out |= (t[:, i::8].long() & 0xF) << (4 * i)
    return out.to(torch.int32).contiguous()
Aw = pack_rows(A)
# B fragment words: Bfrag[w, n] = sum_j B[8w+j, n] << 4j  (w = k-word, n = column)
Bv = B.reshape(8, 8, 64).long() & 0xF
Bfrag = torch.zeros(8, 64, device="cuda", dtype=torch.int64)
for j in range(8):
    Bfrag |= (Bv[:, j, :]) << (4 * j)
Bw = Bfrag.to(torch.int32).contiguous()
Out = torch.zeros(16 * 64 * 2, device="cuda", dtype=torch.int32)
args = [Aw, Bw, Out]
st = [ctypes.c_void_p(t.data_ptr()) for t in args]
p = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in st])
HIP.hipModuleLaunchKernel(fn, 1, 1, 1, 32, 1, 1, (128 * 10 + 4 * 256 + 256 * 32) * 4, None, p, None)
HIP.hipDeviceSynchronize()
host = np.zeros(16 * 64 * 2, dtype=np.int32)
HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out.data_ptr()), 16 * 64 * 2 * 4, 2)
D0 = host[:1024].reshape(16, 64)   # mg0: A row mrow0, calls 0+1
D1 = host[1024:].reshape(16, 64)   # mg1: A row mrow0+8
def unpack_signed(wv):
    wh = wv.cpu().numpy().astype(np.uint32)
    out = np.zeros((wh.shape[0], 64), dtype=np.int32)
    for i in range(8):
        out[:, i::8] = (wh >> (4 * i)) & 0xF
    return np.where(out >= 8, out - 16, out)
An = unpack_signed(Aw)
Btrue = B.cpu().numpy().astype(np.int64)   # B[k][n] direct
# lane (kt,col): acc(mg,ng)[j] = sum over call s of A[row(mg)+8kt? ..] -- rows: readback row = mg*8+j... 
# D row index r = mg*8 + j corresponds to A row (mrow0-relative): mg0 -> rows 0-7 = A rows col.. hmm
# Concretely: lane (kt,col) acc[j] -> D[row = rbase + j] where rbase = (lane>>4)*8 = 8kt.
# A fragment for lane (kt,col): row mrow0 = col (mg0) / col+8 (mg1), words 2kt(+4).
# WMMA semantics: D[m][n] accumulates A[mrow0 + m][k] over ALL lanes -> relative m from rbase+j of lane (kt,col):
# row absolute = mrow0_lane + (8kt + j)?? NO: relative row m = 8kt + j for the OWNING lane; A row = mrow0 + m.
# For mg0: A row = col_of_lane + (8kt + j)?? The lane (kt,col) holds A[col][k]; relative row m=8kt+j is held by lane (kt, m):
# its A fragment row = mrow0(lane) = col' = m -> absolute A row = m. So D row m <-> A row m (mg0), m+8 (mg1).
ref0 = An[:16] @ Btrue            # mg0: D row m = A row m, full K
ref1 = An[8:24] @ Btrue           # mg1: D row m = A row m+8, full K
e0 = np.abs(D0.astype(np.int64) - ref0).max()
e1 = np.abs(D1.astype(np.int64) - ref1).max()
print(f"pinned/clobber chunk probe: call0 err={e0}  call1 err={e1}  {'PASS' if e0 == 0 and e1 == 0 else 'FAIL'}")
print('D0[0,:4] =', D0[0,:4].tolist(), ' ref0[0,:4] =', ref0[0,:4].tolist())
print('D0[8,:4] =', D0[8,:4].tolist(), ' ref0[8,:4] =', ref0[8,:4].tolist())
rowerr = np.abs(D0.astype(np.int64) - ref0[:16]).max(axis=1)
colerr = np.abs(D0.astype(np.int64) - ref0[:16]).max(axis=0)
print('row errs:', rowerr.tolist())
print('ng-block errs:', [colerr[i*16:(i+1)*16].max() for i in range(4)])
