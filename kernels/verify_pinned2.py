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
lines = []
lines.append('"v_mov_b32 v32, 0\\n\\t" "v_mov_b32 v33, 0\\n\\t" "v_mov_b32 v34, 0\\n\\t" "v_mov_b32 v35, 0\\n\\t"')
lines.append('"v_mov_b32 v36, 0\\n\\t" "v_mov_b32 v37, 0\\n\\t" "v_mov_b32 v38, 0\\n\\t" "v_mov_b32 v39, 0\\n\\t"')
for base in range(40, 96, 4):
    lines.append('"v_mov_b32 v' + str(base) + ', 0\\n\\t" "v_mov_b32 v' + str(base+1) + ', 0\\n\\t" "v_mov_b32 v' + str(base+2) + ', 0\\n\\t" "v_mov_b32 v' + str(base+3) + ', 0\\n\\t"')
zero = "\n\t".join(lines)

w = []
# loads: fB0=fB(kt,ng0,1) v96-99; fB1 (kt,ng2,3) v100-103; fB2 (kt+2,ng0,1) v104-107; fB3 v108-111
w.append('"ds_load_2addr_b64 v[96:99], %0 offset0:0 offset1:16\\n\\t"')
w.append('"ds_load_2addr_b64 v[100:103], %1 offset0:0 offset1:16\\n\\t"')
w.append('"ds_load_2addr_b64 v[104:107], %2 offset0:0 offset1:16\\n\\t"')
w.append('"ds_load_2addr_b64 v[108:111], %3 offset0:0 offset1:16\\n\\t"')
w.append('"s_wait_dscnt 0x0\\n\\t"')
# call0: A mg0 = v[112:113], A mg1 = v[116:117]; B pair-row kt: ng0 v[96:97], ng1 v[98:99], ng2 v[100:101], ng3 v[102:103]
for ng, bb in [(0, "v[96:97]"), (1, "v[98:99]"), (2, "v[100:101]"), (3, "v[102:103]")]:
    a0 = 32 + ng * 8
    a1 = 64 + ng * 8
    w.append(f'"v_wmma_i32_16x16x32_iu4 v[{a0}:{a0+7}], v[112:113], {bb}, v[{a0}:{a0+7}] neg_lo:[1,1,0]\\n\\t"')
    w.append(f'"v_wmma_i32_16x16x32_iu4 v[{a1}:{a1+7}], v[116:117], {bb}, v[{a1}:{a1+7}] neg_lo:[1,1,0]\\n\\t"')
# call1: A mg0 = v[114:115], A mg1 = v[118:119]; B pair-row kt+2: ng0 v[104:105], ng1 v[106:107], ng2 v[108:109], ng3 v[110:111]
for ng, bb in [(0, "v[104:105]"), (1, "v[106:107]"), (2, "v[108:109]"), (3, "v[110:111]")]:
    a0 = 32 + ng * 8
    a1 = 64 + ng * 8
    w.append(f'"v_wmma_i32_16x16x32_iu4 v[{a0}:{a0+7}], v[114:115], {bb}, v[{a0}:{a0+7}] neg_lo:[1,1,0]\\n\\t"')
    w.append(f'"v_wmma_i32_16x16x32_iu4 v[{a1}:{a1+7}], v[118:119], {bb}, v[{a1}:{a1+7}] neg_lo:[1,1,0]\\n\\t"')
# store accs to scratch: 8 accs x 4 b64 = 32 stores, offsets 0..255 bytes from %4
st = []
for acc in range(8):
    r0 = 32 + acc * 8
    for q in range(4):
        st.append(f'"ds_store_b64 %4, v[{r0+2*q}:{r0+2*q+1}] offset:{acc*32 + q*8}\\n\\t"')
body = "\n\t".join(lines) + "\n\t" + "\n\t".join(w) + "\n\t" + "\n\t".join(st)
clob = ", ".join(f'"v{r}"' for r in list(range(32, 112)))

SRC = r"""
typedef unsigned int uint;
typedef int v2i __attribute__((ext_vector_type(2)));
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
    register int a00_0 asm("v112"), a00_1 asm("v113");
    register int a01_0 asm("v114"), a01_1 asm("v115");
    register int a10_0 asm("v116"), a10_1 asm("v117");
    register int a11_0 asm("v118"), a11_1 asm("v119");
    a00_0 = lds[mrow0 * AST + 2 * kt];           a00_1 = lds[mrow0 * AST + 2 * kt + 1];
    a01_0 = lds[mrow0 * AST + 2 * kt + 4];       a01_1 = lds[mrow0 * AST + 2 * kt + 5];
    a10_0 = lds[(mrow0 + 8) * AST + 2 * kt];     a10_1 = lds[(mrow0 + 8) * AST + 2 * kt + 1];
    a11_0 = lds[(mrow0 + 8) * AST + 2 * kt + 4]; a11_1 = lds[(mrow0 + 8) * AST + 2 * kt + 5];
    uint aB0 = (uint)(128 * AST + kt * BST + col * 2) * 4;
    uint aB2 = aB0 + 256, aB1 = (uint)(128 * AST + (kt + 2) * BST + col * 2) * 4, aB3 = aB1 + 256;
    uint scratch = (uint)(128 * AST + 4 * BST + lane * 64) * 4;
    __asm__ volatile(
BODY
        : : "v"(aB0), "v"(aB2), "v"(aB1), "v"(aB3), "v"(scratch),
            "v"(a00_0), "v"(a00_1), "v"(a10_0), "v"(a10_1),
            "v"(a01_0), "v"(a01_1), "v"(a11_0), "v"(a11_1)
        : CLOB, "memory");
    __syncthreads();
    // scratch: lane*64 ints: acc0 (mg0 ng0) j0-7, acc1 (mg0 ng1), ..., acc7 (mg1 ng3)
    for (int a = 0; a < 8; ++a) {
        int mg = a / 4, ng = a % 4;
        for (int j = 0; j < 8; ++j)
            Out[(int)(mg * 16 + 8 * kt + j) * 64 + ng * 16 + col] = lds[128 * AST + 4 * BST + lane * 64 + a * 8 + j];
    }
}
""".replace("BODY", body).replace("CLOB", clob)

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
ref0 = np.zeros((32, 64), dtype=np.int64)
ref1 = np.zeros((32, 64), dtype=np.int64)
for ktg in range(2):
    ks0 = slice(16 * ktg, 16 * ktg + 16)              # call0, pair-row kt -> k 16kt..+15
    ks1 = slice(32 + 16 * ktg, 48 + 16 * ktg)         # call1, pair-row kt+2
    rows = slice(8 * ktg, 8 * ktg + 8)                # D rows for this kt group (mg0)
    ref0[rows] += An[rows, ks0] @ Btrue[ks0]          # A row = D row (mg0)
    ref1[slice(16 + 8 * ktg, 16 + 8 * ktg + 8)] += An[slice(8 * ktg + 8, 8 * ktg + 16), ks1] @ Btrue[ks1]
e0 = np.abs(D0.astype(np.int64) - ref0[:16]).max()
e1 = np.abs(D1.astype(np.int64) - ref1[16:32]).max()
print(f"pinned/clobber chunk probe: call0 err={e0}  call1 err={e1}  {'PASS' if e0 == 0 and e1 == 0 else 'FAIL'}")
print('D0[0,:4] =', D0[0,:4].tolist(), ' ref0[0,:4] =', ref0[0,:4].tolist())
print('D0[8,:4] =', D0[8,:4].tolist(), ' ref0[8,:4] =', ref0[8,:4].tolist())
rowerr = np.abs(D0.astype(np.int64) - ref0[:16]).max(axis=1)
colerr = np.abs(D0.astype(np.int64) - ref0[:16]).max(axis=0)
print('row errs:', rowerr.tolist())
print('ng-block errs:', [colerr[i*16:(i+1)*16].max() for i in range(4)])
