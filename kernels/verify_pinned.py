import sys
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, ctypes
import numpy as np
import wmma_gemm_v2 as W
HIP = W.HIP; RTC = W.RTC
HIP.hipDeviceSynchronize.restype = ctypes.c_int
HIP.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

# One warp, one 64-k chunk, FULLY pinned-register chunk body (single asm block).
# Registers: acc0-3 = v32..v63 (8 v8i), B frags v64..v79 (4 v4i), A frags v80..v87.
# This proves manual VGPR allocation survives the compiler -> the assembly campaign's core technique.
ASM_BODY = r"""
typedef unsigned int uint;
typedef int v2i __attribute__((ext_vector_type(2)));
#define AST 10
#define BST 256
extern "C" __global__ void probe(const int* Aw, const int* Bw, int* Out) {
    extern __shared__ int lds[];
    int lane = threadIdx.x & 31;
    int col = lane & 15, kt = lane >> 4;
    int mrow0 = col;
    for (int i = lane; i < 16 * AST; i += 32) {
        int r = i / AST, q = i % AST;
        if (q < 8) lds[r * AST + q] = Aw[r * 8 + q];
    }
    for (int i = lane; i < 4 * BST; i += 32) {
        int p = i >> 8, nl = i & 255;
        if (nl < 64) {
            lds[128 * AST + p * BST + nl * 2]     = Bw[nl * 8 + 2 * p];
            lds[128 * AST + p * BST + nl * 2 + 1] = Bw[nl * 8 + 2 * p + 1];
        }
    }
    __syncthreads();
    uint aB0 = (uint)(128 * AST + kt * BST + col * 2) * 4;
    uint aB2 = aB0 + 256;
    uint aA0 = (uint)(mrow0 * AST + 2 * kt) * 4;
    // pinned scalars
    register int a00_0 asm("v80"), a00_1 asm("v81");
    register int a10_0 asm("v82"), a10_1 asm("v83");
    register int a01_0 asm("v84"), a01_1 asm("v85");
    register int a11_0 asm("v86"), a11_1 asm("v87");
    a00_0 = lds[mrow0 * AST + 2 * kt];     a00_1 = lds[mrow0 * AST + 2 * kt + 1];
    a01_0 = lds[mrow0 * AST + 2 * kt + 4]; a01_1 = lds[mrow0 * AST + 2 * kt + 5];
    a10_0 = lds[(mrow0 + 8) * AST + 2 * kt];     a10_1 = lds[(mrow0 + 8) * AST + 2 * kt + 1];
    a11_0 = lds[(mrow0 + 8) * AST + 2 * kt + 4]; a11_1 = lds[(mrow0 + 8) * AST + 2 * kt + 5];
    register int acc00 asm("v32"); register int acc01 asm("v33");
    register int acc02 asm("v34"); register int acc03 asm("v35");
    register int acc04 asm("v36"); register int acc05 asm("v37");
    register int acc06 asm("v38"); register int acc07 asm("v39");
    register int acc10 asm("v40"); register int acc11 asm("v41");
    register int acc12 asm("v42"); register int acc13 asm("v43");
    register int acc14 asm("v44"); register int acc15 asm("v45");
    register int acc16 asm("v46"); register int acc17 asm("v47");
    acc00 = acc01 = acc02 = acc03 = acc04 = acc05 = acc06 = acc07 = 0;
    acc10 = acc11 = acc12 = acc13 = acc14 = acc15 = acc16 = acc17 = 0;
    __asm__ volatile(
        "ds_load_2addr_b64 v[64:67], %4 offset0:0 offset1:16\n\t"
        "ds_load_2addr_b64 v[68:71], %5 offset0:0 offset1:16\n\t"
        "s_wait_dscnt 0x0\n\t"
        "v_wmma_i32_16x16x32_iu4 v[32:39], v[80:81], v[64:65], v[32:39] neg_lo:[1,1,0]\n\t"
        "v_wmma_i32_16x16x32_iu4 v[40:47], v[80:81], v[66:67], v[40:47] neg_lo:[1,1,0]\n\t"
        "v_wmma_i32_16x16x32_iu4 v[48:55], v[80:81], v[68:69], v[48:55] neg_lo:[1,1,0]\n\t"
        "v_wmma_i32_16x16x32_iu4 v[56:63], v[80:81], v[70:71], v[56:63] neg_lo:[1,1,0]\n\t"
        "v_wmma_i32_16x16x32_iu4 v[32:39], v[82:83], v[72:73], v[32:39] neg_lo:[1,1,0]\n\t"
        "v_wmma_i32_16x16x32_iu4 v[40:47], v[82:83], v[74:75], v[40:47] neg_lo:[1,1,0]\n\t"
        : "+v"(acc00), "+v"(acc01), "+v"(acc02), "+v"(acc03),
          "+v"(acc04), "+v"(acc05), "+v"(acc06), "+v"(acc07),
          "+v"(acc10), "+v"(acc11), "+v"(acc12), "+v"(acc13),
          "+v"(acc14), "+v"(acc15), "+v"(acc16), "+v"(acc17)
        : "v"(aB0), "v"(aB2), "v"(aA0),
          "v"(a00_0), "v"(a00_1), "v"(a10_0), "v"(a10_1),
          "v"(a01_0), "v"(a01_1), "v"(a11_0), "v"(a11_1)
        : "v64", "v65", "v66", "v67", "v68", "v69", "v70", "v71", "memory");
    // dump accs (call0 = k 0-31 with A row mrow0/mrow0+8, B pair-rows kt/kt+2)
    int rbase = (lane >> 4) * 8;
    Out[(rbase + 0) * 64 + col] = acc00; Out[(rbase + 1) * 64 + col] = acc01;
    Out[(rbase + 2) * 64 + col] = acc02; Out[(rbase + 3) * 64 + col] = acc03;
    Out[(rbase + 4) * 64 + col] = acc04; Out[(rbase + 5) * 64 + col] = acc05;
    Out[(rbase + 6) * 64 + col] = acc06; Out[(rbase + 7) * 64 + col] = acc07;
    Out[1024 + (rbase + 0) * 64 + col] = acc10; Out[1024 + (rbase + 1) * 64 + col] = acc11;
    Out[1024 + (rbase + 2) * 64 + col] = acc12; Out[1024 + (rbase + 3) * 64 + col] = acc13;
    Out[1024 + (rbase + 4) * 64 + col] = acc14; Out[1024 + (rbase + 5) * 64 + col] = acc15;
    Out[1024 + (rbase + 6) * 64 + col] = acc16; Out[1024 + (rbase + 7) * 64 + col] = acc17;
}
"""
buf = ctypes.create_string_buffer(ASM_BODY.encode())
prog = ctypes.c_void_p()
assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p), b"pin", 0, None, None) == 0
opts = (ctypes.c_char_p * 4)(b"--offload-arch=gfx1201", b"-O3", b"-DNEG_A=1", b"-DNEG_B=1")
assert RTC.hiprtcCompileProgram(prog, 4, opts) == 0
csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
m2 = ctypes.c_void_p(); assert HIP.hipModuleLoadData(ctypes.byref(m2), code) == 0
fn = ctypes.c_void_p(); assert HIP.hipModuleGetFunction(ctypes.byref(fn), m2, b"probe") == 0

torch.manual_seed(0)
A = torch.randint(-8, 8, (16, 64), device="cuda", dtype=torch.int32)
B = torch.randint(-8, 8, (64, 64), device="cuda", dtype=torch.int32)
def pack_rows(t):
    out = torch.zeros((t.shape[0], 8), device="cuda", dtype=torch.int64)
    for i in range(8):
        out |= (t[:, i::8].long() & 0xF) << (4 * i)
    return out.to(torch.int32).contiguous()
Aw = pack_rows(A); Bw = pack_rows(B.t().contiguous())
Out = torch.zeros(2048, device="cuda", dtype=torch.int32)
args = [Aw, Bw, Out]
st = [ctypes.c_void_p(t.data_ptr()) for t in args]
p = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in st])
HIP.hipModuleLaunchKernel(fn, 1, 1, 1, 32, 1, 1, (128 * 10 + 4 * 256) * 4, None, p, None)
HIP.hipDeviceSynchronize()
host = np.zeros(512, dtype=np.int32)
HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out.data_ptr()), 8192, 2)
D0 = host[:1024].reshape(16, 64)
D1 = host[1024:].reshape(16, 64)
def unpack_signed(w):
    wh = w.cpu().numpy().astype(np.uint32)
    out = np.zeros((wh.shape[0], 64), dtype=np.int32)
    for i in range(8):
        out[:, i::8] = (wh >> (4 * i)) & 0xF
    return np.where(out >= 8, out - 16, out)
An = unpack_signed(Aw); Bn = unpack_signed(Bw); Btrue = Bn.T
# pinned body: call0 = 4 ngs of pair-row kt (k 16kt..16kt+15) with A row mrow0
# lanes kt=0: B pair-row 0 (k 0-15); kt=1: pair-row 1 (k 16-31)
# call1: pair-row kt+2 (k 32-47 / 48-63), A row mrow0+8
ref0 = np.zeros((16, 64), dtype=np.int64)
ref1 = np.zeros((16, 64), dtype=np.int64)
for ktg in range(2):
    ks = slice(16 * ktg, 16 * ktg + 16)
    rows = slice(8 * ktg, 8 * ktg + 8)   # lane readback: kt=1 lanes hold rows 8-15 via rbase
    ref0[rows] += An[rows, ks] @ Btrue[ks]
    ref1[rows] += An[rows + 8, slice(32 + 16 * ktg, 48 + 16 * ktg)] @ Btrue[slice(32 + 16 * ktg, 48 + 16 * ktg)]
e0 = np.abs(D0.astype(np.int64) - ref0).max()
e1 = np.abs(D1.astype(np.int64) - ref1).max()
print(f"pinned-register chunk probe: call0 err={e0}  call1 err={e1}  {'PASS' if e0 == 0 and e1 == 0 else 'FAIL'}")
