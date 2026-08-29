import sys
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, ctypes
import numpy as np
import wmma_gemm_v2 as W
HIP = W.HIP; RTC = W.RTC
HIP.hipDeviceSynchronize.restype = ctypes.c_int
HIP.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

# Exact v7b/v16 chunk geometry: 8 warps, lane=(kt=lane>>4, col=lane&15)
# A: 128 rows x AST 10; B: 4 pair-rows x 256 (b64-pair layout)
SRC = r"""
typedef unsigned int uint;
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
#define AST 10
#define BST 256
extern "C" __global__ void probe(const int* In, int* Out) {
    extern __shared__ int lds[];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.y & 7;
    int col = lane & 15, kt = lane >> 4;
    int mrow0 = (warp >> 1) * 32 + col;
    int nb = (warp & 1) * 128 + col * 2;
    for (int i = threadIdx.y * 32 + lane; i < 128 * 10 + 4 * 256; i += 256)
        lds[i] = i;                       // ramp = int index itself
    __syncthreads();
    // FULL v16 fragment set: asm vs C++ for all 12, all 8 warps
    uint bo = (uint)0;  // buffer 0
    uint aA0 = bo + (uint)(mrow0 * AST + 2 * kt) * 4;
    uint aA1 = bo + (uint)((mrow0 + 16) * AST + 2 * kt) * 4;
    uint aB0 = bo + (uint)(128 * AST + kt * BST + nb) * 4;
    uint aB1 = bo + (uint)(128 * AST + (kt + 2) * BST + nb) * 4;
    uint aB2 = aB0 + 256, aB3 = aB1 + 256;
    v4i fA0, fA1, fB0, fB1, fB2, fB3;
    __asm__ volatile("ds_load_2addr_b64 %0, %1 offset0:0 offset1:2" : "=v"(fA0) : "v"(aA0));
    __asm__ volatile("ds_load_2addr_b64 %0, %1 offset0:0 offset1:2" : "=v"(fA1) : "v"(aA1));
    __asm__ volatile("ds_load_2addr_b64 %0, %1 offset0:0 offset1:16" : "=v"(fB0) : "v"(aB0));
    __asm__ volatile("ds_load_2addr_b64 %0, %1 offset0:0 offset1:16" : "=v"(fB1) : "v"(aB2));
    __asm__ volatile("ds_load_2addr_b64 %0, %1 offset0:0 offset1:16" : "=v"(fB2) : "v"(aB1));
    __asm__ volatile("ds_load_2addr_b64 %0, %1 offset0:0 offset1:16" : "=v"(fB3) : "v"(aB3));
    __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
    // C++ ground truth for the same 12 fragments
    v2i c[12];
    c[0] = *(const v2i*)(lds + mrow0 * AST + 2 * kt);
    c[1] = *(const v2i*)(lds + mrow0 * AST + 2 * kt + 4);
    c[2] = *(const v2i*)(lds + (mrow0 + 16) * AST + 2 * kt);
    c[3] = *(const v2i*)(lds + (mrow0 + 16) * AST + 2 * kt + 4);
    c[4] = *(const v2i*)(lds + 128 * AST + kt * BST + nb);
    c[5] = *(const v2i*)(lds + 128 * AST + kt * BST + nb + 32);
    c[6] = *(const v2i*)(lds + 128 * AST + kt * BST + nb + 64);
    c[7] = *(const v2i*)(lds + 128 * AST + kt * BST + nb + 96);
    c[8] = *(const v2i*)(lds + 128 * AST + (kt + 2) * BST + nb);
    c[9] = *(const v2i*)(lds + 128 * AST + (kt + 2) * BST + nb + 32);
    c[10] = *(const v2i*)(lds + 128 * AST + (kt + 2) * BST + nb + 64);
    c[11] = *(const v2i*)(lds + 128 * AST + (kt + 2) * BST + nb + 96);
    const v4i* f[6] = {&fA0, &fA1, &fB0, &fB1, &fB2, &fB3};
    int o = (warp * 32 + lane) * 48;
    #pragma unroll
    for (int t = 0; t < 6; ++t) {
        const int* fv = (const int*)f[t];
        #pragma unroll
        for (int q = 0; q < 4; ++q) Out[o + t * 4 + q] = fv[q];
    }
    #pragma unroll
    for (int t = 0; t < 12; ++t) {
        Out[o + 24 + t * 2] = c[t].x; Out[o + 24 + t * 2 + 1] = c[t].y;
    }
}
"""
buf = ctypes.create_string_buffer(SRC.encode())
prog = ctypes.c_void_p()
assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p), b"dsgeo", 0, None, None) == 0
opts = (ctypes.c_char_p * 4)(b"--offload-arch=gfx1201", b"-O3", b"-DNEG_A=1", b"-DNEG_B=1")
assert RTC.hiprtcCompileProgram(prog, 4, opts) == 0
csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
m2 = ctypes.c_void_p(); assert HIP.hipModuleLoadData(ctypes.byref(m2), code) == 0
fn = ctypes.c_void_p(); assert HIP.hipModuleGetFunction(ctypes.byref(fn), m2, b"probe") == 0

In = torch.zeros(1, device="cuda", dtype=torch.int32)
Out = torch.zeros(256 * 48, device="cuda", dtype=torch.int32)
args = [In, Out]
st = [ctypes.c_void_p(t.data_ptr()) for t in args]
p = (ctypes.c_void_p * 2)(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in st])
HIP.hipModuleLaunchKernel(fn, 1, 1, 1, 32, 8, 1, (128 * 10 + 4 * 256) * 4, None, p, None)
HIP.hipDeviceSynchronize()
host = np.zeros(256 * 48, dtype=np.int32)
HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out.data_ptr()), 256 * 48 * 4, 2)
host = host.reshape(256, 48)
names = ['a0s0','a0s1','a1s0','a1s1','b00','b01','b02','b03','b10','b11','b12','b13']
bad = {}
for r in range(256):
    for t in range(12):
        asm_v = host[r, (t//2)*4 + (t%2)*2 : (t//2)*4 + (t%2)*2 + 2]
        cpp_v = host[r, 24 + t*2 : 24 + t*2 + 2]
        if not np.array_equal(asm_v, cpp_v):
            bad.setdefault(t, []).append(r)
print(f"per-fragment mismatches (asm vs cpp), 256 lanes:")
for t in range(12):
    print(f"  {names[t]}: {len(bad.get(t, []))}")
for t, rs in bad.items():
    r = rs[0]; lane = r % 32; warp = r // 32
    col = lane & 15; kt = lane >> 4
    print(f"  first {names[t]} bad: warp={warp} lane={lane} col={col} kt={kt} asm={host[r, (t//2)*4+(t%2)*2:(t//2)*4+(t%2)*2+2].tolist()} cpp={host[r, 24+t*2:24+t*2+2].tolist()}")
    break
