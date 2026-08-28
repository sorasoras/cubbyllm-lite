"""Hypothesis sweep: discover the iu4-WMMA input fragment layout empirically.

Single 16x16x16 tile, wave32. The kernel reads pre-packed lane fragments
(A_frag[lane], B_frag[lane]: 8 int4 nibbles per int) and runs
__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12. The HOST packs fragments
under different layout hypotheses; whichever reproduces torch's int matmul
is the hardware's true mapping.

Fragment layout hypotheses (16x16 tile, lane l, slot j, base=(l/16)*8):
  U: element[base + j][l % 16]   (column-distributed, same as C/D store)
  R: element[l % 16][base + j]   (row-distributed)
Swept independently for A (MxK operand) and B (KxN operand).
"""
import ctypes
import numpy as np
import torch

HIP = ctypes.CDLL(r"B:\git\rocm-venv\Lib\site-packages\_rocm_sdk_core\bin\amdhip64_7.dll")
RTC = ctypes.CDLL(r"B:\git\rocm-venv\Lib\site-packages\_rocm_sdk_core\bin\hiprtc0714.dll")
RTC.hiprtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p,
                                    ctypes.c_char_p, ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_char_p)]
RTC.hiprtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
RTC.hiprtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
RTC.hiprtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
RTC.hiprtcGetCodeSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
RTC.hiprtcGetCode.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
HIP.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
HIP.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
HIP.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]

SRC = r"""
typedef int v8i __attribute__((ext_vector_type(8)));
extern "C" __global__ void tile(const int* __restrict__ Af,
                                const int* __restrict__ Bf,
                                int* __restrict__ Out) {
    int lane = threadIdx.x & 31;
    v8i acc = {};
    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(
        0, Af[lane], 0, Bf[lane], acc, 0);
    int col = lane % 16, rbase = (lane / 16) * 8;
    for (int j = 0; j < 8; ++j)
        Out[(rbase + j) * 16 + col] = acc[j];
}
"""

src = ctypes.create_string_buffer(SRC.encode())
prog = ctypes.c_void_p()
assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(src, ctypes.c_char_p),
                               b"tile.hip", 0, None, None) == 0
opts = (ctypes.c_char_p * 2)(b"--offload-arch=gfx1201", b"-O3")
st = RTC.hiprtcCompileProgram(prog, 2, opts)
if st != 0:
    sz = ctypes.c_size_t(); RTC.hiprtcGetProgramLogSize(prog, ctypes.byref(sz))
    log = ctypes.create_string_buffer(sz.value + 1); RTC.hiprtcGetProgramLog(prog, log)
    print(log.value.decode(errors="replace")[-2000:]); raise SystemExit(1)
csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
mod = ctypes.c_void_p(); HIP.hipModuleLoadData(ctypes.byref(mod), code)
fn = ctypes.c_void_p()
assert HIP.hipModuleGetFunction(ctypes.byref(fn), mod, b"tile") == 0, "GetFunction failed"

def pack(t):
    """(16,16) int tensor -> 32 lane ints, nibble j = element base+j? depends on caller."""
    out = torch.zeros(32, dtype=torch.int32, device="cuda")
    for l in range(32):
        base, col = (l // 16) * 8, l % 16
        for j in range(8):
            out[l] |= (t[base + j, col].to(torch.int64) & 0xF) << (4 * j)
    return out

def pack_R(t):
    out = torch.zeros(32, dtype=torch.int32, device="cuda")
    for l in range(32):
        row, jbase = l % 16, (l // 16) * 8
        for j in range(8):
            out[l] |= (t[row, jbase + j].to(torch.int64) & 0xF) << (4 * j)
    return out

torch.manual_seed(0)
# unsigned values 0..7 keep everything representation-clean for mapping discovery
A = torch.randint(0, 8, (16, 16), device="cuda", dtype=torch.int32)
B = torch.randint(0, 8, (16, 16), device="cuda", dtype=torch.int32)
ref = (A.float() @ B.float()).cpu()   # exact: values <= 784 < 2^24

def run(Af, Bf, neg_a=0, neg_b=0):
    Afh, Bfh = Af.contiguous(), Bf.contiguous()
    Out = torch.zeros(256, dtype=torch.int32, device="cuda")
    s1 = ctypes.c_void_p(Afh.data_ptr()); s2 = ctypes.c_void_p(Bfh.data_ptr())
    s3 = ctypes.c_void_p(Out.data_ptr())
    storage = [s1, s2, s3]
    ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in storage])
    st = HIP.hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, None, ptrs, None)
    torch.cuda.synchronize()
    assert st == 0, f"launch {st}"
    return Out.view(16, 16).cpu()

maps = {"U": pack, "R": pack_R}
print("sweep A_map x B_map (values 0..7, neg=0):")
for am in ("U", "R"):
    for bm in ("U", "R"):
        got = run(maps[am](A), maps[bm](B))
        ok = torch.equal(got, ref)
        diff = (got != ref).sum().item()
        print(f"  A={am} B={bm}: {'PASS' if ok else f'FAIL ({diff}/256 wrong)'}")
