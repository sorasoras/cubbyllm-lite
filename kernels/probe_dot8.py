"""Probe V_DOT8_I32_IU4 signedness semantics with known nibble patterns."""
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
typedef unsigned int uint32_t;
extern "C" __global__ void probe(const uint32_t* __restrict__ in,
                                 int* __restrict__ out) {
    // in: [a0, b0, a1, b1, a2, b2, a3, b3] ; out: 4 raw dot results
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        asm volatile("v_dot8_i32_iu4 %0, %1, %2, %0" : "+v"(acc0) : "v"(in[0]), "v"(in[1]));
        asm volatile("v_dot8_i32_iu4 %0, %1, %2, %0" : "+v"(acc1) : "v"(in[2]), "v"(in[3]));
        // swapped operand order (tests whether d = s2 + dot(s1, s0) instead)
        asm volatile("v_dot8_i32_iu4 %0, %1, %2, %0" : "+v"(acc2) : "v"(in[4]), "v"(in[5]));
        asm volatile("v_dot8_i32_iu4 %0, %1, %2, %0" : "+v"(acc3) : "v"(in[6]), "v"(in[7]));
        out[0] = acc0; out[1] = acc1; out[2] = acc2; out[3] = acc3;
    }
}
"""

src = ctypes.create_string_buffer(SRC.encode())
prog = ctypes.c_void_p()
assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(src, ctypes.c_char_p),
                               b"probe.hip", 0, None, None) == 0
opts = (ctypes.c_char_p * 2)(b"--offload-arch=gfx1201", b"-O3")
assert RTC.hiprtcCompileProgram(prog, 2, opts) == 0, "compile failed"
csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
mod = ctypes.c_void_p(); HIP.hipModuleLoadData(ctypes.byref(mod), code)
fn = ctypes.c_void_p(); HIP.hipModuleGetFunction(ctypes.byref(fn), mod, b"probe")

def nib(vals):   # pack 8 nibble values (0..15) into one uint32, element i -> bits 4i
    r = 0
    for i, v in enumerate(vals):
        r |= (v & 0xF) << (4 * i)
    return r

# cases (all 8 nibbles identical):
#   a=0xF (signed -1 / unsigned 15), b=0x1 (signed +1 / unsigned 1)
#   a=0x1, b=0x9 (signed -7 / unsigned 9)
#   a=0x8 (signed -8 / unsigned 8), b=0x7 (signed 7 / unsigned 7)
cases = [
    (nib([0xF]*8), nib([0x1]*8), "a=0xF,b=0x1"),
    (nib([0x1]*8), nib([0x9]*8), "a=0x1,b=0x9"),
    (nib([0x8]*8), nib([0x7]*8), "a=0x8,b=0x7"),
    (nib([0x1]*8), nib([0x1]*8), "a=0x1,b=0x1"),
]
import numpy as _np
vals = _np.array([c[0] for c in cases] + [c[1] for c in cases], dtype=_np.uint32)
inp = torch.from_numpy(vals.view(_np.int32)).to("cuda")
out = torch.zeros(4, dtype=torch.int32, device="cuda")

s_in = ctypes.c_void_p(inp.data_ptr()); s_out = ctypes.c_void_p(out.data_ptr())
storage = [s_in, s_out]
ptrs = (ctypes.c_void_p * 2)(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in storage])
HIP.hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, None, ptrs, None)
torch.cuda.synchronize()

res = out.cpu().tolist()
print("case           | hardware raw | s*a*u*b | u*a*u*b | s*a*s*b | u*a*s*b")
def nib_signed(v):  return v - 16 if v >= 8 else v
for (a, b, name), r in zip(cases, res):
    av, bv = a & 0xF, b & 0xF
    s = lambda x: nib_signed(x)
    print(f"{name:14s} | {r:12d} | {8*s(av)*bv:7d} | {8*av*bv:7d} | {8*s(av)*s(bv):7d} | {8*av*s(bv):7d}")
