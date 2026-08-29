"""Minimal single-block test of v5b's exact kernel: M=64, N=64, K=768.
If this fails, stage-dump LDS + fragments; if it passes, the bug is in the
v5b harness (grouping/padding/grid).
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

import importlib.util
spec = importlib.util.spec_from_file_location("v5b", r"B:\git\cubbyllm-lite\kernels\gemm_v5b.py")
v5b = importlib.util.module_from_spec(spec)
sys.modules["v5b"] = v5b
# prevent __main__ block
v5b.__name__ = "v5b"
spec.loader.exec_module(v5b) if False else None
# Just grab SRC + compile via own compile path (module __main__ guard stops exec)
import types
src_text = open(r"B:\git\cubbyllm-lite\kernels\gemm_v5b.py").read()
SRC = src_text.split('SRC = r"""')[1].split('"""')[0]

def compile_src(src, tag):
    buf = ctypes.create_string_buffer(src.encode())
    prog = ctypes.c_void_p()
    assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p),
                                   tag.encode(), 0, None, None) == 0
    opts = (ctypes.c_char_p * 4)(b"--offload-arch=gfx1201", b"-O3", b"-DNEG_A=1", b"-DNEG_B=1")
    assert RTC.hiprtcCompileProgram(prog, 4, opts) == 0, f"compile failed {tag}"
    csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
    code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
    m2 = ctypes.c_void_p()
    assert HIP.hipModuleLoadData(ctypes.byref(m2), code) == 0
    fn = ctypes.c_void_p()
    assert HIP.hipModuleGetFunction(ctypes.byref(fn), m2, b"gemm_i4_v4") == 0
    return fn

def pack(t):
    R, K = t.shape
    out = torch.zeros((R, K // 8), device=t.device, dtype=torch.int64)
    for i in range(8):
        out |= (t[:, i::8].long() & 0xF) << (4 * i)
    return out.to(torch.int32).contiguous()

def pack_transposed(t):
    N, K = t.shape
    out = torch.zeros((K // 8, N), device=t.device, dtype=torch.int64)
    for i in range(8):
        out |= (t[:, i::8].t().long() & 0xF) << (4 * i)
    return out.to(torch.int32).contiguous()

SHARED = 2 * 1280 * 4
torch.manual_seed(0)
fn = compile_src(SRC, "v5bmin")
scale = torch.ones(1, device="cuda")
M, N, K = 64, 64, 768
Kw = K // 8
A = torch.randint(-8, 8, (M, K), device="cuda", dtype=torch.int32)
Wt = torch.randint(-8, 8, (N, K), device="cuda", dtype=torch.int32)
Ap = pack(A)
Bt = pack_transposed(Wt)
Out = torch.empty((M, N), device="cuda", dtype=torch.float32)

args = [Ap, Bt, scale, Out, M, N, Kw]
storage = [ctypes.c_void_p(t.data_ptr()) if torch.is_tensor(t) else ctypes.c_int32(t) for t in args]
ptrs = (ctypes.c_void_p * len(storage))(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in storage])
st = HIP.hipModuleLaunchKernel(fn, 1, 1, 1, 32, 4, 1, SHARED, None, ptrs, None)
assert st == 0, st
HIP.hipDeviceSynchronize()
host = np.empty(M * N, dtype=np.float32)
HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out.data_ptr()), M * N * 4, 2)
Out_np = host.reshape(M, N)
ref = A.cpu().numpy().astype(np.float32) @ Wt.cpu().numpy().astype(np.float32).T
err = np.abs(Out_np - ref)
print(f"single-block v5b kernel M=64 N=64 K=768: max|err| = {err.max():.1f}  {'PASS' if err.max() == 0 else 'FAIL'}", flush=True)
if err.max() != 0:
    bad = np.argwhere(err > 0)
    print(f"  {len(bad)}/{err.size} cells wrong; first 5: {bad[:5].tolist()}")
    print(f"  row 0 errs: {err[0, :8].tolist()}")
    print(f"  col 0 errs: {err[:8, 0].tolist()}")
