"""Minimal NT=8 repro: single block (64x128), single K-chunk (K=64), full compute."""
import sys
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, ctypes
import numpy as np
import wmma_gemm_v2 as W
HIP = W.HIP
RTC = W.RTC
HIP.hipDeviceSynchronize.restype = ctypes.c_int
HIP.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

# v5c kernel body, single chunk (Kw=8 -> one loop iteration, no prefetch)
src = open(r"B:\git\cubbyllm-lite\kernels\gemm_v5c.py", encoding="utf-8").read()
kern = src.split('SRC = r"""')[1].split('"""')[0]
# patch: M/N fixed by args; keep as-is

def compile_src(src, tag):
    buf = ctypes.create_string_buffer(src.encode())
    prog = ctypes.c_void_p()
    assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p),
                                   tag.encode(), 0, None, None) == 0
    opts = (ctypes.c_char_p * 4)(b"--offload-arch=gfx1201", b"-O3", b"-DNEG_A=1", b"-DNEG_B=1")
    assert RTC.hiprtcCompileProgram(prog, 4, opts) == 0, f"compile failed {tag}"
    csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
    code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
    m2 = ctypes.c_void_p(); HIP.hipModuleLoadData(ctypes.byref(m2), code)
    fn = ctypes.c_void_p()
    assert HIP.hipModuleGetFunction(ctypes.byref(fn), m2, b"gemm_i4_v5c") == 0
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

fn = compile_src(kern, "minnt8")
scale = torch.ones(1, device="cuda")
M, N, K = 64, 128, 64
Kw = K // 8
torch.manual_seed(0)
A = torch.randint(1, 8, (M, K), device="cuda", dtype=torch.int32)
B = torch.randint(1, 8, (N, K), device="cuda", dtype=torch.int32)
Ap, Bt = pack(A), pack_transposed(B)
Out = torch.empty((M, N), device="cuda", dtype=torch.float32)
host = np.empty(M * N, dtype=np.float32)

args = [Ap, Bt, scale, Out, M, N, Kw]
storage = [ctypes.c_void_p(t.data_ptr()) if torch.is_tensor(t) else ctypes.c_int32(t) for t in args]
ptrs = (ctypes.c_void_p * len(storage))(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in storage])
st = HIP.hipModuleLaunchKernel(fn, N // 128, M // 64, 1, 32, 4, 1, 2 * 1600 * 4, None, ptrs, None)
torch.cuda.synchronize()
assert st == 0, f"launch {st}"
HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out.data_ptr()), M * N * 4, 2)
got = host.reshape(M, N)
ref = A.float() @ B.float().T
err = np.abs(got - ref.cpu().numpy()).max()
print(f"minimal NT=8 (single block, K=64): max|err| = {err:.1f}  {'PASS' if err == 0 else 'FAIL'}", flush=True)
bad = (got != ref.cpu().numpy())
print("bad:", int(bad.sum()), "/", bad.size, flush=True)
# per-tile-column-group error (NT=8: column groups of 16)
for i in range(8):
    d = np.abs(got[:, i*16:(i+1)*16] - ref.cpu().numpy()[:, i*16:(i+1)*16]).max()
    print(f"  cols [{i*16:3d},{(i+1)*16:3d}): max|err| = {d:.1f}", flush=True)
if bad.any():
    ii = np.argwhere(bad)[:3]
    for r, c in ii:
        print(f"  Out[{r},{c}] = {got[r,c]:.0f}  ref = {ref[r,c].item():.0f}", flush=True)
