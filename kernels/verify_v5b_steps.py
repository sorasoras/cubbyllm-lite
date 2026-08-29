"""Incremental harness bisection for v5b: single-block PASS already shown.
Step A: N=2048, grid (32,1) — multi-N-block.
Step B: + 8 expert launches with sliced Ap/Out views (T=64 harness replica).
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
K, Kw = 768, 96
E, N = 8, 2048

def run_grid(Ap, Bt, Out, M_rows, grid, tag):
    args = [Ap, Bt, scale, Out, M_rows, N, Kw]
    storage = [ctypes.c_void_p(t.data_ptr()) if torch.is_tensor(t) else ctypes.c_int32(t) for t in args]
    ptrs = (ctypes.c_void_p * len(storage))(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in storage])
    st = HIP.hipModuleLaunchKernel(fn, grid[0], grid[1], 1, 32, 4, 1, SHARED, None, ptrs, None)
    assert st == 0, st
    HIP.hipDeviceSynchronize()

def check(Out, A_np, W_np, tag):
    host = np.empty(Out.numel(), dtype=np.float32)
    HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out.data_ptr()), Out.numel() * 4, 2)
    On = host.reshape(Out.shape)
    err = np.abs(On - A_np @ W_np.T)
    print(f"{tag}: max|err| = {err.max():.1f}  {'PASS' if err.max() == 0 else 'FAIL'}", flush=True)
    if err.max() != 0:
        bad = np.argwhere(err > 1e-3)
        rows_wrong = sorted(set(bad[:, 0].tolist()))
        cols_wrong = sorted(set(bad[:, 1].tolist()))
        print(f"   wrong rows ({len(rows_wrong)}): {rows_wrong[:8]}...  wrong col-blocks: {sorted(set(c // 64 for c in cols_wrong))}")
        r, c = bad[0]
        print(f"   e.g. [{r},{c}]: got {On[r, c]:.1f} want {A_np[r] @ W_np[c]}")
    return err.max()

# Step A: N=2048, single 64-row M-block, full-width output, ONE launch
A = torch.randint(-8, 8, (64, K), device="cuda", dtype=torch.int32)
Wt = torch.randint(-8, 8, (N, K), device="cuda", dtype=torch.int32)
Ap = pack(A); Bt = pack_transposed(Wt)
Out = torch.empty((64, N), device="cuda", dtype=torch.float32)
run_grid(Ap, Bt, Out, 64, (N // 64, 1), "A: N=2048 grid(32,1) one launch")
A_np = A.cpu().numpy().astype(np.float32); W_np = Wt.cpu().numpy().astype(np.float32)
e = check(Out, A_np, W_np, "A")

if e == 0:
    # Step B: replicate T=64 harness: 8 experts, M_pad=512, sliced views
    T = 64
    assign = torch.randint(0, E, (T,))
    counts = torch.bincount(assign, minlength=E)
    segs64 = ((counts + 63) // 64) * 64
    M_pad = int(segs64.sum())
    A_tok = torch.zeros(M_pad, K, device="cuda", dtype=torch.int32)
    W_all = torch.randint(-8, 8, (E * N, K), device="cuda", dtype=torch.int32)
    Ap2 = pack(A_tok)
    Bt_e = [pack_transposed(W_all[eid * N:(eid + 1) * N]).contiguous() for eid in range(E)]
    Out2 = torch.empty((M_pad, N), device="cuda", dtype=torch.float32)
    seg_base = []; pos = 0
    for eid in range(E):
        n_e = int(counts[eid])
        A_tok[pos:pos + n_e] = torch.randint(-8, 8, (n_e, K), device="cuda", dtype=torch.int32)
        seg_base.append((pos, int(segs64[eid]), n_e)); pos += int(segs64[eid])
    Ap2 = pack(A_tok)
    A_np2 = A_tok.cpu().numpy().astype(np.float32)
    W_np2 = W_all.cpu().numpy().astype(np.float32)
    worst = 0.0
    for eid in range(E):
        base, rows64, n_e = seg_base[eid]
        run_grid(Ap2[base:base + rows64], Bt_e[eid], Out2[base:base + rows64], rows64, (N // 64, rows64 // 64), f"B eid{eid}")
        host = np.empty(rows64 * N, dtype=np.float32)
        HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out2[base:base+rows64].data_ptr()), rows64 * N * 4, 2)
        On = host.reshape(rows64, N)
        ref = A_np2[base:base + n_e] @ W_np2[eid * N:(eid + 1) * N].T
        err = np.abs(On[:n_e] - ref)
        worst = max(worst, err.max())
        if err.max() > 0:
            bad = np.argwhere(err > 1e-3)
            cb = sorted(set((bad[:, 1] // 64).tolist()))
            print(f"   eid{eid}: err={err.max():.1f} wrong N-blocks: {cb}")
    print(f"B: 8-expert harness replica: max|err| = {worst:.1f}  {'PASS' if worst == 0 else 'FAIL'}", flush=True)
