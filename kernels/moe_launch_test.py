"""Grouped MoE via per-expert launches of the PROVEN dense gemm_i4 kernel.
Correctness (raw hipMemcpy + numpy dequant) + benchmark vs int8/fp32."""
import sys, time
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, ctypes
import numpy as np
import wmma_gemm_v2 as W
HIP = W.HIP
HIP.hipDeviceSynchronize.restype = ctypes.c_int
HIP.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

mod = W.compile_src(W.SRC, "moe_launch")
fn = W.get_fn(mod, "gemm_i4")
scale = torch.ones(1, device="cuda")
E, N, K = 8, 2048, 768
Kw = K // 8
T = 256
torch.manual_seed(0)
assign = torch.randint(0, E, (T,))
order = torch.argsort(assign, stable=True)
counts = torch.bincount(assign, minlength=E)
segs = (counts + 15) // 16
M_pad = int(segs.sum()) * 16
A_tok = torch.zeros(M_pad, K, device="cuda", dtype=torch.int32)
seg_base = []
pos = 0
for eid in range(E):
    idx = order[assign == eid]; n_e = idx.numel()
    A_tok[pos:pos+n_e] = torch.randint(-8, 8, (n_e, K), device="cuda", dtype=torch.int32)
    seg_base.append((pos, int(segs[eid]) * 16, n_e))
    pos += int(segs[eid]) * 16

# per-expert Bt weights: (Kw, N) transposed packing — the layout the kernel requires
W_all = torch.randint(-8, 8, (E * N, K), device="cuda", dtype=torch.int32)
Bt_e = [W.pack_transposed(W_all[eid*N:(eid+1)*N]).contiguous() for eid in range(E)]
Ap = W.pack(A_tok).contiguous()
Out = torch.empty((M_pad, N), device="cuda", dtype=torch.float32)
host = np.empty(M_pad * N, dtype=np.float32)

def moe_launch():
    for eid in range(E):
        base, rows16, _ = seg_base[eid]
        Ap_e = Ap[base:base+rows16]
        args = [Ap_e, Bt_e[eid], scale, Out[base:base+rows16], rows16, N, Kw]
        storage = [ctypes.c_void_p(t.data_ptr()) if torch.is_tensor(t) else ctypes.c_int32(t)
                   for t in args]
        ptrs = (ctypes.c_void_p * len(storage))(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in storage])
        st = HIP.hipModuleLaunchKernel(fn, N // 64, rows16 // 16, 1, 32, 1, 1, 1280, None, ptrs, None)
        assert st == 0

torch.cuda.synchronize()
moe_launch()
s1 = HIP.hipDeviceSynchronize()
s2 = HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out.data_ptr()),
                   M_pad * N * 4, 2)
print(f"launch+sync={s1} memcpy={s2}", flush=True)

# verify via numpy dequant of the transposed packing
Out_np = host.reshape(M_pad, N)
A_np = A_tok.cpu().numpy().astype(np.float32)
Wdeq = np.zeros((E * N, K), dtype=np.float32)
for eid in range(E):
    bt = Bt_e[eid].cpu().numpy()              # (Kw, N)
    for kw in range(Kw):
        for q in range(8):
            v = ((bt[kw] >> (4 * q)) & 0xF).astype(np.float32)
            Wdeq[eid*N:(eid+1)*N, kw*8+q] = np.where(v > 7, v - 16, v)
err = 0.0
for eid in range(E):
    base, rows16, n_e = seg_base[eid]
    ref = A_np[base:base+n_e] @ Wdeq[eid*N:(eid+1)*N].T
    err = max(err, np.abs(Out_np[base:base+n_e] - ref).max())
print(f"grouped MoE (per-expert launches): max|err| = {err:.1f}  {'PASS' if err == 0 else 'FAIL'}", flush=True)

# benchmark: full grouped pass = 256 tokens routed to 8 experts of 2048x768
gflop = 2 * T * K * N / 1e9
def bench(fn_call, n=50):
    fn_call(); torch.cuda.synchronize()
    st_ = torch.cuda.Event(True); en = torch.cuda.Event(True)
    st_.record()
    for _ in range(n): fn_call()
    en.record(); torch.cuda.synchronize()
    return st_.elapsed_time(en) / n

t_moe = bench(moe_launch)
A_q = A_tok[:T].to(torch.int8).contiguous()
W_q = W_all[:N].to(torch.int8).contiguous()
t_i8 = bench(lambda: torch._int_mm(A_q, W_q.t()))
t_f32 = bench(lambda: A_tok[:T].float() @ W_all[:N].float().T)
print(f"int4 grouped-MoE (8 launches): {t_moe:6.3f} ms  {gflop/t_moe:6.2f} TFLOPS")
print(f"int8 _int_mm single GEMM     : {t_i8:6.3f} ms  {gflop/t_i8:6.2f} TFLOPS")
print(f"fp32 eager single GEMM       : {t_f32:6.3f} ms  {gflop/t_f32:6.2f} TFLOPS")
print(f"int4 MoE vs fp32: {t_f32/t_moe:.2f}x | vs int8: {t_i8/t_moe:.2f}x")
