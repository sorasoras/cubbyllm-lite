"""Grouped MoE at production T using the K=32 int4 WMMA kernel."""
import sys
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, ctypes
import numpy as np
import gemm_k32 as G
HIP = G.HIP
T = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
E, N, K = 8, 2048, 768
Kw = K // 8
fn = G.compile_src(G.SRC, "moe_k32")
scale = torch.ones(1, device="cuda")
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
W_all = torch.randint(-8, 8, (E * N, K), device="cuda", dtype=torch.int32)
Ap = G.pack(A_tok).contiguous()
Bt_e = [G.pack_transposed(W_all[eid*N:(eid+1)*N]).contiguous() for eid in range(E)]
Out = torch.empty((M_pad, N), device="cuda", dtype=torch.float32)
host = np.empty(M_pad * N, dtype=np.float32)

def moe_launch():
    for eid in range(E):
        base, rows16, _ = seg_base[eid]
        args = [Ap[base:base+rows16], Bt_e[eid], scale, Out[base:base+rows16], rows16, N, Kw]
        storage = [ctypes.c_void_p(t.data_ptr()) if torch.is_tensor(t) else ctypes.c_int32(t)
                   for t in args]
        ptrs = (ctypes.c_void_p * len(storage))(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in storage])
        st = HIP.hipModuleLaunchKernel(fn, N // 64, rows16 // 16, 1, 32, 1, 1, 2 * 320 * 4, None, ptrs, None)
        assert st == 0

torch.cuda.synchronize()
moe_launch()
s1 = HIP.hipDeviceSynchronize()
s2 = HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out.data_ptr()),
                   M_pad * N * 4, 2)
print(f"launch+sync={s1} memcpy={s2}", flush=True)
Out_np = host.reshape(M_pad, N)
A_np = A_tok.cpu().numpy().astype(np.float32)
W_np = W_all.cpu().numpy().astype(np.float32)
err = 0.0
for eid in range(E):
    base, rows16, n_e = seg_base[eid]
    ref = A_np[base:base+n_e] @ W_np[eid*N:(eid+1)*N].T
    err = max(err, np.abs(Out_np[base:base+n_e] - ref).max())
print(f"grouped MoE K=32 (T={T}): max|err| = {err:.1f}  {'PASS' if err == 0 else 'FAIL'}", flush=True)

gflop = 2 * T * K * N / 1e9
def bench(fnc, n=50):
    fnc(); torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(n): fnc()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n

t_moe = bench(moe_launch)
A_q = A_tok[:T].to(torch.int8).contiguous()
W_q = W_all[:N].to(torch.int8).contiguous()
t_i8 = bench(lambda: torch._int_mm(A_q, W_q.t()))
t_f32 = bench(lambda: A_tok[:T].float() @ W_all[:N].float().T)
print(f"int4 K=32 grouped-MoE: {t_moe:6.3f} ms  {gflop/t_moe:6.2f} TFLOPS")
print(f"int8 _int_mm         : {t_i8:6.3f} ms  {gflop/t_i8:6.2f} TFLOPS")
print(f"fp32 eager           : {t_f32:6.3f} ms  {gflop/t_f32:6.2f} TFLOPS")
print(f"int4 vs fp32: {t_f32/t_moe:.2f}x | vs int8: {t_i8/t_moe:.2f}x")
