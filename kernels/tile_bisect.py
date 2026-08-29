"""Tile bisect: t=0-only and t=1-only runs vs their own k-block references."""
import sys
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, ctypes
import numpy as np
import gemm_v5b as G
HIP = G.HIP
HIP.hipDeviceSynchronize.restype = ctypes.c_int
HIP.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

fn = G.compile_src(G.SRC, "bisect")   # returns the function handle directly
scale = torch.ones(1, device="cuda")
E, N, K = 2, 128, 128
Kw = K // 8
T = 128
torch.manual_seed(0)
assign = torch.randint(0, E, (T,))
order = torch.argsort(assign, stable=True)
counts = torch.bincount(assign, minlength=E)
segs64 = ((counts + 63) // 64) * 64
M_pad = int(segs64.sum())
A_tok = torch.zeros(M_pad, K, device="cuda", dtype=torch.int32)
seg_base = []
pos = 0
for eid in range(E):
    idx = order[assign == eid]; n_e = idx.numel()
    A_tok[pos:pos+n_e] = torch.randint(-8, 8, (n_e, K), device="cuda", dtype=torch.int32)
    seg_base.append((pos, int(segs64[eid]), n_e))
    pos += int(segs64[eid])
W_all = torch.randint(-8, 8, (E * N, K), device="cuda", dtype=torch.int32)
Ap0 = G.pack(A_tok).contiguous()
Bt0 = G.pack_transposed(W_all).contiguous()
Out = torch.empty((M_pad, N), device="cuda", dtype=torch.float32)
host = np.empty(M_pad * N, dtype=np.float32)

# zero masks: t=0 tile = words (c*8 + 0..3); t=1 tile = words (c*8 + 4..7)
def zero_words(ap, bt, which):
    ap = ap.clone(); bt = bt.clone()
    for c in range(Kw // 8):
        words = range(0, 4) if which == 0 else range(4, 8)
        for w in words:
            ap[:, c * 8 + w] = 0
            bt[c * 8 + w, :] = 0
    return ap, bt

def run(ap, bt):
    for eid in range(E):
        base, rows64, _ = seg_base[eid]
        args = [ap[base:base+rows64], bt[base//1*0:0] if False else bt[:, :].contiguous(), scale,
                Out[base:base+rows64], rows64, N, Kw]
        # Bt is shared (Kw, N) full — slice rows per chunk pattern: Bt covers all words
        args = [ap[base:base+rows64], bt, scale, Out[base:base+rows64], rows64, N, Kw]
        storage = [ctypes.c_void_p(t.data_ptr()) if torch.is_tensor(t) else ctypes.c_int32(t)
                   for t in args]
        ptrs = (ctypes.c_void_p * len(storage))(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in storage])
        st = HIP.hipModuleLaunchKernel(fn, N // 128, rows64 // 64, 1, 32, 4, 1, G.SHARED, None, ptrs, None)
        assert st == 0
    torch.cuda.synchronize()
    HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out.data_ptr()),
                  M_pad * N * 4, 2)
    return host.reshape(M_pad, N).copy()

# k-block masks in HOST k space: t=0 covers k where (k % 64) < 32
mask_t0 = (torch.arange(K) % 64) < 32
A_t0 = torch.zeros_like(A_tok); A_t0[:, mask_t0] = A_tok[:, mask_t0]
W_t0 = torch.zeros_like(W_all); W_t0[:, mask_t0] = W_all[:, mask_t0]
A_t1 = torch.zeros_like(A_tok); A_t1[:, ~mask_t0] = A_tok[:, ~mask_t0]
W_t1 = torch.zeros_like(W_all); W_t1[:, ~mask_t0] = W_all[:, ~mask_t0]

# t=0-only: zero the t=1 words in the PACKED tensors
ap0, bt0 = zero_words(Ap0, Bt0, 1)
got = run(ap0, bt0)
A_ref = A_tok.float().clone(); A_ref[:, ~mask_t0.to(A_tok.device)] = 0
W_ref = W_all.float().clone(); W_ref[:, ~mask_t0.to(A_tok.device)] = 0
err = 0.0
for eid in range(E):
    base, rows64, n_e = seg_base[eid]
    ref = A_ref[base:base+n_e] @ W_ref[eid*N:(eid+1)*N].T
    err = max(err, np.abs(got[base:base+n_e] - ref.cpu().numpy()).max())
print(f"t=0-only: max|err| = {err:.1f}  {'PASS' if err == 0 else 'FAIL'}", flush=True)

# t=1-only
ap1, bt1 = Ap0.clone(), Bt0.clone()
for c in range(Kw // 8):
    for w in range(4):
        ap1[:, c * 8 + w] = 0
        bt1[c * 8 + w, :] = 0
got = run(ap1, bt1)
A_ref1 = A_tok.float().clone(); A_ref1[:, mask_t0.to(A_tok.device)] = 0
W_ref1 = W_all.float().clone(); W_ref1[:, mask_t0.to(A_tok.device)] = 0
err = 0.0
for eid in range(E):
    base, rows64, n_e = seg_base[eid]
    ref = A_ref1[base:base+n_e] @ W_ref1[eid*N:(eid+1)*N].T
    err = max(err, np.abs(got[base:base+n_e] - ref.cpu().numpy()).max())
print(f"t=1-only: max|err| = {err:.1f}  {'PASS' if err == 0 else 'FAIL'}", flush=True)
