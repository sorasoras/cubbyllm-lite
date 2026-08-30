"""INT4 dual-lane: primal + tangent GEMMs both through the gfx1201 WMMA kernel.

The gate experiment for fully-int4 backprop-free training:
  y  = int4kernel(W_q, X)     primal lane (deployed model, QAT built-in)
  dy = int4kernel(E_q, X)     tangent lane (STE: E is the rank-r parameter
                               direction; dy = E x uses the PRIMAL x)
  d  = <P, dy>                directional derivative of L = <P, y>
Reference: exact torch.func.jvp on the dequantized fp32 layer.
PASS if the int4-lane derivative tracks the exact one closely enough to
be a usable descent direction (SGD lives on noise of this scale).
"""
import sys, os, time
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, math
import numpy as np
import gemm_v19 as G

DEV = "cuda"
torch.manual_seed(0)

K, N, M = 64, 128, 256
W = torch.randn(N, K, device=DEV) * 0.1
X = torch.randn(M, K, device=DEV)          # primal activations (analog)
r = 8
A = torch.randn(N, r, device=DEV); B = torch.randn(r, K, device=DEV)
E = (A @ B) / math.sqrt(r)                # EGGROLL rank-8 tangent
P = torch.randn(M, N, device=DEV)         # fixed loss projection: L = mean(P * y)


def exact_dd():
    def loss(w):
        return (P * (X @ w.T)).mean()
    _, d = torch.func.jvp(loss, (W,), (E,))
    return float(d)


def i4_gemm(Wt, Xt, out_scale=None):
    """int4 kernel GEMM: y = Wt @ Xt.T rows, symmetric int4, scaled output."""
    s = 7.0 / Wt.abs().max().clamp(min=1e-8)
    Wq = torch.clamp((Wt / s).round(), -7, 7).to(torch.int32)
    Ap = G.pack(Xt.to(torch.int32).clamp(-7, 7)).contiguous()
    Bt = G.pack_transposed(Wq).contiguous()
    Out8 = torch.empty((M, N), device=DEV, dtype=torch.int8)
    seg = [(0, M, M)]
    te, tm, tn, nt = G.build_tiles(seg, N)
    scale = torch.full((1,), float(s if out_scale is None else out_scale * s), device=DEV)
    G.launch_persistent(FN, Ap, Bt, te, tm, tn, scale, Out8, N, K // 8, nt, 84)
    return Out8.float()


FN = G.compile_src(G.SRC, "dual")

# quantize activations for the primal (deployed format); keep a ternary-ish version
Xq = torch.clamp((X * 4).round() / 4, -7 / 4, 7 / 4)   # 4-lsb grid, int4-compatible

# ---- primal + tangent lanes through the kernel
y = i4_gemm(W, Xq)                                        # primal (deployed int4)
dy = i4_gemm(E, Xq)                                       # tangent lane (STE)
d_i4 = float((P * dy).mean())

# ---- reference: exact fp32 JVP on the dequantized model
d_ref = exact_dd()

rel = abs(d_i4 - d_ref) / (abs(d_ref) + 1e-9)
cos = float(torch.nn.functional.cosine_similarity(
    (P * dy).flatten(), (P * (X @ E.T)).flatten(), dim=0))
print(f"exact fp32 JVP dL/dE : {d_ref:+.4f}")
print(f"int4 dual-lane dL/dE  : {d_i4:+.4f}")
print(f"relative error        : {rel:.3f}")
print(f"lane cosine (vs fp32) : {cos:.4f}")
print(f"VERDICT: {'PASS — usable descent direction' if rel < 0.5 or cos > 0.8 else 'MARGINAL/FAIL'}")

# ---- throughput of the two-lane pattern (batched same-weight traffic)
Ap = G.pack(Xq.to(torch.int32).clamp(-7, 7)).contiguous()
def two_lane():
    a = i4_gemm(W, Xq); b = i4_gemm(E, Xq); return a, b
two_lane()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    two_lane()
torch.cuda.synchronize()
dt = (time.time() - t0) / 20
flop = 4 * M * N * K   # 2 lanes, 2*MNK
print(f"two-lane (primal+tangent) int4: {dt*1e3:.2f} ms  ~{flop/dt/1e12:.1f} TFLOPS-class")
