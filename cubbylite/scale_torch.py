"""215M-scale EGGROLL spike-MoE LM on PyTorch -- ROCm (AMD HIP) / DirectML / CPU.

GPU-resident, op-count-optimized design:
- Fused weight layouts: one QKV matrix per block, one all-expert E1 matrix
  per block -> 1 GEMM instead of 3 + 8.
- ES factors sampled PRE-STACKED on GPU: U (m, pop*R), V (pop*R, n) per
  matrix per generation; antithetic signs folded into the fitness weights.
  Update = one weighted GEMM per matrix. No per-pair stacking, no slicing.
- Perturbation for each eval: d_i = u_i @ v_i as a view-GEMM (no copy),
  applied in-place on GPU. Weights never leave the GPU.

Backends: --backend rocm (torch HIP, device 'cuda'), dml (torch-directml), cpu.
Same validated math as cubbylite.model.SpikeMoELM (frozen hash router,
ternary spikes, top-1 of N routed + 1 shared expert, retrieval head).
"""
import argparse
import time

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--backend", choices=["rocm", "dml", "cpu"], default="rocm")
p.add_argument("--d", type=int, default=768)
p.add_argument("--hid", type=int, default=2048)
p.add_argument("--nexp", type=int, default=8)
p.add_argument("--blocks", type=int, default=7)
p.add_argument("--ctx", type=int, default=128)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--pop", type=int, default=64)
p.add_argument("--sigma", type=float, default=0.05)
p.add_argument("--alpha", type=float, default=0.01)
p.add_argument("--rank", type=int, default=8)
p.add_argument("--gens", type=int, default=80)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", type=str, default="rocm")
args = p.parse_args()

if args.backend == "rocm":
    dev = torch.device("cuda")
elif args.backend == "dml":
    import torch_directml
    dev = torch_directml.device()
else:
    dev = torch.device("cpu")

torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)

# ---------------- data ----------------
train_text = open(r"B:\git\wikitext2_train.txt", encoding="utf-8").read()
test_text = open(r"B:\git\wikitext2_test.txt", encoding="utf-8").read()
train_ids = np.frombuffer(train_text.encode("utf-8"), dtype=np.uint8).astype(np.int64)
test_ids = np.frombuffer(test_text.encode("utf-8"), dtype=np.uint8).astype(np.int64)
V, CTX = 256, args.ctx

def batch(ids, bs, train=True):
    lo = 0 if train else int(len(ids) * 0.9)
    hi = len(ids) - CTX - 1
    ix = rng.integers(lo, hi, size=bs)
    x = np.stack([ids[i:i + CTX] for i in ix])
    y = np.stack([ids[i + 1:i + CTX + 1] for i in ix])
    top = (x.astype(np.int64) * 2654435761) % NEXP      # frozen hash, numpy-side
    masks = (top[..., None] == np.arange(NEXP)[None, None, :]).astype(np.float32)
    return (torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev),
            torch.from_numpy(masks).to(dev))

# ---------------- model ----------------
D, HID, NEXP, NB = args.d, args.hid, args.nexp, args.blocks
def init_w(shape, scale):
    return (torch.randn(*shape) * scale).to(dev)

W = {"emb": init_w((V, D), 0.05)}
for b in range(NB):
    P = f"b{b}."
    W[P + "qkv"] = init_w((3 * D, D), 1 / np.sqrt(D))      # fused Q,K,V
    W[P + "wo"] = torch.zeros((D, D), device=dev)
    W[P + "E1"] = init_w((NEXP * HID, D), 1 / np.sqrt(D))  # fused all-expert E1
    for e in range(NEXP):
        W[f"{P}e{e}_2"] = torch.zeros((D, HID), device=dev)
    W[P + "sh1"] = init_w((HID, D), 1 / np.sqrt(D))
    W[P + "sh2"] = torch.zeros((D, HID), device=dev)
KEYS = list(W.keys())
n_params = sum(w.numel() for w in W.values())
print(f"model {n_params/1e6:.1f}M params on {dev} (d={D}, hid={HID}, {NEXP}+1 experts, {NB} blocks)", flush=True)

CAUSAL = torch.triu(torch.ones((CTX, CTX), dtype=torch.bool, device=dev), 1)

def spike(x):
    thr = 0.5 * x.abs().mean()
    return (x > thr).float() - (x < -thr).float()

def forward(idx, masks):
    x = W["emb"][idx]
    bal = []
    for b in range(NB):
        P = f"b{b}."
        qkv = x @ W[P + "qkv"].T
        q, k_pre, v_pre = qkv[..., :D], qkv[..., D:2 * D], qkv[..., 2 * D:]
        k, v = spike(k_pre), spike(v_pre)
        att = q @ k.transpose(-1, -2) / np.sqrt(D)
        att = torch.where(CAUSAL, torch.tensor(-1e9, device=dev), att)
        x = x + (torch.softmax(att, -1) @ v) @ W[P + "wo"].T

        h_all = spike(x @ W[P + "E1"].T)                 # (B,T,NEXP*HID)
        out = torch.zeros_like(x)
        for e in range(NEXP):
            h_e = h_all[..., e * HID:(e + 1) * HID]
            out = out + (h_e @ W[f"{P}e{e}_2"].T) * masks[..., e:e + 1]
        out = out + spike(x @ W[P + "sh1"].T) @ W[P + "sh2"].T
        x = x + out
        bal.append(masks.mean(dim=(0, 1)).tolist())
    h = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    e = W["emb"] / (W["emb"].norm(dim=-1, keepdim=True) + 1e-8)
    return h @ e.T * 8.0, bal

def ce(idx, tgt, masks):
    logits, _ = forward(idx, masks)
    logp = torch.log_softmax(logits, -1)
    return -logp.gather(-1, tgt.unsqueeze(-1)).mean()

x_ev, y_ev, m_ev = batch(test_ids, 16, train=False)
t0 = time.time()
print(f"uniform CE = {np.log(V):.3f}", flush=True)
print(f"gen    0  ce {ce(x_ev, y_ev, m_ev).item():.3f}  (init {time.time()-t0:.0f}s)", flush=True)

# ---------------- EGGROLL: pre-stacked factors on GPU ----------------
base = {k: W[k].clone() for k in KEYS}
R = min(args.rank, args.d, args.hid)

# per generation: sample U (m, pop*R), V (pop*R, n) directly per matrix
def sample_factors():
    fs = {}
    for k in KEYS:
        s = tuple(W[k].shape)
        if len(s) == 1:
            fs[k] = torch.from_numpy(rng.standard_normal((s[0], args.pop)).astype(np.float32)).to(dev)
        else:
            u = torch.from_numpy(rng.standard_normal((s[0], R * args.pop)).astype(np.float32)).to(dev)
            v = torch.from_numpy(rng.standard_normal((R * args.pop, s[1])).astype(np.float32)).to(dev)
            fs[k] = (u, v)
    return fs

def pair_views(fs, pp):
    """Views of pair pp's (u, v) -- zero-copy slices of the stacked factors."""
    out = {}
    for k in KEYS:
        f = fs[k]
        if not isinstance(f, tuple):
            out[k] = f[:, pp:pp + 1]
            continue
        u, v = f
        out[k] = (u[:, pp * R:(pp + 1) * R], v[pp * R:(pp + 1) * R, :])
    return out

for gen in range(1, args.gens + 1):
    t1 = time.time()
    x_fit, y_fit, m_fit = batch(train_ids, args.batch)
    fs = sample_factors()
    half = args.pop // 2
    fit = []
    for pp in range(half):
        pv = pair_views(fs, pp)
        d = {}
        for k in KEYS:
            f = pv[k]
            d[k] = f if not isinstance(f, tuple) else f[0] @ f[1] / np.sqrt(R)
        for k in KEYS:
            W[k].copy_(base[k]).add_(d[k], alpha=args.sigma)
        fit.append(-ce(x_fit, y_fit, m_fit).item())
        for k in KEYS:
            W[k].copy_(base[k]).add_(d[k], alpha=-args.sigma)
        fit.append(-ce(x_fit, y_fit, m_fit).item())
        del d
    f = np.array(fit, dtype=np.float64)
    f = (f - f.mean()) / (f.std() + 1e-8)
    # antithetic signs interleaved: pair pp -> weights (+f[pp], -f[pp])
    fw = np.empty(args.pop, dtype=np.float32)
    fw[0::2], fw[1::2] = f[:half], -f[:half]

    alpha_t = args.alpha * 150.0 / (gen + 150.0)
    for k in KEYS:
        fk = fs[k]
        if not isinstance(fk, tuple):
            a = fk @ torch.from_numpy(fw).to(dev)
            a = (a - a.mean()) / (a.std() + 1e-8)
            base[k] += alpha_t * a
            continue
        u, v = fk
        fw_t = torch.from_numpy(np.repeat(fw, R).astype(np.float32)).to(dev)
        acc = (u * fw_t) @ v / np.sqrt(R)
        a = (acc - acc.mean()) / (acc.std() + 1e-8)
        base[k] += alpha_t * a
    for k in KEYS:
        W[k].copy_(base[k])

    if gen % 10 == 0 or gen == args.gens:
        c = ce(x_ev, y_ev, m_ev).item()
        _, bal = forward(x_ev, m_ev)
        b = np.array(bal).mean(axis=0)
        print(f"gen {gen:4d}  ce {c:.3f}  ppl {np.exp(c):7.1f}  "
              f"bal {b.min():.2f}/{b.max():.2f}  ({time.time()-t0:.0f}s, "
              f"{time.time()-t1:.1f}s/gen)", flush=True)

print(f"DONE tag={args.tag} params={n_params/1e6:.1f}M gens={args.gens} total={time.time()-t0:.0f}s", flush=True)
