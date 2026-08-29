import sys, re
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, ctypes
import numpy as np
import wmma_gemm_v2 as W
HIP = W.HIP
HIP.hipDeviceSynchronize.restype = ctypes.c_int
HIP.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

s_all = open(r"B:\git\cubbyllm-lite\kernels\wmma_gemm_v2.py", encoding="utf-8").read()
body = re.search(r'extern "C" __global__ void moe_i4.*?\n}\n', s_all, re.S).group(0)
b = re.sub(r"int e = blk_expert\[blockIdx\.y\];",
           "const uint32_t* Wpe = Wp + (unsigned long long)blk_expert[blockIdx.y] * (unsigned long long)N * (unsigned long long)Kw;\n"
           "    const uint32_t* Ape = Ap + (unsigned long long)blk_mbase[blockIdx.y] * (unsigned long long)Kw;\n"
           "    float* Oe = Out + (unsigned long long)blk_mbase[blockIdx.y] * (unsigned long long)N;\n"
           "    int e = 0;", body)
b = re.sub(r"int mb = blk_mbase\[blockIdx\.y\];", "int mb = 0;", b)
b = re.sub(r"Ap\[\(mb \+ w / 2\) \* Kw \+ kw \+ w % 2\]", "Ape[(w / 2) * Kw + kw + w % 2]", b)
b = re.sub(r"Wp\[\(\(long\)e \* N \+ n0 \+ i \* 16 \+ r\) \* Kw \+ kw \+ q\]",
           "Wpe[(n0 + i * 16 + r) * Kw + kw + q]", b)
b = re.sub(r"Out\[\(mb \+ rbase \+ j\) \* N \+ n0 \+ i \* 16 \+ col\]",
           "Oe[(rbase + j) * N + n0 + i * 16 + col]", b)
b = b.replace("Wp[((long)e * N + n0 + i * 16 + r)", "Wp[(n0 + i * 16 + r)]")
b = b.replace("Ap[(mb + w / 2) * Kw", "Ap[(w / 2) * Kw")
b = b.replace("Out[(mb + rbase + j) * N", "Out[(rbase + j) * N")
assert "blk_expert" not in b.split("extern")[0] or True
src_off = W.SRC.replace(body, b)

mod = W.compile_src(src_off, "moe_off3")
fn = W.get_fn(mod, "moe_i4")
scale = torch.ones(1, device="cuda")
E, N, K = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
Kw = K // 8
T = 256
assign = torch.randint(0, E, (T,))
order = torch.argsort(assign, stable=True)
counts = torch.bincount(assign, minlength=E)
segs = (counts + 15) // 16
nblocks = int(segs.sum())
blk_expert = torch.repeat_interleave(torch.arange(E), segs).int().contiguous()
blk_mbase = (torch.arange(nblocks, dtype=torch.int32) * 16).contiguous()
M_pad = nblocks * 16
A_tok = torch.zeros(M_pad, K, device="cuda", dtype=torch.int32)
pos = 0
for eid in range(E):
    idx = order[assign == eid]; n_e = idx.numel()
    A_tok[pos:pos+n_e] = torch.randint(-8, 8, (n_e, K), device="cuda", dtype=torch.int32)
    pos += int(segs[eid]) * 16
Ap = W.pack(A_tok).contiguous()
Wp = W.pack(torch.randint(-8, 8, (E * N, K), device="cuda", dtype=torch.int32)).contiguous()
Out_m = torch.empty((M_pad, N), device="cuda", dtype=torch.float32)
host_m = np.empty(M_pad * N, dtype=np.float32)

W.launch1(fn, (N // 64, nblocks, 1), [Ap, Wp, blk_expert, blk_mbase, scale, Out_m, N, Kw])
s1 = HIP.hipDeviceSynchronize()
s2 = HIP.hipMemcpy(host_m.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out_m.data_ptr()),
                   M_pad * N * 4, 2)
print(f"pointer-offset: sync={s1} memcpy={s2}", flush=True)
if s2 == 0:
    Out_np = host_m.reshape(M_pad, N)
    A_np = A_tok.cpu().numpy().astype(np.float32)
    Wq = Wp.cpu().numpy().reshape(E * N, Kw)
    Wdeq = np.zeros((E * N, K), dtype=np.float32)
    for i in range(8):
        Wdeq[:, i::8] = ((Wq >> (4 * i)) & 0xF).astype(np.float32)
    Wdeq = np.where(Wdeq > 7, Wdeq - 16, Wdeq)
    err = 0.0
    for bi in range(nblocks):
        eid = int(blk_expert.cpu().numpy()[bi]); mb = int(blk_mbase.cpu().numpy()[bi])
        refb = A_np[mb:mb+16] @ Wdeq[eid*N:(eid+1)*N].T
        err = max(err, np.abs(Out_np[mb:mb+16] - refb).max())
    print(f"grouped moe (pointer-offset): max|err| = {err:.1f}  {'PASS' if err == 0 else 'FAIL'}", flush=True)
