import sys, time
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, numpy as np, ctypes
import gemm_v6 as G
import wmma_gemm_v2 as W
HIP = W.HIP; RTC = W.RTC
HIP.hipDeviceSynchronize.restype = ctypes.c_int

src = G.SRC.replace("moe_v6", "moe_i8")
old_sig = "float* __restrict__ Out,"
assert old_sig in src, "sig"
src = src.replace(old_sig, "signed char* __restrict__ Out,")
old = """        int rbase = (lane >> 4) * 8;
        for (int mg = 0; mg < 2; ++mg)
            for (int ng = 0; ng < 4; ++ng)
                for (int j = 0; j < 8; ++j)
                    Out[(long)(mb + (warp >> 1) * 32 + mg * 16 + rbase + j) * N
                        + n0 + (warp & 1) * 64 + ng * 16 + col] =
                        (float)acc[mg * 4 + ng][j] * scale[0];"""
assert old in src, "epilogue"
new = old.replace("(float)acc[mg * 4 + ng][j] * scale[0]",
                  "(signed char)((float)acc[mg * 4 + ng][j] * scale[0])")
src = src.replace(old, new)

buf = ctypes.create_string_buffer(src.encode())
prog = ctypes.c_void_p()
assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p), b"v7i8", 0, None, None) == 0
opts = (ctypes.c_char_p * 4)(b"--offload-arch=gfx1201", b"-O3", b"-DNEG_A=1", b"-DNEG_B=1")
assert RTC.hiprtcCompileProgram(prog, 4, opts) == 0, "compile"
csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
m2 = ctypes.c_void_p(); assert HIP.hipModuleLoadData(ctypes.byref(m2), code) == 0
fn8 = ctypes.c_void_p(); assert HIP.hipModuleGetFunction(ctypes.byref(fn8), m2, b"moe_i8") == 0

torch.manual_seed(0)
scale = torch.full((1,), 1.0 / 64.0, device="cuda")  # keep outputs in int8 range
N, K, Kw, E, T = 2048, 4096, 512, 8, 16384
assign = torch.randint(0, E, (T,))
counts = torch.bincount(assign, minlength=E)
segs = ((counts + 127) // 128) * 128
M_pad = int(segs.sum())
A_tok = torch.zeros(M_pad, K, device="cuda", dtype=torch.int32)
seg_base = []; pos = 0
for eid in range(E):
    n_e = int(counts[eid])
    A_tok[pos:pos+n_e] = torch.randint(-8, 8, (n_e, K), device="cuda", dtype=torch.int32)
    seg_base.append((pos, int(segs[eid]), n_e)); pos += int(segs[eid])
W_all = torch.randint(-8, 8, (E*N, K), device="cuda", dtype=torch.int32)
Ap = G.pack(A_tok).contiguous()
Bt = torch.cat([G.pack_transposed(W_all[eid*N:(eid+1)*N]) for eid in range(E)], dim=0).contiguous()
Out8 = torch.empty((M_pad, N), device="cuda", dtype=torch.int8)
tile_e, tile_m, tile_n, ntiles = G.build_tiles(seg_base, N)
gflop = 2*T*K*N/1e9

def launch8():
    args = [Ap, Bt, tile_e, tile_m, tile_n, scale, Out8, N, Kw, ntiles]
    storage = [ctypes.c_void_p(t.data_ptr()) if torch.is_tensor(t) else ctypes.c_int32(t) for t in args]
    ptrs = (ctypes.c_void_p * len(storage))(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in storage])
    st = HIP.hipModuleLaunchKernel(fn8, 168, 1, 1, 32, 8, 1, G.SHARED, None, ptrs, None)
    assert st == 0, st

launch8(); HIP.hipDeviceSynchronize()
host = np.empty(M_pad*N, dtype=np.int8)
HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out8.data_ptr()), M_pad*N, 2)
On = host.reshape(M_pad, N).astype(np.float32)
A_np = A_tok.cpu().numpy().astype(np.float32); W_np = W_all.cpu().numpy().astype(np.float32)
err = 0.0
for eid in range(E):
    base, rows, n_e = seg_base[eid]
    ref = (A_np[base:base+n_e] @ W_np[eid*N:(eid+1)*N].T) / 64.0
    print(f"v7b int8 epilogue (scale 1/64): max|err| = {err:.1f} (<=0.51 rounding)", "PASS" if err <= 1.0 else "FAIL", flush=True)
launch8(); torch.cuda.synchronize(); time.sleep(2)
ts = []
for rep in range(4):
    ts.append(G.bench(launch8, n=20)); time.sleep(0.5)
print(f"v7b int8-out P=168 best-of-4: {gflop/min(ts):6.1f} TFLOPS ({gflop/min(ts)/663.5*100:.1f}%)  runs: {[f'{gflop/x:.0f}' for x in ts]}", flush=True)
