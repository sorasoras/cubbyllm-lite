"""v26: SWMMAC 2:4-sparse grouped MoE GEMM — the Round-15 recipe, ported to the
v19 persistent skeleton.

Role flip vs v19: SWMMAC's sparse operand is A, so the WEIGHTS sit in the A
slot (2:4 int4, per-row patterns via the decoded idx), and the ACTIVATIONS
sit in the dense B slot. Output D = 16 features x 16 tokens per (warp, ng) —
the epilogue writes Out[token, feature] with the transposed D mapping.

Structural differences vs v19:
- Chunk = 64 k (KCW=8 words): one swmmac_i32_16x16x64_iu4 per (lane, ng)
  replaces two dense 16x16x32 calls — half the MMA issue slots for the same
  logical MACs (mmapeak: 1138.4 expanded TOPS vs 663.5 dense).
- Activations (B slot) are read DIRECTLY from global (one v4i per lane per
  chunk, reused across all 8 ng) — they have zero cross-warp reuse, so no
  LDS staging at all.
- LDS = weights-only double buffer: 128 features x 6 words (4 val + 2 idx)
  = 3 KB per chunk, 6 KB total vs v19's 16.4 KB -> occupancy headroom.

idx encoding (Round 15): the 16 stored nibbles form 8 pairs; pair t owns the
4-k group {4t..4t+3} of the lane's 32-k block; idx field t =
pos(slot 2t) | pos(slot 2t+1) << 2. Packer: slot 2t <- value at first live
position, slot 2t+1 <- second; Wpack (E, K/64, N, 6) = [vals kb0, vals kb1,
idx kb0, idx kb1] per (feature, chunk), fully coalesced staging.

Usage: python gemm_v26.py [T] [K,k,...]
"""
import sys, os, ctypes, subprocess, re
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch
import numpy as np
import wmma_gemm_v2 as W
import gemm_v19 as V19

HIP = V19.HIP
HERE = os.path.dirname(os.path.abspath(__file__))
READELF = r"B:\git\rocm-venv\Lib\site-packages\_rocm_sdk_core\lib\llvm\bin\llvm-readelf.exe"
OBJDUMP = r"B:\git\rocm-venv\Lib\site-packages\_rocm_sdk_core\lib\llvm\bin\llvm-objdump.exe"

SRC = r"""
typedef unsigned int uint32_t;
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
#define NT 8
#define WARPS 16
#define BUFW (128 * 6)          // per 64-k chunk: 128 features x [4 val + 2 idx] words
extern "C" __global__ void moe_v26(const uint32_t* __restrict__ Ap,       // (M, K/8) dense packed
                                    const uint32_t* __restrict__ Wpack,   // (E, K/64, N, 6)
                                    const int* __restrict__ tile_e,
                                    const int* __restrict__ tile_m,
                                    const int* __restrict__ tile_n,
                                    const float* __restrict__ scale,
                                    signed char* __restrict__ Out,
                                    int N, int Kw, int ntiles) {
    extern __shared__ int lds[];   // 2 x BUFW = 6 KB
    int lane = threadIdx.x & 31;
    int warp = threadIdx.y & 15;
    int f16 = lane & 15, kb = lane >> 4;   // A: feature-in-band / k-block; B: token-in-band / k-block
    int nchunk = Kw >> 3;                 // 64-k chunks
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        int e = tile_e[tile];
        int mb = tile_m[tile];
        int n0 = tile_n[tile];
        const uint32_t* Arow = Ap + (long)(mb + warp * 16 + f16) * Kw;
        v8i acc[NT];
        for (int i = 0; i < NT; ++i) acc[i] = {};
        auto loadw = [&](int c, int buf) {
            const uint32_t* src = Wpack + (((long)e * nchunk + c) * N + n0) * 6;
            int* LW = lds + buf * BUFW;
            for (int w = threadIdx.y * 32 + lane; w < BUFW; w += 512)
                LW[w] = (int)src[w];
        };
        loadw(0, 0);
        __syncthreads();
        for (int c = 0; c < nchunk; ++c) {
            int* LW = lds + (c & 1) * BUFW;
            // B fragment: this lane's token, this chunk's 64 k (32 per k-block)
            v4i b = *(const v4i*)(Arow + c * 8 + kb * 4);
            for (int ng = 0; ng < NT; ++ng) {
                int base = (ng * 16 + f16) * 6;
                v2i a = *(const v2i*)(LW + base + kb * 2);
                int idx = LW[base + 4 + kb];
                acc[ng] = __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(
                    true, a, true, b, acc[ng], idx, false);
            }
            if (c + 1 < nchunk) loadw(c + 1, (c + 1) & 1);
            __syncthreads();
        }
        // D[l][j] = (feature = n0 + 16ng + kb*8 + j, token = mb + 16w + f16)
        for (int ng = 0; ng < NT; ++ng)
            for (int j = 0; j < 8; ++j)
                Out[(long)(mb + warp * 16 + f16) * N + n0 + ng * 16 + kb * 8 + j] =
                    (signed char)((float)acc[ng][j] * scale[0]);
        __syncthreads();
    }
}
"""

SHARED = 2 * 128 * 6 * 4


def compile_v26(tag="v26"):
    RTC = W.RTC
    buf = ctypes.create_string_buffer(SRC.encode())
    prog = ctypes.c_void_p()
    assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p),
                                   tag.encode(), 0, None, None) == 0
    opts = (ctypes.c_char_p * 2)(b"--offload-arch=gfx1201", b"-O3")
    rc = RTC.hiprtcCompileProgram(prog, 2, opts)
    if rc != 0:
        lsz = ctypes.c_size_t()
        RTC.hiprtcGetProgramLogSize(prog, ctypes.byref(lsz))
        log = ctypes.create_string_buffer(lsz.value)
        RTC.hiprtcGetProgramLog(prog, log)
        raise RuntimeError(f"hiprtc failed:\n{log.value.decode(errors='replace')}")
    csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
    code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
    open(os.path.join(HERE, f"_{tag}.hsaco"), "wb").write(code.raw[:csz.value])
    mod = ctypes.c_void_p()
    assert HIP.hipModuleLoadData(ctypes.byref(mod), code) == 0
    fn = ctypes.c_void_p()
    assert HIP.hipModuleGetFunction(ctypes.byref(fn), mod, b"moe_v26") == 0
    return fn


def gen_2to4(E, N, K, device="cuda"):
    """Random 2:4 weights: per (row, 4-k group) two live positions p1 != p2.
    Returns (W_expanded (E*N, K) int32, p1, p2, v1, v2)."""
    R, G = E * N, K // 4
    p1 = torch.randint(0, 4, (R, G), device=device)
    p2 = (p1 + 1 + torch.randint(0, 3, (R, G), device=device)) % 4
    v1 = torch.randint(-8, 8, (R, G), device=device)
    v2 = torch.randint(-8, 8, (R, G), device=device)
    W = torch.zeros(R, K, device=device, dtype=torch.int32)
    rows = torch.arange(R, device=device).unsqueeze(1).expand(R, G)
    grps = torch.arange(G, device=device).unsqueeze(0).expand(R, G)
    W[rows, grps * 4 + p1] = v1.to(torch.int32)
    W[rows, grps * 4 + p2] = v2.to(torch.int32)
    return W, p1, p2, v1, v2


def pack_2to4(p1, p2, v1, v2, E, N, K):
    """Round-15 recipe -> Wpack (E, K/64, N, 6) uint32-bit-patterns (int32 tensor).
    Per (feature, 64-k chunk): [vals kb0 (2 words), vals kb1 (2), idx kb0, idx kb1].
    Slot 2t <- value at first live position (p1), slot 2t+1 <- second (p2);
    idx field t = p1 | p2 << 2. t = group index within its 32-k block (0..7):
    slots (2t, 2t+1) are nibbles of word t//4; the 4 groups of a word OR together."""
    R, G = E * N, K // 4
    nblk, nchunk = K // 32, K // 64
    t = torch.arange(G, device=p1.device).unsqueeze(0)      # global group index (1, G)
    tl = t % 8                                               # group within its 32-k block
    blk = t // 8                                             # 32-k block index (1, G)
    word_val = ((v1.long() & 0xF) << (4 * ((2 * tl) % 8))) | \
               ((v2.long() & 0xF) << (4 * ((2 * tl + 1) % 8)))   # (R, G)
    idx_val = ((p1.long() & 3) << (4 * tl)) | ((p2.long() & 3) << (4 * tl + 2))
    Wwflat = torch.zeros(R, nblk * 2, dtype=torch.int64, device=p1.device)
    Wwflat.scatter_add_(1, (blk * 2 + tl // 4).long().expand(R, G), word_val)
    idxb = torch.zeros(R, nblk, dtype=torch.int64, device=p1.device)
    idxb.scatter_add_(1, blk.long().expand(R, G), idx_val)
    Ww = Wwflat.view(R, nblk, 2)
    per_chunk = torch.zeros(R, nchunk, 6, dtype=torch.int64, device=p1.device)
    per_chunk[:, :, 0:2] = Ww[:, 0::2, :]
    per_chunk[:, :, 2:4] = Ww[:, 1::2, :]
    per_chunk[:, :, 4] = idxb[:, 0::2]
    per_chunk[:, :, 5] = idxb[:, 1::2]
    out = per_chunk.reshape(E, N, nchunk, 6).permute(0, 2, 1, 3).contiguous()
    return out.to(torch.int32)


def make_problem(T, K, E=8, N=2048):
    Kw = K // 8
    assign = torch.randint(0, E, (T,))
    counts = torch.bincount(assign, minlength=E)
    segs = ((counts + 255) // 256) * 256
    M_pad = int(segs.sum())
    A_tok = torch.zeros(M_pad, K, device="cuda", dtype=torch.int32)
    seg_base = []
    pos = 0
    for eid in range(E):
        n_e = int(counts[eid])
        A_tok[pos:pos + n_e] = torch.randint(-8, 8, (n_e, K), device="cuda", dtype=torch.int32)
        seg_base.append((pos, int(segs[eid]), n_e)); pos += int(segs[eid])
    W4, p1, p2, v1, v2 = gen_2to4(E, N, K)
    Wpack = pack_2to4(p1, p2, v1, v2, E, N, K)
    Wd = torch.randint(-8, 8, (E * N, K), device="cuda", dtype=torch.int32)  # dense (for v19)
    Ap = V19.pack(A_tok).contiguous()
    Bt = torch.cat([V19.pack_transposed(Wd[eid * N:(eid + 1) * N]) for eid in range(E)],
                  dim=0).contiguous()
    Out26 = torch.empty((M_pad, N), device="cuda", dtype=torch.int8)
    Out19 = torch.empty((M_pad, N), device="cuda", dtype=torch.int8)
    tile_e, tile_m, tile_n, ntiles = V19.build_tiles(seg_base, N)
    return dict(Ap=Ap, Wpack=Wpack, Bt=Bt, Out26=Out26, Out19=Out19, seg_base=seg_base,
                A_tok=A_tok, W4=W4, Wd=Wd, M_pad=M_pad, N=N, Kw=Kw, ntiles=ntiles,
                tile_e=tile_e, tile_m=tile_m, tile_n=tile_n)


def launch26(pb, fn, scale, P):
    V19.launch_persistent(fn, pb["Ap"], pb["Wpack"], pb["tile_e"], pb["tile_m"], pb["tile_n"],
                          scale, pb["Out26"], pb["N"], pb["Kw"], pb["ntiles"], P)


def launch19(pb, fn, scale, P):
    V19.launch_persistent(fn, pb["Ap"], pb["Bt"], pb["tile_e"], pb["tile_m"], pb["tile_n"],
                          scale, pb["Out19"], pb["N"], pb["Kw"], pb["ntiles"], P)


def _copy_out(t):
    HIP.hipDeviceSynchronize()
    host = np.empty(t.numel(), dtype=np.int8)
    HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(t.data_ptr()), t.numel(), 2)
    return host.reshape(t.shape).astype(np.float32)


def verify(pb, fn26, fn19, s):
    scale = torch.full((1,), s, device="cuda")
    launch26(pb, fn26, scale, 84)
    launch19(pb, fn19, scale, 84)
    O26 = _copy_out(pb["Out26"])
    O19 = _copy_out(pb["Out19"])
    A_np = pb["A_tok"].cpu().numpy().astype(np.float32)
    W4 = pb["W4"].cpu().numpy().astype(np.float32)
    Wd = pb["Wd"].cpu().numpy().astype(np.float32)
    e26 = e19 = 0.0
    for eid, (base, rows, n_e) in enumerate(pb["seg_base"]):
        if n_e == 0:
            continue
        A = A_np[base:base + n_e]
        r26 = A @ W4[eid * pb["N"]:(eid + 1) * pb["N"]].T
        r19 = A @ Wd[eid * pb["N"]:(eid + 1) * pb["N"]].T
        e26 = max(e26, np.abs(O26[base:base + n_e] - r26 * s).max())
        e19 = max(e19, np.abs(O19[base:base + n_e] - r19 * s).max())
    return e26, e19


def meta_txt(path):
    r = subprocess.run([READELF, "-n", path], capture_output=True, text=True, timeout=120)
    out = {}
    for k in ["vgpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count"]:
        m = re.search(rf"\.{re.escape(k)}:\s+(\d+)", r.stdout)
        if m:
            out[k] = int(m.group(1))
    return out


def isa_ops(path):
    r = subprocess.run([OBJDUMP, "--disassemble-symbols=moe_v26", "--arch-name=amdgcn",
                        "--mcpu=gfx1201", path], capture_output=True, text=True, timeout=600)
    hist = {}
    for m in re.finditer(r"^\t([a-z][a-z0-9_.]*)\s", r.stdout, re.M):
        hist[m.group(1)] = hist.get(m.group(1), 0) + 1
    return {k: v for k, v in hist.items() if "swmmac" in k or "ds_" in k or "global_" in k}


if __name__ == "__main__":
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
    Ks = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [768, 4096]
    S = 1.0 / 128.0
    torch.manual_seed(0)
    print(f"v26 SWMMAC 2:4 sparse MoE: T={T} scale={S}", flush=True)
    fn26 = compile_v26()
    fn19 = V19.compile_src(V19.SRC, "v26_ref19")
    km = meta_txt(os.path.join(HERE, "_v26.hsaco"))
    print(f"v26 metadata: {km} (v19: vgpr=113 sgpr=46 spills=0, LDS 16.4KB; v26 LDS {SHARED}B)",
          flush=True)
    print(f"v26 key ops: {isa_ops(os.path.join(HERE, '_v26.hsaco'))}", flush=True)

    for K in Ks:
        assert K % 64 == 0, "K must be a multiple of 64 (64-k chunks)"
        pb = make_problem(T, K)
        gflop = 2 * T * K * pb["N"] / 1e9
        e26, e19 = verify(pb, fn26, fn19, S)
        print(f"\n=== K={K} ntiles={pb['ntiles']} gflop={gflop:.1f} ===")
        print(f"correctness: v26 err={e26:.1f} | v19 err={e19:.1f} "
              f"({'both PASS' if max(e26, e19) <= 1 else 'FAIL'})", flush=True)
        scale = torch.full((1,), S, device="cuda")
        for P in (84, 112, 168, 224):
            t26 = min(V19.bench(lambda P=P: launch26(pb, fn26, scale, P), 30) for _ in range(6))
            t19 = min(V19.bench(lambda P=P: launch19(pb, fn19, scale, P), 30) for _ in range(6))
            print(f"P={P:3d}: v26-SWMMAC {gflop / t26:6.1f} | v19-dense {gflop / t19:6.1f} "
                  f"| speedup {t19 / t26:.3f}x", flush=True)
