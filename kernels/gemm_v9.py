"""v9: 128x256 blocks with acc recycling: 2m x 16n per warp in two 8-wide
batches sharing A frags -> 160 B/WMMA LDS at 64-reg acc (occupancy kept).

Block: 256 threads (8 warps), 128x256 output tile. Warp w -> 32x128
sub-tile (2 m-groups x 8 n-groups): A frags reused 8x, B frags 2x ->
160 B/WMMA LDS; block AI 341 FLOP/B vs 262 at 128x128. 64-k chunks, 2 WMMA calls per
chunk (verified scheme: call s feeds lane-group kt words (4kt+2s, +1) —
covers all 64 k exactly once, 8192 MACs/call, 663.5 TOPS instr peak).
Double-buffered LDS (A stride 9, B stride 129 — conflict-reduced).

PERSISTENT: one launch for the whole grouped MoE; grid = P blocks, each
loops over (expert, m, n) tiles via host-built tile tables. Removes the
8-launch Python overhead that would starve the GPU at 400+ TFLOPS.
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

SRC = r"""
typedef unsigned int uint32_t;
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
#define NT 8
#define KCW 8
#define AST 10
#define BST 512          // B stored word-PAIR-interleaved: pair p, col c at [p*256 + c*2]
#define BUFSZ (128 * AST + 4 * BST)
// Persistent grouped MoE: tiles = (expert, m-base) pairs x N/128 col tiles.
// 64-k chunks, 2 WMMA calls: call s feeds lane-group kt words (4kt+2s, +1)
// — pairs (2kt+s). A and B fragments loaded as single 64-bit LDS accesses.
extern "C" __global__ void moe_v9(const uint32_t* __restrict__ Ap,
                                  const uint32_t* __restrict__ Bt,
                                  const int* __restrict__ tile_e,
                                  const int* __restrict__ tile_m,
                                  const int* __restrict__ tile_n,
                                  const float* __restrict__ scale,
                                  float* __restrict__ Out,
                                  int N, int Kw, int ntiles) {
    extern __shared__ int lds[];   // 2 bufs x (128*AST + 4*BST) ints
    int lane = threadIdx.x & 31;
    int warp = threadIdx.y & 7;
    int col = lane & 15, kt = lane >> 4;
    int row_local = warp * 16 + col;
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        int e = tile_e[tile];
        int mb = tile_m[tile];
        int n0 = tile_n[tile];
        const uint32_t* Aptr = Ap + (long)mb * Kw;
        const uint32_t* Bptr = Bt + (long)e * Kw * N + n0;
        v8i acc[16];
        for (int i = 0; i < 16; ++i) acc[i] = {};
        auto load = [&](int kw, int buf) {
            int* LA = lds + buf * BUFSZ;
            int* LB = LA + 128 * AST;
            for (int w = threadIdx.y * 32 + lane; w < 128 * 4; w += 256) {
                int r = w >> 2, qq = w & 3;
                *(v2i*)(LA + r * AST + qq * 2) = *(const v2i*)(Aptr + (long)r * Kw + kw + qq * 2);
            }
            for (int w = threadIdx.y * 32 + lane; w < 4 * 256; w += 256) {
                int p = w >> 8, nl = w & 255;
                v2i val;
                val.x = Bptr[(long)(kw + 2 * p) * N + nl];
                val.y = Bptr[(long)(kw + 2 * p + 1) * N + nl];
                *(v2i*)(LB + p * BST + nl * 2) = val;
            }
        };
        load(0, 0);
        __syncthreads();
        for (int kw = 0; kw < Kw; kw += KCW) {
            int* LA = lds + ((kw / KCW) & 1) * BUFSZ;
            int* LB = LA + 128 * AST;
            int mrow0 = (warp >> 1) * 32 + col;
            int nb = (warp & 1) * 256 + col * 2;
            for (int s = 0; s < KCW / 4; ++s) {
                v2i a0 = *(const v2i*)(LA + mrow0 * AST + 4 * s + 2 * kt);
                v2i a1 = *(const v2i*)(LA + (mrow0 + 16) * AST + 4 * s + 2 * kt);
                int pb = (2 * s + kt) * BST + nb;
                for (int ng = 0; ng < 8; ++ng) {
                    v2i b = *(const v2i*)(LB + pb + ng * 32);
                    acc[ng] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, a0, 1, b, acc[ng], 0);
                }
            }
            // batch 0 (ng 0..7) done -> write out, reuse acc for batch 1 (ng 8..15)
            int rbase0 = (lane >> 4) * 8;
            for (int mg = 0; mg < 2; ++mg)
                for (int ng = 0; ng < 8; ++ng)
                    for (int j = 0; j < 8; ++j)
                        Out[(long)(mb + (warp >> 1) * 32 + mg * 16 + rbase0 + j) * N
                            + n0 + (warp & 1) * 128 + ng * 16 + col] =
                            (float)acc[mg * 8 + ng][j] * scale[0];
            for (int i = 0; i < 16; ++i) acc[i] = {};
            for (int s = 0; s < KCW / 4; ++s) {
                v2i a0 = *(const v2i*)(LA + mrow0 * AST + 4 * s + 2 * kt);
                v2i a1 = *(const v2i*)(LA + (mrow0 + 16) * AST + 4 * s + 2 * kt);
                int pb = (2 * s + kt) * BST + nb;
                for (int ng = 0; ng < 8; ++ng) {
                    v2i b = *(const v2i*)(LB + pb + 256 + ng * 32);
                    acc[ng] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, a0, 1, b, acc[ng], 0);
                }
            }
            // prefetch next chunk into the OTHER buffer while acc math drains;
            // single end-of-chunk sync orders both the buffer swap and the load
            if (kw + KCW < Kw) load(kw + KCW, ((kw / KCW) + 1) & 1);
            __syncthreads();
        }
        int rbase = (lane >> 4) * 8;
        for (int mg = 0; mg < 2; ++mg)
            for (int ng = 0; ng < 8; ++ng)
                for (int j = 0; j < 8; ++j)
                    Out[(long)(mb + (warp >> 1) * 32 + mg * 16 + rbase + j) * N
                        + n0 + (warp & 1) * 128 + 128 + ng * 16 + col] =
                        (float)acc[mg * 8 + ng][j] * scale[0];
        __syncthreads();
    }
}
"""

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
    assert HIP.hipModuleGetFunction(ctypes.byref(fn), m2, b"moe_v9") == 0
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

SHARED = 2 * (128 * 10 + 4 * 512) * 4

def n_cus():
    try:
        v = ctypes.c_int32(0)
        # hipDeviceAttributeMultiprocessorCount = 22
        if HIP.hipDeviceGetAttribute(ctypes.byref(v), 22, 0) == 0 and v.value > 0:
            return v.value
    except Exception:
        pass
    return 64  # attribute unreliable in this context; assume >= 64-block saturation

def launch_persistent(fn, Ap, Bt, tile_e, tile_m, tile_n, scale, Out, N, Kw, ntiles, P):
    args = [Ap, Bt, tile_e, tile_m, tile_n, scale, Out, N, Kw, ntiles]
    storage = [ctypes.c_void_p(t.data_ptr()) if torch.is_tensor(t) else ctypes.c_int32(t) for t in args]
    ptrs = (ctypes.c_void_p * len(storage))(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in storage])
    st = HIP.hipModuleLaunchKernel(fn, P, 1, 1, 32, 8, 1, SHARED, None, ptrs, None)
    assert st == 0, f"launch {st}"

def build_tiles(seg_base, N):
    """tile list ordered (expert, m, n); tile_m is the GLOBAL M base in Ap."""
    te, tm, tn = [], [], []
    nt = N // 256
    for eid, (base, rows, n_e) in enumerate(seg_base):
        if rows <= 0:
            continue
        for mt in range(rows // 128):
            for nc in range(nt):
                te.append(eid); tm.append(base + mt * 128); tn.append(nc * 256)
    return (torch.tensor(te, device="cuda", dtype=torch.int32),
            torch.tensor(tm, device="cuda", dtype=torch.int32),
            torch.tensor(tn, device="cuda", dtype=torch.int32), len(te))

def bench(fnc, n=50):
    fnc(); torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(n): fnc()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n

if __name__ == "__main__":
    torch.manual_seed(0)
    fn = compile_src(SRC, "v6")
    scale = torch.ones(1, device="cuda")
    N = 2048
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 768
    Kw = K // 8
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
    E = 8
    P = 2 * n_cus()
    print(f"v6 persistent: K={K} T={T} P={P} blocks ({n_cus()} CUs), shared={SHARED} B", flush=True)

    assign = torch.randint(0, E, (T,))
    counts = torch.bincount(assign, minlength=E)
    segs128 = ((counts + 127) // 128) * 128
    M_pad = int(segs128.sum())
    A_tok = torch.zeros(M_pad, K, device="cuda", dtype=torch.int32)
    seg_base = []
    pos = 0
    for eid in range(E):
        n_e = int(counts[eid])
        A_tok[pos:pos + n_e] = torch.randint(-8, 8, (n_e, K), device="cuda", dtype=torch.int32)
        seg_base.append((pos, int(segs128[eid]), n_e)); pos += int(segs128[eid])
    W_all = torch.randint(-8, 8, (E * N, K), device="cuda", dtype=torch.int32)
    Ap = pack(A_tok).contiguous()
    # per-expert (Kw, N) blocks concatenated along the word axis -> (E*Kw, N)
    Bt = torch.cat([pack_transposed(W_all[eid * N:(eid + 1) * N]) for eid in range(E)], dim=0).contiguous()
    Out = torch.empty((M_pad, N), device="cuda", dtype=torch.float32)
    tile_e, tile_m, tile_n, ntiles = build_tiles(seg_base, N)

    def moe():
        launch_persistent(fn, Ap, Bt, tile_e, tile_m, tile_n, scale, Out, N, Kw, ntiles, P)

    moe()
    HIP.hipDeviceSynchronize()
    host = np.empty(M_pad * N, dtype=np.float32)
    HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(Out.data_ptr()), M_pad * N * 4, 2)
    Out_np = host.reshape(M_pad, N)
    A_np = A_tok.cpu().numpy().astype(np.float32)
    W_np = W_all.cpu().numpy().astype(np.float32)
    err = 0.0
    for eid in range(E):
        base, rows128, n_e = seg_base[eid]
        if n_e == 0: continue
        ref = A_np[base:base + n_e] @ W_np[eid * N:(eid + 1) * N].T
        err = max(err, np.abs(Out_np[base:base + n_e] - ref).max())
    print(f"v6 grouped MoE correctness: max|err| = {err:.1f}  {'PASS' if err == 0 else 'FAIL'}", flush=True)

    gflop = 2 * T * K * N / 1e9
    t_moe = bench(moe)
    A_q = A_tok[:T].to(torch.int8).contiguous()
    W_q = W_all[:N].to(torch.int8).contiguous()
    t_i8 = bench(lambda: torch._int_mm(A_q, W_q.t()))
    print(f"v6 int4 grouped-MoE  : {t_moe:7.3f} ms  {gflop/t_moe:7.1f} TFLOPS  ({gflop/t_moe/663.5*100:.1f}% of 663.5 peak)")
    print(f"int8 _int_mm         : {t_i8:7.3f} ms  {gflop/t_i8:7.1f} TFLOPS")
