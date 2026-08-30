"""v25: v19 + int4-packed epilogue — attacks the K=768 write ceiling.

v19's epilogue: 64 per-lane byte stores (int8 out; 16B-of-32B sector runs,
2x amplification), plus a downstream int8->int4 repack for the next W4A4 layer.

Both v25 variants replace it with: nibble-clamp (scale 1/256, range [-8,7])
-> pack the lane's 8 accs into a column-word -> CROSS-LANE TRANSPOSE ->
one dword store per (lane, n-group), in the next layer's pack() format
(word (r, w) = columns 8w..8w+7, nibble i = col 8w+i) — the inter-layer
repack disappears. Per 256x128 tile: 32KB -> 16KB written, 64 byte-stores/lane
-> 8 dword-stores/lane.

  v25a (moe_v25):  ds_bpermute transpose, 8 exchanges per word. CORRECT but
    0.69-0.89x — the 64 convergent bpermutes cost more than the write savings.
  v25b (moe_v25b): LDS-scratch transpose. Each lane writes its 8 column-words
    to the (idle, post-k-loop) double buffer — slot = warp*256 + ng*32 +
    kt*16 + l16, exactly 4096 ints = the whole 16.4KB LDS — one barrier, then
    each lane gathers 8 consecutive dwords as 2x ds_read_b128 (32B-aligned,
    broadcast across the 8 r-lanes, conflict-free banks) and assembles its
    output word. 24 LDS instr/lane/tile, no convergent ops.

ds_bpermute GOTCHA (SDK's own __shfl lowering, amd_warp_functions.h:129):
the lane argument is a BYTE offset — (lane << 2), not the lane index.

Usage: python gemm_v25.py [T] [K,k,...] [int4_scale]
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

SRC_A = r"""
typedef unsigned int uint32_t;
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
#define NT 8
#define KCW 4
#define AST 6
#define BST 256
#define WARPS 16
#define BUFSZ (256 * AST + 2 * BST)
extern "C" __global__ void moe_v25(const uint32_t* __restrict__ Ap,
                                    const uint32_t* __restrict__ Bt,
                                    const int* __restrict__ tile_e,
                                    const int* __restrict__ tile_m,
                                    const int* __restrict__ tile_n,
                                    const float* __restrict__ scale,
                                    uint32_t* __restrict__ OutW,
                                    int N, int Kw, int ntiles) {
    extern __shared__ int lds[];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.y & 15;
    int col = lane & 15, kt = lane >> 4;
    int row_local = warp * 16 + col;
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        int e = tile_e[tile];
        int mb = tile_m[tile];
        int n0 = tile_n[tile];
        const uint32_t* Aptr = Ap + (long)mb * Kw;
        const uint32_t* Bptr = Bt + (long)e * Kw * N + n0;
        v8i acc[NT];
        for (int i = 0; i < NT; ++i) acc[i] = {};
        auto load = [&](int kw, int buf) {
            int* LA = lds + buf * BUFSZ;
            int* LB = LA + 256 * AST;
            for (int w = threadIdx.y * 32 + lane; w < 256 * 2; w += 512) {
                int r = w >> 1, qq = w & 1;
                *(v2i*)(LA + r * AST + qq * 2) = *(const v2i*)(Aptr + (long)r * Kw + kw + qq * 2);
            }
            for (int w = threadIdx.y * 32 + lane; w < 2 * 128; w += 512) {
                int p = w >> 7, nl = w & 127;
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
            int* LB = LA + 256 * AST;
            v2i a = *(const v2i*)(LA + row_local * AST + 2 * kt);
            int pb = kt * BST + col * 2;
            for (int ng = 0; ng < NT; ++ng) {
                v2i b = *(const v2i*)(LB + pb + ng * 32);
                acc[ng] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, a, 1, b, acc[ng], 0);
            }
            if (kw + KCW < Kw) load(kw + KCW, ((kw / KCW) + 1) & 1);
            __syncthreads();
        }
        float s = scale[0];
        int r = lane & 7;
        int q = (lane >> 3) & 1;
        int src0 = ((lane >> 4) << 4) + (q << 3);
        for (int ng = 0; ng < NT; ++ng) {
            unsigned cw = 0;
            for (int j = 0; j < 8; ++j) {
                int v = (int)((float)acc[ng][j] * s);
                v = v < -8 ? -8 : (v > 7 ? 7 : v);
                cw |= ((unsigned)v & 0xFu) << (4 * j);
            }
            unsigned w = 0;
            for (int i = 0; i < 8; ++i)
                w |= (((unsigned)__builtin_amdgcn_ds_bpermute((src0 + i) << 2,
                        (int)cw) >> (4 * r)) & 0xFu) << (4 * i);
            OutW[(long)(mb + warp * 16 + (lane >> 4) * 8 + r) * (N >> 3)
                 + ((n0 + (ng << 4) + (q << 3)) >> 3)] = w;
        }
        __syncthreads();
    }
}
"""

SRC_B = r"""
typedef unsigned int uint32_t;
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
#define NT 8
#define KCW 4
#define AST 6
#define BST 256
#define WARPS 16
#define BUFSZ (256 * AST + 2 * BST)
extern "C" __global__ void moe_v25b(const uint32_t* __restrict__ Ap,
                                     const uint32_t* __restrict__ Bt,
                                     const int* __restrict__ tile_e,
                                     const int* __restrict__ tile_m,
                                     const int* __restrict__ tile_n,
                                     const float* __restrict__ scale,
                                     uint32_t* __restrict__ OutW,
                                     int N, int Kw, int ntiles) {
    extern __shared__ int lds[];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.y & 15;
    int col = lane & 15, kt = lane >> 4;
    int row_local = warp * 16 + col;
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        int e = tile_e[tile];
        int mb = tile_m[tile];
        int n0 = tile_n[tile];
        const uint32_t* Aptr = Ap + (long)mb * Kw;
        const uint32_t* Bptr = Bt + (long)e * Kw * N + n0;
        v8i acc[NT];
        for (int i = 0; i < NT; ++i) acc[i] = {};
        auto load = [&](int kw, int buf) {
            int* LA = lds + buf * BUFSZ;
            int* LB = LA + 256 * AST;
            for (int w = threadIdx.y * 32 + lane; w < 256 * 2; w += 512) {
                int r = w >> 1, qq = w & 1;
                *(v2i*)(LA + r * AST + qq * 2) = *(const v2i*)(Aptr + (long)r * Kw + kw + qq * 2);
            }
            for (int w = threadIdx.y * 32 + lane; w < 2 * 128; w += 512) {
                int p = w >> 7, nl = w & 127;
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
            int* LB = LA + 256 * AST;
            v2i a = *(const v2i*)(LA + row_local * AST + 2 * kt);
            int pb = kt * BST + col * 2;
            for (int ng = 0; ng < NT; ++ng) {
                v2i b = *(const v2i*)(LB + pb + ng * 32);
                acc[ng] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, a, 1, b, acc[ng], 0);
            }
            if (kw + KCW < Kw) load(kw + KCW, ((kw / KCW) + 1) & 1);
            __syncthreads();
        }
        // ---- int4-packed epilogue: LDS-scratch transpose (b128 gather) ----
        // Phase 1: pack my column-word per ng into scratch
        //   slot = warp*256 + ng*32 + kt*16 + l16   (4096 ints = whole LDS,
        //   reusable after the k-loop's final barrier)
        // Phase 2 (one barrier): lane (r, q) gathers 8 consecutive dwords
        //   (2x ds_read_b128; the 8 r-lanes broadcast the same addresses)
        //   and assembles word (row, cols q*8..q*8+7).
        float s = scale[0];
        int l16 = lane & 15;
        for (int ng = 0; ng < NT; ++ng) {
            unsigned cw = 0;
            for (int j = 0; j < 8; ++j) {
                int v = (int)((float)acc[ng][j] * s);
                v = v < -8 ? -8 : (v > 7 ? 7 : v);
                cw |= ((unsigned)v & 0xFu) << (4 * j);
            }
            lds[warp * 256 + ng * 32 + (kt << 4) + l16] = (int)cw;
        }
        __syncthreads();
        int r = l16 & 7, q = l16 >> 3;
        int s0 = warp * 256 + (kt << 4) + (q << 3);
        for (int ng = 0; ng < NT; ++ng) {
            const v4i* p = (const v4i*)(lds + s0 + ng * 32);
            v4i c0 = p[0], c1 = p[1];
            unsigned w = ((unsigned)c0.x >> (4 * r)) & 0xFu;
            w |= (((unsigned)c0.y >> (4 * r)) & 0xFu) << 4;
            w |= (((unsigned)c0.z >> (4 * r)) & 0xFu) << 8;
            w |= (((unsigned)c0.w >> (4 * r)) & 0xFu) << 12;
            w |= (((unsigned)c1.x >> (4 * r)) & 0xFu) << 16;
            w |= (((unsigned)c1.y >> (4 * r)) & 0xFu) << 20;
            w |= (((unsigned)c1.z >> (4 * r)) & 0xFu) << 24;
            w |= (((unsigned)c1.w >> (4 * r)) & 0xFu) << 28;
            OutW[(long)(mb + warp * 16 + (kt << 3) + r) * (N >> 3)
                 + ((n0 + (ng << 4) + (q << 3)) >> 3)] = w;
        }
        __syncthreads();
    }
}
"""


def compile_k(src, tag, name):
    RTC = W.RTC
    buf = ctypes.create_string_buffer(src.encode())
    prog = ctypes.c_void_p()
    assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p),
                                   tag.encode(), 0, None, None) == 0
    opts = (ctypes.c_char_p * 4)(b"--offload-arch=gfx1201", b"-O3", b"-DNEG_A=1", b"-DNEG_B=1")
    rc = RTC.hiprtcCompileProgram(prog, 4, opts)
    if rc != 0:
        lsz = ctypes.c_size_t()
        RTC.hiprtcGetProgramLogSize(prog, ctypes.byref(lsz))
        log = ctypes.create_string_buffer(lsz.value)
        RTC.hiprtcGetProgramLog(prog, log)
        raise RuntimeError(f"hiprtc compile failed {tag}:\n{log.value.decode(errors='replace')}")
    csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
    code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
    open(os.path.join(HERE, f"_{tag}.hsaco"), "wb").write(code.raw[:csz.value])
    mod = ctypes.c_void_p()
    assert HIP.hipModuleLoadData(ctypes.byref(mod), code) == 0
    fn = ctypes.c_void_p()
    assert HIP.hipModuleGetFunction(ctypes.byref(fn), mod, name.encode()) == 0
    return fn


def meta_txt(path):
    r = subprocess.run([READELF, "-n", path], capture_output=True, text=True, timeout=120)
    out = {}
    for k in ["vgpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count"]:
        m = re.search(rf"\.{re.escape(k)}:\s+(\d+)", r.stdout)
        if m:
            out[k] = int(m.group(1))
    return out


def isa_stores(path, tag, symbol):
    r = subprocess.run([OBJDUMP, f"--disassemble-symbols={symbol}", "--arch-name=amdgcn",
                        "--mcpu=gfx1201", path], capture_output=True, text=True, timeout=600)
    open(os.path.join(HERE, f"_{tag}.asm"), "w").write(r.stdout)
    hist = {}
    for m in re.finditer(r"^\t([a-z][a-z0-9_.]*)\s", r.stdout, re.M):
        hist[m.group(1)] = hist.get(m.group(1), 0) + 1
    return {k: v for k, v in hist.items()
            if "global_store" in k or "bpermute" in k or "ds_read" in k or "ds_write" in k}


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
    W_all = torch.randint(-8, 8, (E * N, K), device="cuda", dtype=torch.int32)
    Ap = V19.pack(A_tok).contiguous()
    Bt = torch.cat([V19.pack_transposed(W_all[eid * N:(eid + 1) * N]) for eid in range(E)],
                   dim=0).contiguous()
    Out8 = torch.empty((M_pad, N), device="cuda", dtype=torch.int8)
    OutW = torch.empty((M_pad, N // 8), device="cuda", dtype=torch.int32)
    tile_e, tile_m, tile_n, ntiles = V19.build_tiles(seg_base, N)
    return dict(Ap=Ap, Bt=Bt, Out8=Out8, OutW=OutW, seg_base=seg_base, A_tok=A_tok,
                W_all=W_all, M_pad=M_pad, N=N, Kw=Kw, ntiles=ntiles,
                tile_e=tile_e, tile_m=tile_m, tile_n=tile_n)


def _refs(pb):
    A_np = pb["A_tok"].cpu().numpy().astype(np.float32)
    W_np = pb["W_all"].cpu().numpy().astype(np.float32)
    return A_np, W_np


def verify_v19(pb, fn, s8):
    scale = torch.full((1,), s8, device="cuda")
    V19.launch_persistent(fn, pb["Ap"], pb["Bt"], pb["tile_e"], pb["tile_m"], pb["tile_n"],
                          scale, pb["Out8"], pb["N"], pb["Kw"], pb["ntiles"], 84)
    HIP.hipDeviceSynchronize()
    host = np.empty(pb["M_pad"] * pb["N"], dtype=np.int8)
    HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(pb["Out8"].data_ptr()),
                  pb["M_pad"] * pb["N"], 2)
    Out_np = host.reshape(pb["M_pad"], pb["N"]).astype(np.float32)
    A_np, W_np = _refs(pb)
    err = 0.0
    for eid, (base, rows, n_e) in enumerate(pb["seg_base"]):
        if n_e == 0:
            continue
        ref = A_np[base:base + n_e] @ W_np[eid * pb["N"]:(eid + 1) * pb["N"]].T
        err = max(err, np.abs(Out_np[base:base + n_e] - (ref * s8)).max())
    return err


def verify_v25(pb, fn, s4):
    scale = torch.full((1,), s4, device="cuda")
    V19.launch_persistent(fn, pb["Ap"], pb["Bt"], pb["tile_e"], pb["tile_m"], pb["tile_n"],
                          scale, pb["OutW"], pb["N"], pb["Kw"], pb["ntiles"], 84)
    HIP.hipDeviceSynchronize()
    words = pb["OutW"].cpu().numpy()
    words = words.view(np.uint32) if words.dtype == np.int32 else words
    nib = ((words[:, :, None] >> (4 * np.arange(8, dtype=np.uint32)))
           & np.uint32(0xF)).reshape(pb["M_pad"], pb["N"]).astype(np.int32)
    nib = np.where(nib > 7, nib - 16, nib)
    A_np, W_np = _refs(pb)
    err = 0.0
    for eid, (base, rows, n_e) in enumerate(pb["seg_base"]):
        if n_e == 0:
            continue
        ref = A_np[base:base + n_e] @ W_np[eid * pb["N"]:(eid + 1) * pb["N"]].T
        refn = np.clip(np.trunc(ref * s4), -8, 7)
        err = max(err, np.abs(nib[base:base + n_e] - refn).max())
    return err


def bench_v(pb, fn, out, s, P, n=30):
    scale = torch.full((1,), s, device="cuda")
    def call():
        V19.launch_persistent(fn, pb["Ap"], pb["Bt"], pb["tile_e"], pb["tile_m"], pb["tile_n"],
                              scale, out, pb["N"], pb["Kw"], pb["ntiles"], P)
    return V19.bench(call, n)


if __name__ == "__main__":
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
    Ks = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [4096, 768]
    S4 = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0 / 256.0
    S8 = 1.0 / 128.0
    torch.manual_seed(0)
    print(f"v25 int4-packed epilogue: T={T} scale4={S4} scale8={S8}", flush=True)
    fn19 = V19.compile_src(V19.SRC, "v25_ref19")
    fn_a = compile_k(SRC_A, "v25a", "moe_v25")
    fn_b = compile_k(SRC_B, "v25b", "moe_v25b")

    for tag in ("v25a", "v25b"):
        km = meta_txt(os.path.join(HERE, f"_{tag}.hsaco"))
        ops = isa_stores(os.path.join(HERE, f"_{tag}.hsaco"), tag,
                         "moe_v25" if tag == "v25a" else "moe_v25b")
        print(f"{tag} metadata: {km} | LDS/store ops: {ops}", flush=True)
    print("v19 reference: vgpr=113 sgpr=46 spills=0 | 64x global_store_b8", flush=True)

    for K in Ks:
        pb = make_problem(T, K)
        gflop = 2 * T * K * pb["N"] / 1e9
        e19 = verify_v19(pb, fn19, S8)
        ea = verify_v25(pb, fn_a, S4)
        eb = verify_v25(pb, fn_b, S4)
        print(f"\n=== K={K} ntiles={pb['ntiles']} gflop={gflop:.1f} ===")
        print(f"correctness: v19 err={e19:.1f} | v25a err={ea:.1f} | v25b err={eb:.1f} "
              f"({'all PASS' if max(e19, ea, eb) <= 1 else 'FAIL'})", flush=True)
        for P in (84, 112, 168):
            t19 = min(bench_v(pb, fn19, pb["Out8"], S8, P) for _ in range(6))
            ta = min(bench_v(pb, fn_a, pb["OutW"], S4, P) for _ in range(6))
            tb = min(bench_v(pb, fn_b, pb["OutW"], S4, P) for _ in range(6))
            print(f"P={P:3d}: v19-int8 {gflop / t19:6.1f} | v25a {gflop / ta:6.1f} "
                  f"({t19 / ta:.3f}x) | v25b {gflop / tb:6.1f} ({t19 / tb:.3f}x)", flush=True)
        wb = pb["M_pad"] * pb["N"]
        print(f"write bytes/call: int8 {wb / 1e6:.1f} MB -> int4 {wb / 2e6:.1f} MB "
              f"(+ downstream repack eliminated)", flush=True)
