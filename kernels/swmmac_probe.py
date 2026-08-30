"""swmmac_probe.py — decode the idx (2:4 metadata) semantics of
__builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32 on gfx1201.

Methodology (nibble_probe pattern): one warp, one SWMMAC call per launch,
host-crafted A/B fragments, D readback, offline analysis.

Fragments (mmapeak signatures):
  A: i32x2/lane = 16 stored nibbles (half of expanded K=64) — the SPARSE side.
  B: i32x4/lane = 32 nibbles — dense, 16n x 64k across the wave.
  D: i32x8/lane = 16x16 output.
  idx: int operand — scalar or per-lane (probe tests both).

Stages:
  S0  sanity: all-ones A/B at idx=0 (expect D=32 if 2:4); clamp on/off; zero-A.
  S2b A lane -> output row (B all-ones, one lane/nibble per launch).
  S3  B lane -> output column (A all-ones, one B lane active).
  S5  A(lane,pos) -> B-position k-coincidence at idx=0. Key trick: single
      A-nibble value 1 + B position-coded 1/2 by bit b => D[m][n]=1+bit_b(k_A);
      10 bits identify the matching B-position among 1024.
  S6a idx sweep 0..255: all-ones A x coded B -> per-idx D signature, cluster.
  S6b full k-map decode for one representative per idx cluster.
  S6c per-lane idx test (different idx per lane in group 0).

Usage: python swmmac_probe.py [stages, e.g. 0,2,3,5,6]   (default: all)
"""
import sys, ctypes
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import numpy as np
import torch
import wmma_gemm_v2 as W

HIP, RTC = W.HIP, W.RTC

def make_src(clamp_lit):
    # FINDING: neg_a/neg_b/clamp are compile-time constants (immediate fields in
    # the instruction encoding); idx is a runtime operand. Kernel name carries
    # the clamp literal so both variants can coexist as separate modules.
    return f"""
typedef unsigned int uint32_t;
typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
extern "C" __global__ void swmmac_probe(const uint32_t* __restrict__ A,
                                        const uint32_t* __restrict__ B,
                                        const int* __restrict__ IDX,
                                        int* __restrict__ D) {{
    int lane = threadIdx.x & 31;
    i32x2 a; a.x = (int)A[lane * 2]; a.y = (int)A[lane * 2 + 1];
    i32x4 b;
    b.x = (int)B[lane * 4]; b.y = (int)B[lane * 4 + 1];
    b.z = (int)B[lane * 4 + 2]; b.w = (int)B[lane * 4 + 3];
    i32x8 c = {{0, 0, 0, 0, 0, 0, 0, 0}};
    c = __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(true, a, true, b, c,
                                                      IDX[lane], {clamp_lit});
    int* out = D + lane * 8;
    for (int i = 0; i < 8; ++i) out[i] = c[i];
}}
"""

FN = {0: None, 1: None}
_A_t = _B_t = _I_t = _D_t = None


def compile_probe():
    for clamp in (0, 1):
        src = make_src("true" if clamp else "false")
        buf = ctypes.create_string_buffer(src.encode())
        prog = ctypes.c_void_p()
        assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p),
                                       b"swmmac_probe", 0, None, None) == 0
        opts = (ctypes.c_char_p * 2)(b"--offload-arch=gfx1201", b"-O3")
        rc = RTC.hiprtcCompileProgram(prog, 2, opts)
        if rc != 0:
            lsz = ctypes.c_size_t()
            RTC.hiprtcGetProgramLogSize(prog, ctypes.byref(lsz))
            log = ctypes.create_string_buffer(lsz.value)
            RTC.hiprtcGetProgramLog(prog, log)
            raise RuntimeError(f"hiprtc failed (clamp={clamp}):\n{log.value.decode(errors='replace')}")
        csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
        code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
        mod = ctypes.c_void_p()
        assert HIP.hipModuleLoadData(ctypes.byref(mod), code) == 0
        fn = ctypes.c_void_p()
        assert HIP.hipModuleGetFunction(ctypes.byref(fn), mod, b"swmmac_probe") == 0
        FN[clamp] = fn


def pack_words(nib):
    """(L, Wd, 8) signed nibble values -> (L, Wd) bit-packed words."""
    n = np.asarray(nib, dtype=np.int64)
    out = np.zeros(n.shape[:2], dtype=np.uint32)
    for j in range(8):
        out |= (n[:, :, j] & 0xF).astype(np.uint32) << (4 * j)
    return out


def run(A_nib, B_nib, idx, clamp=0):
    """Launch once. A_nib: (32,2,8), B_nib: (32,4,8) signed values; idx scalar or (32,)."""
    global _A_t, _B_t, _I_t, _D_t
    if _A_t is None:
        _A_t = torch.zeros((32, 2), dtype=torch.int32, device="cuda")
        _B_t = torch.zeros((32, 4), dtype=torch.int32, device="cuda")
        _I_t = torch.zeros(32, dtype=torch.int32, device="cuda")
        _D_t = torch.zeros((32, 8), dtype=torch.int32, device="cuda")
    _A_t.copy_(torch.from_numpy(pack_words(A_nib).view(np.int32)))
    _B_t.copy_(torch.from_numpy(pack_words(B_nib).view(np.int32)))
    idx_arr = np.full(32, idx, dtype=np.int64) if np.isscalar(idx) else np.asarray(idx).astype(np.int64)
    _I_t.copy_(torch.from_numpy(idx_arr.astype(np.int32)))
    _D_t.zero_()
    args = [_A_t, _B_t, _I_t, _D_t]
    storage = [ctypes.c_void_p(t.data_ptr()) if torch.is_tensor(t) else ctypes.c_int32(t) for t in args]
    ptrs = (ctypes.c_void_p * len(storage))(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p) for b in storage])
    torch.cuda.synchronize()
    st = HIP.hipModuleLaunchKernel(FN[1 if clamp else 0], 1, 1, 1, 32, 1, 1, 0, None, ptrs, None)
    torch.cuda.synchronize()
    assert st == 0, f"launch {st}"
    return _D_t.cpu().numpy()


def as_mn(D):
    """D (32,8) -> (16,16) under the dense-WMMA D hypothesis:
    D[lane][j] = (m=(lane>>4)*8+j, n=lane&15). Validated by S2 structure."""
    M = np.zeros((16, 16), dtype=np.int64)
    for l in range(32):
        for j in range(8):
            M[(l >> 4) * 8 + j, l & 15] = D[l, j]
    return M


def code_B(bit):
    """B nibble values = 1 + ((q >> bit) & 1), q = position index 0..1023."""
    q = np.arange(32 * 4 * 8, dtype=np.int64).reshape(32, 4, 8)
    return 1 + ((q >> bit) & 1)


# ------------------------------------------------------------------ stages
def stage0():
    print("== S0: sanity ==")
    A = np.ones((32, 2, 8), dtype=np.int64)
    B = np.ones((32, 4, 8), dtype=np.int64)
    for clamp in (0, 1):
        D = run(A, B, 0, clamp)
        print(f"  all-ones A/B idx=0 clamp={clamp}: unique={np.unique(D)}")
    D = run(np.zeros_like(A), B, 0)
    print(f"  zero-A: max|D|={np.abs(D).max()}")
    # spot structure check: single A nibble, all-ones B
    A = np.zeros((32, 2, 8), dtype=np.int64); A[0, 0, 0] = 1
    D = run(A, np.ones((32, 4, 8), dtype=np.int64), 0)
    M = as_mn(D)
    print(f"  A[0][0]=1, B=ones: as_mn nonzero rows={[m for m in range(16) if M[m].any()]}, "
          f"vals={np.unique(M[M != 0])}")
    print(f"    raw hits (lane,j): {[(l, j) for l in range(32) for j in range(8) if D[l, j]]}")


def stage2b():
    """One lane active (pos 0), B all-ones -> which output row each A lane owns."""
    print("== S2b: A lane -> row ==")
    B = np.ones((32, 4, 8), dtype=np.int64)
    row_of = {}
    for l in range(32):
        A = np.zeros((32, 2, 8), dtype=np.int64)
        A[l, 0, 0] = 1
        M = as_mn(run(A, B, 0))
        row_of[l] = [m for m in range(16) if M[m].any()]
    print(f"  lane -> row: {row_of}")
    return {l: v[0] for l, v in row_of.items() if len(v) == 1}


def stage3():
    """A all-ones, one B lane all-ones -> which output column(s) that B lane owns."""
    print("== S3: B lane -> column ==")
    A = np.ones((32, 2, 8), dtype=np.int64)
    col_of = {}
    for lb in range(32):
        B = np.zeros((32, 4, 8), dtype=np.int64)
        B[lb] = 1
        M = as_mn(run(A, B, 0))
        cols = [n for n in range(16) if M[:, n].any()]
        col_of[lb] = cols
    print(f"  B-lane -> cols: {col_of}")
    return col_of


def stage5(idx=0, group=0, positions=None):
    """Decode A(lane,pos) -> matching B-position via 10-bit coded B."""
    LANES = list(range(16 * group, 16 * (group + 1)))
    positions = list(range(16)) if positions is None else list(positions)
    res = {}
    for p in positions:
        acc = {}
        for b in range(10):
            A = np.zeros((32, 2, 8), dtype=np.int64)
            A[LANES, p // 8, p % 8] = 1
            M = as_mn(run(A, code_B(b), idx))
            for i, l in enumerate(LANES):
                m = ROW_OF[l]
                nz = M[m][M[m] != 0]
                acc.setdefault(l, 0)
                if len(nz):
                    acc[l] |= int(nz[0] - 1) << b
        res[p] = dict(acc)
    print(f"== S5: A-pos -> B-position (idx={idx}, group {group}) ==")
    for p in positions:
        line = " ".join(f"l{l}:{res[p].get(l, -1):4d}" for l in LANES)
        print(f"  p={p:2d}: {line}")
    return res


def stage6a():
    """Signature sweep: all-ones A x coded B for idx in 0..255 (+ specials); cluster.
    NOTE: codes must vary WITHIN a 4-nibble k-group -> use bits 0/1 of q
    (bit 3 is word-parity = constant across a group pair -> blind)."""
    print("== S6a: idx signature sweep ==")
    A = np.ones((32, 2, 8), dtype=np.int64)
    clusters = {}
    for idx in list(range(256)) + [0x44444444, 0x11111111, 0xAAAAAAAA, 0xFFFFFFFF & 0x7FFFFFFF, -1]:
        s = run(A, code_B(0), idx).tobytes() + run(A, code_B(1), idx).tobytes()
        clusters.setdefault(s, []).append(idx)
    print(f"  {len(clusters)} distinct signatures among 265 idx values:")
    for i, (sig, idxs) in enumerate(clusters.items()):
        D0 = np.frombuffer(sig[:32 * 8 * 4], dtype=np.int32).reshape(32, 8)
        M = as_mn(D0)
        print(f"  cluster {i}: n={len(idxs)} idx={[hex(x) for x in idxs[:10]]}"
              f"{'...' if len(idxs) > 10 else ''} row0={M[0][:8]}")
    return clusters


def stage6c():
    """Per-lane idx test: group-0 lanes with differing idx (codes bit 0/1 = group-internal)."""
    print("== S6c: per-lane idx ==")
    A = np.ones((32, 2, 8), dtype=np.int64)
    A[16:] = 0  # group 1 silent
    base = as_mn(run(A, code_B(0), np.zeros(32, dtype=np.int64)))
    idxs = np.zeros(32, dtype=np.int64)
    for probe_lane, probe_idx in ((0, 0), (1, 0x44444444), (2, 0xAAAAAAAA), (3, 0xEEEEEEEE)):
        idxs[:] = 0
        idxs[probe_lane] = probe_idx
        M = as_mn(run(A, code_B(0), idxs))
        m = ROW_OF[probe_lane]
        diff = not np.array_equal(M[m], base[m])
        print(f"  lane {probe_lane} idx={probe_idx:#010x}: row {m} changed={diff} "
              f"(row {M[m][:8]} vs base {base[m][:8]})")


def stage5b(idx=0, blane=0):
    """Direct k-decode: A lane 0, all 16 positions value-coded (1..7,-1..-7);
    B = single nibble 1 swept over B-lane `blane`'s 32 nibbles.
    D[0][0] = sum of values of A positions mapping to that k (collision shows as a sum)."""
    print(f"== S5b: direct k-decode via single-B sweep (idx={idx}, B-lane {blane}) ==")
    vals = [1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8, 1]  # pos15 dup -> second pass
    for half in (0, 1):
        A = np.zeros((32, 2, 8), dtype=np.int64)
        for i in range(8):
            p = half * 8 + i
            A[0, p // 8, p % 8] = vals[p] if not (half == 1 and p == 15) else 2
        out = {}
        for q in range(blane * 32, blane * 32 + 32):
            B = np.zeros((32, 4, 8), dtype=np.int64)
            B[q // 32, (q % 32) // 8, q % 8] = 1
            M = as_mn(run(A, B, idx))
            if M[0, 0] != 0:
                out[q % 32] = M[0, 0]
        print(f"  positions {half*8}-{half*8+7} values={vals[half*8:half*8+8]}:")
        print(f"    B-lane-{blane} nibble -> hit value: {out}")


def stage7():
    """idx encoding decode: value-coded A positions vs single-B-nibble sweep
    for candidate idx values. Hypothesis: idx = 8 pairs x 4 bits; each stored
    nibble has a 2-bit selector = its position within the 4-k group."""
    print("== S7: idx encoding candidates ==")
    vals = [1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8, 2]
    A = np.zeros((32, 2, 8), dtype=np.int64)
    for p, v in enumerate(vals):
        A[0, p // 8, p % 8] = v
    cands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 0x44444444, 0x11111111,
             0x55555555, 0x66666666, 0x99999999, 0xAAAAAAAA, -1 & 0xFFFFFFFF, -1]
    for idx in cands:
        hits = {}
        for q in range(32):
            B = np.zeros((32, 4, 8), dtype=np.int64)
            B[0, q // 8, q % 8] = 1
            M = as_mn(run(A, B, idx))
            if M[0, 0] != 0:
                hits[q] = int(M[0, 0])
        inv = {v: k for k, v in enumerate(vals)}
        pretty = {q: (inv.get(h, f"sum{h}") if h in inv or True else h) for q, h in hits.items()}
        print(f"  idx={idx:#010x} ({idx:11d}): B-nibble -> {pretty}")


def stage8(idx, sel_pattern):
    """Full-pair correctness test: A lane 0 all 16 nibbles value-coded, B lane 0
    32 nibbles value-coded; compare D[0][0] against the numpy expanded-dot
    reference built from `sel_pattern` (list of 8 (s_even, s_odd) group positions)."""
    rng = np.random.default_rng(7)
    aval = rng.integers(-8, 8, 16)
    bval = rng.integers(-8, 8, 32)
    A = np.zeros((32, 2, 8), dtype=np.int64)
    for p, v in enumerate(aval):
        A[0, p // 8, p % 8] = v
    B = np.zeros((32, 4, 8), dtype=np.int64)
    for q, v in enumerate(bval):
        B[0, q // 8, q % 8] = v
    D = as_mn(run(A, B, idx))
    # reference: pair t -> group nibbles 4t + s_even (for pos 2t), 4t + s_odd (pos 2t+1)
    aexp = np.zeros(32, dtype=np.int64)
    for t in range(8):
        se, so = sel_pattern[t]
        aexp[4 * t + se] = aval[2 * t]
        aexp[4 * t + so] += aval[2 * t + 1]   # += catches selector collisions
    ref = int((aexp * bval).sum())
    ok = "PASS" if D[0, 0] == ref else "FAIL"
    print(f"  idx={idx:#010x} sel={sel_pattern[:4]}...: D={D[0, 0]} ref={ref} {ok}")
    return D[0, 0] == ref


def stage9():
    """End-to-end: full-wave RANDOM 2:4 A with PER-ROW patterns + random B,
    D vs numpy expanded-dot reference. Validates the complete packer recipe:
      A lane l = kb*16+m holds row m's k-block-kb values, 8 pairs (slots 2t,
      2t+1 linear); pair t <-> B-nibble group {4t..4t+3}; idx field t =
      pos(slot 2t) | pos(slot 2t+1)<<2; B lane kb*16+n nibble kk <-> k=32kb+kk."""
    print("== S9: full-wave 2:4 correctness (random per-row patterns) ==")
    rng = np.random.default_rng(42)
    Aexp = np.zeros((16, 64), dtype=np.int64)
    A_nib = np.zeros((32, 2, 8), dtype=np.int64)
    IDX = np.zeros(32, dtype=np.int64)
    for m in range(16):
        for kb in range(2):
            lane = kb * 16 + m
            idx = 0
            for t in range(8):
                pos = rng.choice(4, 2, replace=False)
                vals = rng.integers(-8, 8, 2)
                g = 32 * kb + 4 * t
                Aexp[m, g + pos[0]] = vals[0]
                Aexp[m, g + pos[1]] = vals[1]
                A_nib[lane, (2 * t) // 8, (2 * t) % 8] = vals[0]
                A_nib[lane, (2 * t + 1) // 8, (2 * t + 1) % 8] = vals[1]
                idx |= (pos[0] & 3) << (4 * t) | (pos[1] & 3) << (4 * t + 2)
            IDX[lane] = idx
    Bexp = np.zeros((16, 64), dtype=np.int64)
    B_nib = np.zeros((32, 4, 8), dtype=np.int64)
    for n in range(16):
        for kb in range(2):
            lane = kb * 16 + n
            for kk in range(32):
                Bexp[n, 32 * kb + kk] = rng.integers(-8, 8)
                B_nib[lane, kk // 8, kk % 8] = Bexp[n, 32 * kb + kk]
    D = as_mn(run(A_nib, B_nib, IDX))
    ref = Aexp @ Bexp.T
    err = int(np.abs(D - ref).max())
    print(f"  max|D - numpy ref| = {err}  {'PASS' if err == 0 else 'FAIL'}")
    if err:
        print(f"  D[0,:4]={D[0,:4]} ref[0,:4]={ref[0,:4]}")
        print(f"  D[:,0]={D[:,0]} ref[:,0]={ref[:,0]}")
    return err == 0


ROW_OF = {}

if __name__ == "__main__":
    stages = sys.argv[1].split(",") if len(sys.argv) > 1 else ["0", "2", "3", "5", "6"]
    compile_probe()
    if "0" in stages:
        stage0()
    if "2" in stages:
        ROW_OF.update(stage2b())
    if "3" in stages:
        stage3()
    if "5" in stages and ROW_OF:
        stage5(0, 0)
        stage5(0, 1)
        stage5b(0, 0)
        stage5b(0, 16)
    if "7" in stages:
        stage7()
    if "8" in stages:
        print("== S8: encoding hypothesis correctness tests ==")
        # hypothesis: pair t nibble-2t sel = bits [4t+1:4t], nibble-(2t+1) sel = bits [4t+3:4t+2]
        def mk(sels):  # sels: list of (se, so) per pair
            x = 0
            for t, (se, so) in enumerate(sels):
                x |= (se & 3) << (4 * t) | (so & 3) << (4 * t + 2)
            return x
        stage8(mk([(0, 0)] * 8), [(0, 0)] * 8)          # idx=0 duplicate pattern
        stage8(mk([(0, 1)] * 8), [(0, 1)] * 8)          # natural 2:4
        stage8(mk([(2, 3)] * 8), [(2, 3)] * 8)
        stage8(mk([(3, 1)] * 8), [(3, 1)] * 8)
        rng = np.random.default_rng(3)
        rand = [(int(a), int(b)) for a, b in rng.integers(0, 4, (8, 2))]
        stage8(mk(rand), rand)                          # arbitrary selectors incl. collisions
        stage8(mk([(0, 2)] * 8), [(0, 2)] * 8)
        # swapped-field hypothesis control: (so, se) order in the word
        def mk2(sels):
            x = 0
            for t, (se, so) in enumerate(sels):
                x |= (so & 3) << (4 * t) | (se & 3) << (4 * t + 2)
            return x
        stage8(mk2([(0, 1)] * 8), [(0, 1)] * 8)          # if encoding is swapped, this PASSes
    if "6" in stages and ROW_OF:
        stage6a()
        stage6c()
    if "9" in stages:
        ok = stage9()
        print("PROBE COMPLETE" if ok else "PROBE FAILED")
    print("done.")
