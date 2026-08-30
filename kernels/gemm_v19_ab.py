"""v19 compiler A/B: hiprtc 7.14 vs offline amdclang 23 (same SDK install).

Offline path: amdclang --hip-path=SDK -x hip --offload-arch=gfx1201
--cuda-device-only -O3 -D__HIPCC_RTC__=1.  The __HIPCC_RTC__ define avoids the
MSVC <cmath> isgreater-overload contamination on Windows; _rtc_shim.h supplies
the four identifiers hiprtc injects (threadIdx/blockIdx/gridDim/__syncthreads)
with verbatim SDK semantics incl. the release/acquire fences around s_barrier.
The bundled object is extracted with clang-offload-bundler and loaded through
the same hipModuleLoadData/ctypes harness as the hiprtc build.

Measures: numpy-dequant correctness gate, P-sweep best-of-4 TFLOPS,
VGPR/spill counts from the .note msgpack, hot-loop ISA histogram.
__launch_bounds__ variants (offline only; hiprtc measured-ignores them).

Usage: python gemm_v19_ab.py [T] [K,k,...] [P,p,...]
"""
import sys, os, ctypes, struct, subprocess, re
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import numpy as np
import torch
import wmma_gemm_v2 as W
import gemm_v19 as V19

HIP = V19.HIP
HERE = os.path.dirname(os.path.abspath(__file__))
SDK = r"B:\git\rocm-venv\Lib\site-packages\_rocm_sdk_core"
LBIN = os.path.join(SDK, "lib", "llvm", "bin")
AMDCLANG = os.path.join(LBIN, "amdclang.exe")
BUNDLER = os.path.join(LBIN, "clang-offload-bundler.exe")
OBJDUMP = os.path.join(LBIN, "llvm-objdump.exe")


# ------------------------------------------------------------------ compilers
def compile_hiprtc(src, tag):
    RTC = W.RTC
    buf = ctypes.create_string_buffer(src.encode())
    prog = ctypes.c_void_p()
    assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p),
                                   tag.encode(), 0, None, None) == 0
    opts = (ctypes.c_char_p * 4)(b"--offload-arch=gfx1201", b"-O3", b"-DNEG_A=1", b"-DNEG_B=1")
    assert RTC.hiprtcCompileProgram(prog, 4, opts) == 0, f"hiprtc compile failed {tag}"
    csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
    code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
    data = code.raw[:csz.value]
    open(os.path.join(HERE, f"_{tag}.hsaco"), "wb").write(data)
    mod = ctypes.c_void_p()
    assert HIP.hipModuleLoadData(ctypes.byref(mod), code) == 0
    fn = ctypes.c_void_p()
    assert HIP.hipModuleGetFunction(ctypes.byref(fn), mod, b"moe_v19") == 0
    return fn, data


def compile_offline(src, tag, attr=None):
    body = open(os.path.join(HERE, "_rtc_shim.h")).read() + "\n" + src
    if attr:
        assert body.count("void moe_v19(") == 1
        body = body.replace("void moe_v19(", f"void {attr} moe_v19(")
    cpp = os.path.join(HERE, f"_{tag}.cpp")
    bundled = os.path.join(HERE, f"_{tag}.b.o")
    hsaco = os.path.join(HERE, f"_{tag}.hsaco")
    open(cpp, "w").write(body)
    r = subprocess.run([AMDCLANG, f"--hip-path={SDK}", "-x", "hip", "--offload-arch=gfx1201",
                        "--cuda-device-only", "-O3", "-D__HIPCC_RTC__=1", "-c", cpp, "-o", bundled],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-4000:])
        raise RuntimeError(f"offline compile failed {tag}")
    r2 = subprocess.run([BUNDLER, "-type=o", "-targets=hipv4-amdgcn-amd-amdhsa--gfx1201",
                         "-input", bundled, "-output", hsaco, "-unbundle"],
                        capture_output=True, text=True, timeout=300)
    assert r2.returncode == 0, r2.stderr
    data = open(hsaco, "rb").read()
    buf = ctypes.create_string_buffer(data)
    mod = ctypes.c_void_p()
    assert HIP.hipModuleLoadData(ctypes.byref(mod), buf) == 0, f"load failed {tag}"
    fn = ctypes.c_void_p()
    assert HIP.hipModuleGetFunction(ctypes.byref(fn), mod, b"moe_v19") == 0
    return fn, data


# -------------------------------------------------- code object metadata (msgpack)
def _mp(b, i=0):
    c = b[i]; i += 1
    if c <= 0x7f: return c, i
    if c >= 0xe0: return c - 256, i
    if 0x80 <= c <= 0x8f:
        n, d = c & 0xf, {}
        for _ in range(n):
            k, i = _mp(b, i); v, i = _mp(b, i); d[k] = v
        return d, i
    if 0x90 <= c <= 0x9f:
        n, a = c & 0xf, []
        for _ in range(n):
            v, i = _mp(b, i); a.append(v)
        return a, i
    if 0xa0 <= c <= 0xbf:
        n = c & 0x1f; return b[i:i + n], i + n
    if c == 0xc0: return None, i
    if c == 0xc2: return False, i
    if c == 0xc3: return True, i
    if c == 0xc4: n = b[i]; i += 1; return b[i:i + n], i + n
    if c == 0xc7: n = b[i]; return ("ext", b[i + 1], b[i + 2:i + 2 + n]), i + 2 + n
    if c == 0xc8: n = struct.unpack_from("<H", b, i)[0]; return ("ext", b[i + 2], b[i + 3:i + 3 + n]), i + 3 + n
    if c == 0xc9: n = struct.unpack_from("<I", b, i)[0]; return ("ext", b[i + 4], b[i + 5:i + 5 + n]), i + 5 + n
    if c == 0xcc: return b[i], i + 1
    if c == 0xcd: return struct.unpack_from("<H", b, i)[0], i + 2
    if c == 0xce: return struct.unpack_from("<I", b, i)[0], i + 4
    if c == 0xcf: return struct.unpack_from("<Q", b, i)[0], i + 8
    if c == 0xd0: return struct.unpack_from("<b", b, i)[0], i + 1
    if c == 0xd1: return struct.unpack_from("<h", b, i)[0], i + 2
    if c == 0xd2: return struct.unpack_from("<i", b, i)[0], i + 4
    if c == 0xd3: return struct.unpack_from("<q", b, i)[0], i + 8
    if c == 0xdb: n = struct.unpack_from("<I", b, i)[0]; i += 4; return b[i:i + n], i + n
    if c == 0xdd:
        n = struct.unpack_from("<I", b, i)[0]; i += 4; a = []
        for _ in range(n):
            v, i = _mp(b, i); a.append(v)
        return a, i
    if c == 0xd9: n = b[i]; i += 1; return b[i:i + n], i + n
    if c == 0xda: n = struct.unpack_from("<H", b, i)[0]; i += 2; return b[i:i + n], i + n
    if c == 0xdc:
        n = struct.unpack_from("<H", b, i)[0]; i += 2; a = []
        for _ in range(n):
            v, i = _mp(b, i); a.append(v)
        return a, i
    if c == 0xde:
        n = struct.unpack_from("<H", b, i)[0]; i += 2; d = {}
        for _ in range(n):
            k, i = _mp(b, i); v, i = _mp(b, i); d[k] = v
        return d, i
    raise ValueError(f"msgpack byte {c:#x} at {i - 1}")


def kernel_meta(elf):
    e_shoff = struct.unpack_from("<Q", elf, 0x28)[0]
    e_shentsize = struct.unpack_from("<H", elf, 0x3A)[0]
    e_shnum = struct.unpack_from("<H", elf, 0x3C)[0]
    blob = b""
    for i in range(e_shnum):
        o = e_shoff + i * e_shentsize
        _name, typ, _fl, _addr, off, size = struct.unpack_from("<IIQQQQ", elf, o)
        if typ == 7:  # SHT_NOTE
            blob += elf[off:off + size]
    p = 0
    while p + 12 <= len(blob):
        namesz, descsz, ntype = struct.unpack_from("<III", blob, p)
        if namesz == 0:
            break
        name = blob[p + 12:p + 12 + namesz - 1]
        doff = p + 12 + ((namesz + 3) // 4) * 4
        desc = blob[doff:doff + descsz]
        if name in (b"AMD", b"AMDGPU") and ntype == 32:  # NT_AMDGPU_METADATA
            meta, _ = _mp(desc)
            return meta[b"amdhsa"][b"kernels"][0]
        p = doff + ((descsz + 3) // 4) * 4
    raise ValueError("no amdhsa metadata note")


# ------------------------------------------------------------------ ISA stats
def isa_hist(hsaco_path, tag):
    r = subprocess.run([OBJDUMP, "--disassemble-symbols=moe_v19", "--arch-name=amdgcn",
                        "--mcpu=gfx1201", hsaco_path], capture_output=True, text=True, timeout=600)
    asm = r.stdout
    if "moe_v19" not in asm:  # fallback: disassemble all, keep only .text
        r = subprocess.run([OBJDUMP, "--disassemble-all", "--arch-name=amdgcn",
                            "--mcpu=gfx1201", hsaco_path], capture_output=True, text=True, timeout=600)
        asm = r.stdout
        if "Disassembly of section .text" in asm:
            asm = asm.split("Disassembly of section .text", 1)[1]
    open(os.path.join(HERE, f"_{tag}.asm"), "w").write(asm)
    hist = {}
    for m in re.finditer(r"^\t([a-z][a-z0-9_.]*)\s", asm, re.M):
        hist[m.group(1)] = hist.get(m.group(1), 0) + 1
    return hist


def group(hist):
    g = {}
    for op, n in hist.items():
        if op.startswith("v_wmma"): k = "v_wmma*"
        elif op.startswith(("ds_load", "ds_read")): k = "ds_load/read*"
        elif op.startswith(("ds_write", "ds_store")): k = "ds_write/store*"
        elif op.startswith("s_wait"): k = "s_wait*"
        elif op.startswith("s_barrier"): k = "s_barrier*"
        elif op.startswith("global_load"): k = "global_load*"
        elif op.startswith("global_store"): k = "global_store*"
        elif op.startswith("v_mov"): k = "v_mov*"
        elif op.startswith("v_cvt"): k = "v_cvt*"
        else: k = "other"
        g[k] = g.get(k, 0) + n
    return g


# ------------------------------------------------------------------ problem
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
    Out = torch.empty((M_pad, N), device="cuda", dtype=torch.int8)
    tile_e, tile_m, tile_n, ntiles = V19.build_tiles(seg_base, N)
    return dict(Ap=Ap, Bt=Bt, Out=Out, seg_base=seg_base, A_tok=A_tok, W_all=W_all,
                M_pad=M_pad, N=N, Kw=Kw, ntiles=ntiles,
                tile_e=tile_e, tile_m=tile_m, tile_n=tile_n)


def launch(pb, fn, scale, P):
    V19.launch_persistent(fn, pb["Ap"], pb["Bt"], pb["tile_e"], pb["tile_m"], pb["tile_n"],
                          scale, pb["Out"], pb["N"], pb["Kw"], pb["ntiles"], P)


def verify(pb, fn, scale):
    launch(pb, fn, scale, 84)
    HIP.hipDeviceSynchronize()
    host = np.empty(pb["M_pad"] * pb["N"], dtype=np.int8)
    HIP.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(pb["Out"].data_ptr()),
                  pb["M_pad"] * pb["N"], 2)
    Out_np = host.reshape(pb["M_pad"], pb["N"]).astype(np.float32)
    A_np = pb["A_tok"].cpu().numpy().astype(np.float32)
    W_np = pb["W_all"].cpu().numpy().astype(np.float32)
    err = 0.0
    for eid, (base, rows, n_e) in enumerate(pb["seg_base"]):
        if n_e == 0:
            continue
        ref = A_np[base:base + n_e] @ W_np[eid * pb["N"]:(eid + 1) * pb["N"]].T
        err = max(err, np.abs(Out_np[base:base + n_e] - (ref / 128.0)).max())
    return err


def bench_p(pb, fn, scale, P, n=30):
    def call():
        launch(pb, fn, scale, P)
    return V19.bench(call, n)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
    Ks = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [4096, 768]
    Ps = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [56, 84, 112, 168]
    n2 = 2 * V19.n_cus()
    if n2 not in Ps:
        Ps = sorted(Ps + [n2])
    torch.manual_seed(0)
    scale = torch.full((1,), 1.0 / 128.0, device="cuda")
    print(f"device: {torch.cuda.get_device_name(0)}  CUs(attr)={V19.n_cus()}  2*CUs={n2}", flush=True)

    variants = []
    fn, data = compile_hiprtc(V19.SRC, "ab_hiprtc")
    variants.append(("hiprtc-7.14", "ab_hiprtc", fn, data))
    fn, data = compile_offline(V19.SRC, "ab_clang")
    variants.append(("amdclang23-offline", "ab_clang", fn, data))
    # NOTE: __launch_bounds__ as a token does not parse in offline -x hip mode
    # (treated as an identifier -> declarator error). hiprtc accepts it and
    # ignores it (RESULTS.md Round 3) — consistent with a macro-void there.
    # The AMDGPU-native occupancy attributes are the real levers:
    fn, data = compile_offline(V19.SRC, "ab_clang_fwg",
                               attr="__attribute__((amdgpu_flat_work_group_size(512, 512)))")
    variants.append(("clang+fwg512", "ab_clang_fwg", fn, data))
    fn, data = compile_offline(V19.SRC, "ab_clang_fwg_wpe",
                               attr="__attribute__((amdgpu_flat_work_group_size(512, 512), "
                                    "amdgpu_waves_per_eu(2)))")
    variants.append(("clang+fwg+wpe2", "ab_clang_fwg_wpe", fn, data))

    print("\n=== code object metadata ===")
    for name, tag, fn, data in variants:
        try:
            km = kernel_meta(data)
            print(f"{name:20s} vgpr={km.get(b'.vgpr_count')} agpr={km.get(b'.agpr_count')} "
                  f"sgpr={km.get(b'.sgpr_count')} vgpr_spill={km.get(b'.vgpr_spill_count')} "
                  f"sgpr_spill={km.get(b'.sgpr_spill_count')} "
                  f"lds={km.get(b'.workgroup_group_segment_byte_size')} size={len(data)}B")
        except Exception as e:
            print(f"{name:20s} meta parse FAILED: {e}")

    for K in Ks:
        pb = make_problem(T, K)
        gflop = 2 * T * K * pb["N"] / 1e9
        print(f"\n=== K={K} T={T} ntiles={pb['ntiles']} gflop={gflop:.1f} ===")
        for name, tag, fn, data in variants:
            err = verify(pb, fn, scale)
            rows = []
            for P in Ps:
                t = min(bench_p(pb, fn, scale, P) for _ in range(4))
                rows.append((P, gflop / t))
            best = max(tf for _, tf in rows)
            curve = " ".join(f"P{P}:{tf:.0f}" for P, tf in rows)
            print(f"{name:20s} err={err:.1f} {'PASS' if err <= 1.0 else 'FAIL'} | "
                  f"{curve} | best {best:.1f} TFLOPS ({best / 663.5 * 100:.1f}%)", flush=True)

    print("\n=== ISA histogram (grouped; .asm files written) ===")
    hists = {name: isa_hist(os.path.join(HERE, f"_{tag}.hsaco"), tag)
             for name, tag, fn, data in variants}
    keys = ["v_wmma*", "ds_load/read*", "ds_write/store*", "s_wait*", "s_barrier*",
            "global_load*", "global_store*", "v_mov*", "v_cvt*", "other"]
    hdr = f"{'class':16s}" + "".join(f"{n:>20s}" for n in hists)
    print(hdr)
    for k in keys:
        print(f"{k:16s}" + "".join(f"{group(h).get(k, 0):>20d}" for h in hists.values()))
    for name in ["hiprtc-7.14", "amdclang23-offline"]:
        top = sorted(hists[name].items(), key=lambda kv: -kv[1])[:12]
        print(f"\n{name} top ops: " + ", ".join(f"{op}x{n}" for op, n in top))
