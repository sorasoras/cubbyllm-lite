"""Follow-up to gemm_v19_ab.py: metadata + ISA from the saved code objects
(readelf text parse — robust), plus a focused re-bench checking the P168
waves-per-EU outlier and the extended P grid (the card reports 64 CUs, so the
recorded P=84 optimum may not be this machine's optimum today).
Run after gemm_v19_ab.py."""
import sys, os, re, subprocess
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch
import gemm_v19_ab as AB
import gemm_v19 as V19

READELF = os.path.join(AB.LBIN, "llvm-readelf.exe")
variants = [("hiprtc-7.14", "ab_hiprtc"), ("amdclang23-offline", "ab_clang"),
            ("clang+fwg512", "ab_clang_fwg"), ("clang+fwg+wpe2", "ab_clang_fwg_wpe")]


def meta_txt(path):
    r = subprocess.run([READELF, "-n", path], capture_output=True, text=True, timeout=120)
    out = {}
    for k in ["vgpr_count", "agpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count",
              "workgroup_group_segment_byte_size", "max_flat_workgroup_size", "wavefront_size"]:
        m = re.search(rf"\.{re.escape(k)}:\s+(\d+)", r.stdout)
        if m:
            out[k] = int(m.group(1))
    return out


print("=== code object metadata (llvm-readelf -n) ===")
for name, tag in variants:
    km = meta_txt(os.path.join(AB.HERE, f"_{tag}.hsaco"))
    print(f"{name:20s} vgpr={km.get('vgpr_count')} agpr={km.get('agpr_count')} "
          f"sgpr={km.get('sgpr_count')} vgpr_spill={km.get('vgpr_spill_count')} "
          f"sgpr_spill={km.get('sgpr_spill_count')} lds={km.get('workgroup_group_segment_byte_size')} "
          f"maxflatWG={km.get('max_flat_workgroup_size')} wave={km.get('wavefront_size')}")

print("\n=== ISA histogram (moe_v19 only) ===")
hists = {}
for name, tag in variants:
    hists[name] = AB.isa_hist(os.path.join(AB.HERE, f"_{tag}.hsaco"), tag)
keys = ["v_wmma*", "ds_load/read*", "ds_write/store*", "s_wait*", "s_barrier*",
        "global_load*", "global_store*", "v_mov*", "v_cvt*", "other"]
print(f"{'class':16s}" + "".join(f"{n[:18]:>18s}" for n in hists))
for k in keys:
    print(f"{k:16s}" + "".join(f"{AB.group(h).get(k, 0):>18d}" for h in hists.values()))
for name in ["hiprtc-7.14", "amdclang23-offline", "clang+fwg+wpe2"]:
    top = sorted(hists[name].items(), key=lambda kv: -kv[1])[:10]
    print(f"\n{name} top ops: " + ", ".join(f"{op}x{n}" for op, n in top))

print("\n=== focused re-bench (K=4096 T=16384, best-of-8 x 40) ===")
torch.manual_seed(0)
scale = torch.full((1,), 1.0 / 128.0, device="cuda")
fn_r, _ = AB.compile_hiprtc(V19.SRC, "ab_hiprtc")
fn_w, _ = AB.compile_offline(V19.SRC, "ab_clang_fwg_wpe",
    attr="__attribute__((amdgpu_flat_work_group_size(512, 512), amdgpu_waves_per_eu(2)))")
pb = AB.make_problem(16384, 4096)
gflop = 2 * 16384 * 4096 * pb["N"] / 1e9
for name, fn, Ps in [("hiprtc", fn_r, (84, 128, 168, 192, 224)),
                     ("clang+fwg+wpe2", fn_w, (84, 168, 192, 224))]:
    for P in Ps:
        t = min(AB.bench_p(pb, fn, scale, P, n=40) for _ in range(8))
        print(f"{name:16s} P={P:3d}: {gflop / t:.1f} TFLOPS", flush=True)
