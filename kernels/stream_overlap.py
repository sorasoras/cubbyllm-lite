"""Stream co-scheduling probe (gfx1201 / Windows ROCm): does a memory-bound
kernel overlap with a compute-bound int4 GEMM launched concurrently on a
second stream? Decides the overlap family (streams / sub-batch pipelining /
megakernel wavefronts) for the training stack.

Arms (host-timed, device-synchronized):
  gemm   : 10x v4-class int4 GEMM (M=16384, N=2048, K=4096) on stream A
  mem    : Adam-like sweep (read 2, write 1, x reps) on stream B
  serial : gemm loop then sweep, both on stream A
  over   : gemm loop on stream A concurrent with the sweep on stream B
Verdict: T_over ~ T_gemm+T_mem => streams serialize (no co-scheduling);
         T_over ~ max(T_gemm, T_mem) => full overlap.
"""
import sys
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import torch, ctypes, time
import numpy as np
import gemm_v4 as G4
HIP = G4.HIP
RTC = G4.RTC

MEM_SRC = r"""
extern "C" __global__ void memsweep(float* out, const float* a, const float* b,
                                     float s, int n, int reps) {
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int r = 0; r < reps; ++r)
        for (int i = i0; i < n; i += stride)
            out[i] = a[i] * s + b[i];
}
"""


def compile_fn(src, tag, name):
    buf = ctypes.create_string_buffer(src.encode())
    prog = ctypes.c_void_p()
    assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p),
                                   tag.encode(), 0, None, None) == 0
    opts = (ctypes.c_char_p * 2)(b"--offload-arch=gfx1201", b"-O3")
    assert RTC.hiprtcCompileProgram(prog, 2, opts) == 0, f"compile failed {tag}"
    csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
    code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
    mod = ctypes.c_void_p()
    assert HIP.hipModuleLoadData(ctypes.byref(mod), code) == 0
    fn = ctypes.c_void_p()
    assert HIP.hipModuleGetFunction(ctypes.byref(fn), mod, name.encode()) == 0
    return fn


def launch(fn, grid, block, shared, stream, args):
    storage = []
    for t in args:
        if torch.is_tensor(t):
            storage.append(ctypes.c_void_p(t.data_ptr()))
        elif isinstance(t, float):
            storage.append(ctypes.c_float(t))
        else:
            storage.append(ctypes.c_int32(t))
    ptrs = (ctypes.c_void_p * len(storage))(*[ctypes.cast(ctypes.byref(b), ctypes.c_void_p)
                                              for b in storage])
    st = HIP.hipModuleLaunchKernel(fn, grid[0], grid[1], grid[2], block[0], block[1],
                                   block[2], shared, stream, ptrs, None)
    assert st == 0, f"launch {st}"


def sync():
    assert HIP.hipDeviceSynchronize() == 0


def timed(f):
    sync()
    t0 = time.perf_counter()
    f()
    sync()
    return time.perf_counter() - t0


if __name__ == "__main__":
    sA = ctypes.c_void_p(); assert HIP.hipStreamCreate(ctypes.byref(sA)) == 0
    sB = ctypes.c_void_p(); assert HIP.hipStreamCreate(ctypes.byref(sB)) == 0
    torch.manual_seed(0)

    gfn = G4.compile_src(G4.SRC, "v4probe")
    mfn = compile_fn(MEM_SRC, "memsweep", "memsweep")

    M, N, K = 16384, 2048, 4096
    Kw = K // 8
    A = torch.randint(-8, 8, (M, K), device="cuda", dtype=torch.int32)
    Bm = torch.randint(-8, 8, (N, K), device="cuda", dtype=torch.int32)
    Ap = G4.pack(A).contiguous()
    Bt = G4.pack_transposed(Bm).contiguous()
    scale = torch.ones(1, device="cuda")
    Out = torch.empty((M, N), device="cuda", dtype=torch.float32)

    # correctness spot-check: one 64x64 tile vs numpy
    launch(gfn, (1, 1, 1), (32, 4, 1), G4.SHARED, sA,
           [Ap[:64], Bt, scale, Out[:64], 64, N, Kw])
    sync()
    ref = A[:64].cpu().numpy().astype(np.float32) @ Bm[:64].cpu().numpy().astype(np.float32).T
    err = float(np.abs(Out[:64, :64].cpu().numpy() - ref).max())
    print(f"GEMM spot-check max|err| = {err}  {'PASS' if err == 0 else 'FAIL'}", flush=True)

    Nf = 64 * 1024 * 1024
    a = torch.randn(Nf, device="cuda")
    b = torch.randn(Nf, device="cuda")
    o = torch.empty(Nf, device="cuda")

    def gemm_loop(stream, n=10):
        for _ in range(n):
            launch(gfn, (N // 64, M // 64, 1), (32, 4, 1), G4.SHARED, stream,
                   [Ap, Bt, scale, Out, M, N, Kw])

    def mem_run(stream, reps):
        launch(mfn, (512, 1, 1), (256, 1, 1), 0, stream, [o, a, b, 1.0, Nf, reps])

    T_gemm = timed(lambda: gemm_loop(sA))
    T_mem0 = timed(lambda: mem_run(sB, 24))
    reps = max(1, round(24 * T_gemm / T_mem0))          # match durations
    T_mem = timed(lambda: mem_run(sB, reps))
    print(f"gemm x10        : {T_gemm*1e3:8.1f} ms "
          f"({10 * 2*M*N*K/1e12/T_gemm:6.1f} TFLOPS int4)", flush=True)
    print(f"mem sweep x{reps:<3d}  : {T_mem*1e3:8.1f} ms "
          f"({reps * 3 * Nf * 4 / 1e9 / T_mem:6.1f} GB/s)", flush=True)

    T_serial = timed(lambda: (gemm_loop(sA), mem_run(sA, reps)))
    print(f"serial (1 stream): {T_serial*1e3:8.1f} ms", flush=True)

    for i in range(3):
        T_over = timed(lambda: (gemm_loop(sA), mem_run(sB, reps)))
        gain = (T_gemm + T_mem) / T_over
        print(f"overlapped  run{i} : {T_over*1e3:8.1f} ms   "
              f"speedup vs serial {(T_serial/T_over):.2f}x   "
              f"vs sum {gain:.2f}x   (max-alone = {max(T_gemm, T_mem)*1e3:.1f} ms)",
              flush=True)

    # control: memory kernel queued FIRST (dispatcher-bias check)
    for i in range(2):
        T_over = timed(lambda: (mem_run(sB, reps), gemm_loop(sA)))
        print(f"mem-first   run{i} : {T_over*1e3:8.1f} ms   "
              f"speedup vs serial {(T_serial/T_over):.2f}x", flush=True)

    # control: half-size GEMM grid (256 blocks ~ machine-half) + memory kernel
    def gemm_half(stream, n=10):
        for _ in range(n):
            launch(gfn, (N // 64, M // 64 // 8, 1), (32, 4, 1), G4.SHARED, stream,
                   [Ap, Bt, scale, Out, M // 8, N, Kw])
    T_gemm_h = timed(lambda: gemm_half(sA))
    for i in range(2):
        T_over = timed(lambda: (gemm_half(sA), mem_run(sB, reps)))
        print(f"half-gemm   run{i} : {T_over*1e3:8.1f} ms   (gemm-half alone "
              f"{T_gemm_h*1e3:.1f} ms, mem alone {T_mem*1e3:.1f} ms)", flush=True)
