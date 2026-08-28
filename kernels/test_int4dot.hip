// Native INT4xINT4 GEMM for RDNA4 (gfx1201) via V_DOT8_I32_IU4.
//
// Semantics assumption (verified numerically by this program):
//   v_dot8_i32_iu4 d, a, b, c  ->  d = c + sum_i a_i * b_i
//   with a = SIGNED int4 x8, b = UNSIGNED int4 x8  (V_DOT4_I32_IU8 analogy)
//
// Layout:
//   A (activations, M x K): ternary {-1,0,1} packed SIGNED int4, 8 per u32
//   B (weights,     N x K): signed int4 + 8 -> unsigned nibble, 8 per u32
//   Correction: sum(A.*.(B+8)) = sum(A.*.B) + 8*rowsum(A)
//              => true = raw - 8 * rowsum_A   (one scalar per row, host-precomputed)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

#include <vector>

__device__ __forceinline__ int dot8_iu(int acc, uint32_t a, uint32_t b) {
#if defined(__gfx1201__) || defined(__gfx1100__) || defined(__gfx1101__)
    asm volatile("v_dot8_i32_iu4 %0, %1, %2, %0" : "+v"(acc) : "v"(a), "v"(b));
#else
    // portable fallback (also used to validate semantics on CPU path)
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int sa = (int)(a >> (4 * i)) & 0xF; if (sa >= 8) sa -= 16;  // signed nibble
        int sb = (int)(b >> (4 * i)) & 0xF;                          // unsigned
        acc += sa * sb;
    }
#endif
    return acc;
}

extern "C" __global__ void gemm_i4_iu(const uint32_t* __restrict__ A,
                                      const uint32_t* __restrict__ Bp,
                                      const float* __restrict__ rowcorr,
                                      float* __restrict__ Out,
                                      int M, int N, int Kw) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    int acc = 0;
    const uint32_t* arow = A + (size_t)m * Kw;
    const uint32_t* brow = Bp + (size_t)n * Kw;
    for (int kw = 0; kw < Kw; ++kw)
        acc = dot8_iu(acc, arow[kw], brow[kw]);
    Out[(size_t)m * N + n] = (float)acc - rowcorr[m];
}

static void pack_signed(const std::vector<int>& v, std::vector<uint32_t>& out) {
    size_t rows = out.size();  // rows = M*Kw words total (row-major)
    (void)rows;
}

int main() {
    const int M = 256, N = 2048, K = 768;          // our e1 shape
    const int Kw = K / 8;
    srand(42);

    // ground truth operands
    std::vector<int> A(M * K), W(N * K);
    for (auto& x : A) x = (rand() % 3) - 1;                 // ternary spikes
    for (auto& x : W) x = (rand() % 16) - 8;                // signed int4
    // pack
    std::vector<uint32_t> Ap((size_t)M * Kw, 0), Bp((size_t)N * Kw, 0);
    auto pack = [&](const std::vector<int>& v, std::vector<uint32_t>& p,
                    bool offset8, size_t rows, int K_) {
        for (size_t r = 0; r < rows; ++r)
            for (int k = 0; k < K_; ++k) {
                int val = v[r * K_ + k] + (offset8 ? 8 : 0);
                p[r * (K_ / 8) + k / 8] |= ((uint32_t)(val & 0xF)) << (4 * (k % 8));
            }
    };
    pack(A, Ap, false, M, K);
    pack(W, Bp, true, N, K);

    // row correction: 8 * rowsum(A)
    std::vector<float> corr(M);
    for (int m = 0; m < M; ++m) {
        long s = 0; for (int k = 0; k < K; ++k) s += A[m * K + k];
        corr[m] = 8.0f * (float)s;
    }

    // CPU reference
    std::vector<float> ref((size_t)M * N);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            long s = 0;
            for (int k = 0; k < K; ++k) s += A[m * K + k] * W[n * K + k];
            ref[(size_t)m * N + n] = (float)s;
        }

    // device buffers
    uint32_t *dA, *dB; float *dC, *dcorr;
    hipMalloc(&dA, Ap.size() * 4); hipMalloc(&dB, Bp.size() * 4);
    hipMalloc(&dC, (size_t)M * N * 4); hipMalloc(&dcorr, M * 4);
    hipMemcpy(dA, Ap.data(), Ap.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dB, Bp.data(), Bp.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dcorr, corr.data(), M * 4, hipMemcpyHostToDevice);

    dim3 blk(32, 8), grd((N + 31) / 32, (M + 7) / 8);
    gemm_i4_iu<<<grd, blk>>>(dA, dB, dcorr, dC, M, N, Kw);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) { printf("launch err: %s\n", hipGetErrorString(err)); return 1; }
    hipDeviceSynchronize();

    std::vector<float> out((size_t)M * N);
    hipMemcpy(out.data(), dC, out.size() * 4, hipMemcpyDeviceToHost);

    double maxerr = 0; int bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double e = __builtin_fabs(out[i] - ref[i]);
        if (e > maxerr) maxerr = e;
        if (e > 0.5) bad++;
    }
    printf("correctness: max|err|=%.1f  mismatches=%zu/%zu  %s\n",
           maxerr, bad, ref.size(), bad == 0 ? "PASS" : "FAIL");

    // benchmark
    int iters = 200;
    hipEvent_t t0, t1; hipEventCreate(&t0); hipEventCreate(&t1);
    hipEventRecord(t0);
    for (int i = 0; i < iters; ++i)
        gemm_i4_iu<<<grd, blk>>>(dA, dB, dcorr, dC, M, N, Kw);
    hipEventRecord(t1); hipEventSynchronize(t1);
    float ms; hipEventElapsedTime(&ms, t0, t1);
    printf("native int4 GEMM (M=%d,K=%d,N=%d): %.3f ms/call  (fp32 was 0.13, int8 0.03)\n",
           M, K, N, ms / iters);
    return bad != 0;
}
