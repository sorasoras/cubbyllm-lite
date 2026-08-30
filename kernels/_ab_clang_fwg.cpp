// RTC-mode shim for offline hip-clang compiles on Windows (gfx1201).
//
// Rationale: hiprtc programs compile with __HIPCC_RTC__=1 and hiprtc itself
// injects threadIdx/blockIdx/gridDim/__syncthreads. A bare offline compile
// with -D__HIPCC_RTC__=1 skips the MSVC host-header contamination (the
// isgreater overload family) but the clang wrapper does not provide these
// four identifiers. This shim supplies them with the same builtins the
// SDK's own device path uses, so the kernel body compiles identically.
// The A/B correctness gate (max|err| = 0.0 vs numpy dequant) verifies the
// semantics; the ISA diff verifies the prologue matches.

typedef struct { unsigned x, y, z; } __shim_dim3_t;

static inline __device__ __shim_dim3_t __shim_threadIdx(void) {
    __shim_dim3_t t;
    t.x = (unsigned)__builtin_amdgcn_workitem_id_x();
    t.y = (unsigned)__builtin_amdgcn_workitem_id_y();
    t.z = 0u;
    return t;
}

static inline __device__ __shim_dim3_t __shim_blockIdx(void) {
    __shim_dim3_t b;
    b.x = (unsigned)__builtin_amdgcn_workgroup_id_x();
    b.y = (unsigned)__builtin_amdgcn_workgroup_id_y();
    b.z = 0u;
    return b;
}

typedef struct { unsigned x, y, z; } __shim_grid_t;

#define threadIdx (__shim_threadIdx())
#define blockIdx  (__shim_blockIdx())
#define gridDim   (__shim_grid_t{ \
    (unsigned)(__builtin_amdgcn_grid_size_x() / __builtin_amdgcn_workgroup_size_x()), 0u, 0u})

// SDK: __syncthreads -> __barrier(__CLK_GLOBAL_MEM_FENCE | __CLK_LOCAL_MEM_FENCE)
// -> __work_group_barrier -> fence(release, workgroup) + s_barrier +
//    fence(acquire, workgroup). Verbatim from amd_device_functions.h:675-682.
static inline __device__ __attribute__((convergent)) void __syncthreads(void) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}


typedef unsigned int uint32_t;
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
#define NT 8
#define KCW 4
#define AST 6
#define BST 256          // 128 cols x 2 ints per pair-row
#define WARPS 16
#define BUFSZ (256 * AST + 2 * BST)
extern "C" __global__ void __attribute__((amdgpu_flat_work_group_size(512, 512))) moe_v19(const uint32_t* __restrict__ Ap,
                                    const uint32_t* __restrict__ Bt,
                                    const int* __restrict__ tile_e,
                                    const int* __restrict__ tile_m,
                                    const int* __restrict__ tile_n,
                                    const float* __restrict__ scale,
                                    signed char* __restrict__ Out,
                                    int N, int Kw, int ntiles) {
    extern __shared__ int lds[];   // 2 bufs x (256*AST + 2*BST) = 16.4 KB
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
            // one 16x16x32 call per (row, ng): lane kt holds k 16kt..+15
            v2i a = *(const v2i*)(LA + row_local * AST + 2 * kt);
            int pb = kt * BST + col * 2;
            for (int ng = 0; ng < NT; ++ng) {
                v2i b = *(const v2i*)(LB + pb + ng * 32);
                acc[ng] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, a, 1, b, acc[ng], 0);
            }
            if (kw + KCW < Kw) load(kw + KCW, ((kw / KCW) + 1) & 1);
            __syncthreads();
        }
        int rbase = (lane >> 4) * 8;
        for (int ng = 0; ng < NT; ++ng)
            for (int j = 0; j < 8; ++j)
                Out[(long)(mb + warp * 16 + rbase + j) * N + n0 + ng * 16 + col] =
                    (signed char)((float)acc[ng][j] * scale[0]);
        __syncthreads();
    }
}
