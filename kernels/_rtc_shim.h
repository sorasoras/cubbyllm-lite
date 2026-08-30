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
