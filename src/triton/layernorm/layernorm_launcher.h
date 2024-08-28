#include <stdint.h>

using _layer_norm_fwd_fused_kernel_ptr_t =
    void (*)(void *, void *, void *, void *, void *, void *, int32_t, int32_t,
             float, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

using _layer_norm_bwd_dx_fused_kernel_ptr_t =
    void (*)(void *, void *, void *, void *, void *, void *, void *, void *,
             void *, int32_t, int32_t, uint32_t, uint32_t, uint32_t, uint32_t,
             uint32_t, uint32_t);

using _layer_norm_bwd_dwdb_kernel_ptr_t = void (*)(void *, void *, void *,
                                                   void *, int32_t, int32_t,
                                                   uint32_t, uint32_t, uint32_t,
                                                   uint32_t, uint32_t,
                                                   uint32_t);

extern "C" {
// Pointer type (=Memref) becomes int64_t + MemRef struct
// FIXME: understand what this int64_t is used for.

void(_layer_norm_fwd_fused)(void *, void *, void *, void *, void *, void *,
                            int32_t, int32_t, float, uint32_t, uint32_t,
                            uint32_t, uint32_t, uint32_t, uint32_t);

void(_layer_norm_bwd_dx_fused)(void *, void *, void *, void *, void *, void *,
                               void *, void *, void *, int32_t, int32_t,
                               uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                               uint32_t);

void(_layer_norm_bwd_dwdb)(void *, void *, void *, void *, int32_t, int32_t,
                           uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                           uint32_t);
}

void _layer_norm_fwd_fused_omp(uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                               _layer_norm_fwd_fused_kernel_ptr_t kernel_ptr,
                               void *arg0, void *arg1, void *arg2, void *arg3,
                               void *arg4, void *arg5, int32_t arg6,
                               int32_t arg7, float arg8);

void _layer_norm_bwd_dx_fused_omp(
    uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    _layer_norm_bwd_dx_fused_kernel_ptr_t kernel_ptr, void *arg0, void *arg1,
    void *arg2, void *arg3, void *arg4, void *arg5, void *arg6, void *arg7,
    void *arg8, int32_t arg9, int32_t arg10);

void _layer_norm_bwd_dwdb_omp(uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                              _layer_norm_bwd_dwdb_kernel_ptr_t kernel_ptr,
                              void *arg0, void *arg1, void *arg2, void *arg3,
                              int32_t arg4, int32_t arg5);