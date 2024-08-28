#include "layernorm_launcher.h"
#include "support/omp.h"
#include "support/support.h"
#include <algorithm>
#include <optional>
#include <stdio.h>

void _layer_norm_fwd_fused_omp(uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                               _layer_norm_fwd_fused_kernel_ptr_t kernel_ptr,
                               void *arg0, void *arg1, void *arg2, void *arg3,
                               void *arg4, void *arg5, int32_t arg6,
                               int32_t arg7, float arg8) {
  // TODO: Consider using omp collapse(3) clause for simplicity?
  auto all_grids = get_all_grids(gridX, gridY, gridZ);
  size_t N = gridX * gridY * gridZ;

  std::optional<int> max_threads = getIntEnv("TRITON_CPU_MAX_THREADS");
  if (max_threads.has_value())
    max_threads =
        std::max(1, std::min(max_threads.value(), omp_get_max_threads()));
  else
    max_threads = omp_get_max_threads();

  if (getBoolEnv("TRITON_CPU_OMP_DEBUG"))
    printf("N: %zu, max_threads: %d\n", N, max_threads.value());

    // For now, use the default chunk size, total iterations / max_threads.
#pragma omp parallel for schedule() num_threads(max_threads.value())
  for (size_t i = 0; i < N; ++i) {
    const auto [x, y, z] = all_grids[i];
    (*kernel_ptr)(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, x, y, z,
                  gridX, gridY, gridZ);
  }
}

void _layer_norm_bwd_dx_fused_omp(
    uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    _layer_norm_bwd_dx_fused_kernel_ptr_t kernel_ptr, void *arg0, void *arg1,
    void *arg2, void *arg3, void *arg4, void *arg5, void *arg6, void *arg7,
    void *arg8, int32_t arg9, int32_t arg10) {
  // TODO: Consider using omp collapse(3) clause for simplicity?
  auto all_grids = get_all_grids(gridX, gridY, gridZ);
  size_t N = gridX * gridY * gridZ;

  std::optional<int> max_threads = getIntEnv("TRITON_CPU_MAX_THREADS");
  if (max_threads.has_value())
    max_threads =
        std::max(1, std::min(max_threads.value(), omp_get_max_threads()));
  else
    max_threads = omp_get_max_threads();

  if (getBoolEnv("TRITON_CPU_OMP_DEBUG"))
    printf("N: %zu, max_threads: %d\n", N, max_threads.value());

    // For now, use the default chunk size, total iterations / max_threads.
#pragma omp parallel for schedule() num_threads(max_threads.value())
  for (size_t i = 0; i < N; ++i) {
    const auto [x, y, z] = all_grids[i];
    (*kernel_ptr)(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9,
                  arg10, x, y, z, gridX, gridY, gridZ);
  }
}

void _layer_norm_bwd_dwdb_omp(uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                              _layer_norm_bwd_dwdb_kernel_ptr_t kernel_ptr,
                              void *arg0, void *arg1, void *arg2, void *arg3,
                              int32_t arg4, int32_t arg5) {
  // TODO: Consider using omp collapse(3) clause for simplicity?
  auto all_grids = get_all_grids(gridX, gridY, gridZ);
  size_t N = gridX * gridY * gridZ;

  std::optional<int> max_threads = getIntEnv("TRITON_CPU_MAX_THREADS");
  if (max_threads.has_value())
    max_threads =
        std::max(1, std::min(max_threads.value(), omp_get_max_threads()));
  else
    max_threads = omp_get_max_threads();

  if (getBoolEnv("TRITON_CPU_OMP_DEBUG"))
    printf("N: %zu, max_threads: %d\n", N, max_threads.value());

    // For now, use the default chunk size, total iterations / max_threads.
#pragma omp parallel for schedule() num_threads(max_threads.value())
  for (size_t i = 0; i < N; ++i) {
    const auto [x, y, z] = all_grids[i];
    (*kernel_ptr)(arg0, arg1, arg2, arg3, arg4, arg5, x, y, z, gridX, gridY,
                  gridZ);
  }
}