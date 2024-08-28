#include "softmax_kernel_launcher.h"
#include "support/omp.h"
#include "support/support.h"
#include <algorithm>
#include <optional>
#include <stdio.h>

void softmax_kernel_run_omp(uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                            softmax_kernel_kernel_ptr_t kernel_ptr, void *arg0,
                            void *arg1, int32_t arg2, int32_t arg3,
                            int32_t arg4) {
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
#pragma omp parallel for schedule(static) num_threads(max_threads.value())
  for (size_t i = 0; i < N; ++i) {
    const auto [x, y, z] = all_grids[i];
    (*kernel_ptr)(arg0, arg1, arg2, arg3, arg4, x, y, z, gridX, gridY, gridZ);
  }
}
