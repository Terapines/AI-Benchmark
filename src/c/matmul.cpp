#include "kernel/matmul.h"
#include "support/support.h"
#include "support/omp.h"

void matmul(float *arg0, float *arg1, float *arg2, int M, int N, int K) {

  std::optional<int> max_threads = getIntEnv("TRITON_CPU_MAX_THREADS");
  if (max_threads.has_value())
    max_threads =
        std::max(1, std::min(max_threads.value(), omp_get_max_threads()));
  else
    max_threads = omp_get_max_threads();

  if (getBoolEnv("TRITON_CPU_OMP_DEBUG"))
    printf("max_threads: %d\n", max_threads.value());

    // For now, use the default chunk size, total iterations / max_threads.
#pragma omp parallel for schedule(static) num_threads(max_threads.value())
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      arg2[i * N + j] = 0;
      for (int k = 0; k < K; k++) {
        arg2[i * N + j] += arg0[i * K + k] * arg1[k * N + j];
      }
    }
  }
}
