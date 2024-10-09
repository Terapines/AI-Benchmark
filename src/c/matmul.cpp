#include "kernel/matmul.h"
#include "support/omp.h"
#include "support/support.h"

#define BLOCK_SIZE_M 16
#define BLOCK_SIZE_N 16

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
    }
  }

#pragma omp parallel for collapse(2) schedule(static) num_threads(max_threads.value())
  for (int i = 0; i < M; i += BLOCK_SIZE_M) {
    int i_end = std::min(M, i + BLOCK_SIZE_M);
    for (int j = 0; j < N; j += BLOCK_SIZE_N) {
      int j_end = std::min(N, j + BLOCK_SIZE_N);
#pragma omp simd
      for (int kk = 0; kk < K; ++kk) {
        for (int ii = i; ii < i_end; ++ii) {
          for (int jj = j; jj < j_end; ++jj) {
            arg2[ii * N + jj] += arg0[ii * K + kk] * arg1[kk * N + jj];
          }
        }
      }
    }
  }

}
