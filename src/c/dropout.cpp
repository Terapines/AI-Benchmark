#include "kernel/dropout.h"
#include "support/omp.h"
#include "support/support.h"
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <optional>
#include <random>

void dropout(float *input, float *output, int N, float ratio, int seed) {
  std::optional<int> max_threads = getIntEnv("TRITON_CPU_MAX_THREADS");
  if (max_threads.has_value())
    max_threads =
        std::max(1, std::min(max_threads.value(), omp_get_max_threads()));
  else
    max_threads = omp_get_max_threads();

  if (getBoolEnv("TRITON_CPU_OMP_DEBUG"))
    printf("max_threads: %d\n", max_threads.value());

  srand(seed);

#pragma omp parallel num_threads(max_threads.value())
  {
    std::mt19937 rng(seed + omp_get_thread_num());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

#pragma omp for
    for (int i = 0; i < N; ++i) {
      float random_value = dist(rng);
      if (random_value < ratio) {
        output[i] = 0.0f;
      } else {
        output[i] = input[i] / (1.0f - ratio);
      }
    }
  }
}
