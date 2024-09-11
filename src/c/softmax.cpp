#include "kernel/softmax.h"
#include "support/support.h"
#include "support/omp.h"
#include <math.h>

void softmax(float *input, float *out, const int R, const int C) {

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
  for (int i = 0; i < R; i++) {
    // 找到该行的最大值
    float *input_r = input + i * C;
    float max_val = input_r[0];
    for (int j = 1; j < C; j++) {
      if (input_r[j] > max_val) {
        max_val = input_r[j];
      }
    }

    // 减去最大值并计算指数
    float sum_exp = 0.0;
    float *out_r = out + i * C;
    for (int j = 0; j < C; j++) {
      out_r[j] = exp(input_r[j] - max_val);
      sum_exp += out_r[j];
    }

    // 归一化
    for (int j = 0; j < C; j++) {
      out_r[j] /= sum_exp;
    }
  }
}
