/*
 * @Author: yuanhang.zhang
 * @Date: 2021-09-22 15:22:24
 * @Last Modified by: yuanhang
 * @Last Modified time: 2021-09-23 10:59:46
 */

#include "kernel/correlation.h"
#include "support/omp.h"
#include "support/support.h"
#include <cstring>

__attribute__((noinline)) void correlation(int8_t *src0_arr, int8_t *src1_arr,
                                           int8_t *out_arr, size_t in_channel,
                                           size_t height, size_t width,
                                           size_t out_channel,
                                           size_t out_shift) {

  std::optional<int> max_threads = getIntEnv("TRITON_CPU_MAX_THREADS");
  if (max_threads.has_value())
    max_threads =
        std::max(1, std::min(max_threads.value(), omp_get_max_threads()));
  else
    max_threads = omp_get_max_threads();

  if (getBoolEnv("TRITON_CPU_OMP_DEBUG"))
    printf("max_threads: %d\n", max_threads.value());

  const size_t BLOCK_SIZE_W = 8;
#pragma omp parallel for collapse(2) num_threads(max_threads.value())
  for (size_t d = 0; d < out_channel; ++d) {
    for (size_t i = 0; i < height; ++i) {
      for (size_t j = d; j < width; j += BLOCK_SIZE_W) {
        int16_t sum_data[BLOCK_SIZE_W] = {0};
        for (size_t k = 0; k < in_channel; ++k) {
          size_t vl = std::min(BLOCK_SIZE_W, width - j);
#pragma omp simd
          for (size_t w = 0; w < vl; ++w) {
            size_t in_idx1 = k * width * height + i * width + j + w;
            size_t in_idx2 = in_idx1 - d;
            sum_data[w] += (int16_t)(src0_arr[in_idx1]) * src1_arr[in_idx2];
          }
        }

        size_t vl = std::min(BLOCK_SIZE_W, width - j);
#pragma omp simd
        for (size_t w = 0; w < vl; ++w) {
          size_t out_idx = d * width * height + i * width + j + w;
          out_arr[out_idx] = (int8_t)(sum_data[w] >> out_shift);
        }
      }
    }
  }
}
