/*
 * @Author: yuanhang.zhang
 * @Date: 2021-09-22 15:22:24
 * @Last Modified by: yuanhang
 * @Last Modified time: 2021-09-23 10:59:46
 */

#include "kernel/correlation.h"
#include "support/support.h"
#include "support/omp.h"
#include <cstring>

__attribute__((noinline)) void correlation(const int8_t *__restrict__ src0_arr,
                                           const int8_t *__restrict__ src1_arr,
                                           int8_t *__restrict__ out_arr,
                                           size_t in_channel,
                                           size_t height,
                                           size_t width,
                                           size_t out_channel,
                                           size_t out_shift)
{

  std::optional<int> max_threads = getIntEnv("TRITON_CPU_MAX_THREADS");
  if (max_threads.has_value())
    max_threads =
        std::max(1, std::min(max_threads.value(), omp_get_max_threads()));
  else
    max_threads = omp_get_max_threads();

  if (getBoolEnv("TRITON_CPU_OMP_DEBUG"))
    printf("max_threads: %d\n", max_threads.value());
    
  // out_channel->h_blocks->w_blocks->in_channel->h_within_block(1)->w_within_block(8,simd)
  const size_t BLOCK_SIZE_W = 8;
  #pragma omp parallel for collapse(2) num_threads(max_threads.value())
  for (size_t d = 0; d < out_channel; ++d) {
    for (size_t i = 0; i < height; ++i) {
      for (size_t j = d; j < width; j += BLOCK_SIZE_W) {
        int16_t *sum_data = new int16_t[BLOCK_SIZE_W];
        memset(sum_data, 0, BLOCK_SIZE_W * sizeof(int16_t));

        for (size_t k = 0; k < in_channel; ++k) {
          #pragma omp simd
          for (size_t w = 0; w < std::min(BLOCK_SIZE_W, width - j); ++w) {
            size_t in_idx1 = k * width * height + i * width + j + w;
            size_t in_idx2 = in_idx1 - d;
            sum_data[w] += (int16_t)(src0_arr[in_idx1]) * src1_arr[in_idx2];
          }
        }

        #pragma omp simd
        for (size_t w = 0; w < std::min(BLOCK_SIZE_W, width - j); ++w) {
          size_t out_idx = d * width * height + i * width + j + w;
          out_arr[out_idx] = (int8_t)(sum_data[w] >> out_shift);
        }
        delete[] sum_data;
      }
    }
  }
}
