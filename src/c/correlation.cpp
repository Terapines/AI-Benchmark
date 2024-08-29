/*
 * @Author: yuanhang.zhang
 * @Date: 2021-09-22 15:22:24
 * @Last Modified by: yuanhang
 * @Last Modified time: 2021-09-23 10:59:46
 */

#include "kernel/correlation.h"

__attribute__((noinline)) void correlation(int8_t *src0_arr, int8_t *src1_arr,
                                           int8_t *out_arr, size_t in_channel,
                                           size_t height, size_t width,
                                           size_t out_channel,
                                           size_t out_shift) {

  size_t in_idx1 = 0, in_idx2 = 0, out_idx = 0;

  for (size_t i = 0; i < height; ++i) {
    for (size_t d = 0; d < out_channel; ++d) {
// This pragma is used by outer loop vectorization.
#pragma clang loop vectorize(assume_safety)

      for (size_t j = d; j < width; j++) {
        out_idx = d * width * height + i * width + j;
        int16_t sum_data = 0;
        for (size_t k = 0; k < in_channel; ++k) {
          in_idx1 = k * width * height + i * width + j;
          in_idx2 = k * width * height + i * width + j - d;
          sum_data += (int16_t)src0_arr[in_idx1] * src1_arr[in_idx2];
        }
        out_arr[out_idx] = (int8_t)(sum_data >> out_shift);
      }
    }
  }
}
