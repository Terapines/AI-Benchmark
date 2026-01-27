#include "kernel/correlation.h"
#include "support/omp.h"
#include "support/support.h"
#include <cstring>

__attribute__((noinline)) void correlation(int8_t *src0_arr, int8_t *src1_arr,
                                           int8_t *out_arr, size_t in_channel,
                                           size_t height, size_t width,
                                           size_t out_channel,
                                           size_t out_shift) {

  // // Print input parameters
  // printf("=== correlation function called ===\n");
  // printf("in_channel: %zu, height: %zu, width: %zu, out_channel: %zu, "
  //        "out_shift: %zu\n",
  //        in_channel, height, width, out_channel, out_shift);

  // // Print first few input values
  // printf("First 10 src0_arr values: ");
  // for (size_t i = 0; i < in_channel * height * width; ++i) {
  //   printf("%d ", src0_arr[i]);
  // }
  // printf("\n");

  // printf("First 10 src1_arr values: ");
  // for (size_t i = 0; i < in_channel * height * width; ++i) {
  //   printf("%d ", src1_arr[i]);
  // }
  // printf("\n");

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
      for (size_t j = 0; j < width; j += BLOCK_SIZE_W) {
        int16_t sum_data[BLOCK_SIZE_W] = {0};
        for (size_t k = 0; k < in_channel; ++k) {
          // printf("=== k=%zu ===\n", k);  // Add this
          size_t vl = std::min(BLOCK_SIZE_W, width - j);
#pragma omp simd
          for (size_t w = 0; w < vl; ++w) {
            // printf("d=%zu, i=%zu, j=%zu, k=%zu, w=%zu\n", d, i, j, k, w);
            size_t in_idx1 = k * width * height + i * width + j + w;
            // printf("in_idx1=%zu\n", in_idx1);
            size_t in_idx2 = in_idx1 - d;
            // printf("in_idx2=%zu\n", in_idx2);
            // printf("before sum_data[%zu]=%d\n", w, sum_data[w]);
            // printf("src0_arr[%zu]=%d, src1_arr[%zu]=%d, (int16_t)(src0_arr[%zu]) * src1_arr[%zu] = %d\n", in_idx1, src0_arr[in_idx1], in_idx2, src1_arr[in_idx2], in_idx1, in_idx2, (int16_t)(src0_arr[in_idx1]) * src1_arr[in_idx2]);
            sum_data[w] += (int16_t)(src0_arr[in_idx1]) * src1_arr[in_idx2];
            // printf("aftersum_data[%zu] = %d\n", w, sum_data[w]);
          }
          // printf("After k=%zu, sum_data[0]=%d\n", k, sum_data[0]);  // Add this
        }

        size_t vl = std::min(BLOCK_SIZE_W, width - j);
#pragma omp simd
        for (size_t w = 0; w < vl; ++w) {
          size_t out_idx = d * width * height + i * width + j + w;
          out_arr[out_idx] = (int8_t)(sum_data[w] >> out_shift);
          // printf("out_idx=%zu, sum_data[w]=%d, out_shift=%zu, output=%d\n",
          //         out_idx, sum_data[w], out_shift, out_arr[out_idx]);
        }
      }
    }
  }

  // // Print first few output values
  // printf("First 10 out_arr values: ");
  // for (size_t i = 0; i < out_channel * height * width; ++i) {
  //   printf("%d ", out_arr[i]);
  // }
  // printf("\n");
  // printf("=== correlation function finished ===\n");
}
