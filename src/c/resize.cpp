#include "kernel/resize.h"
#include "support/omp.h"
#include "support/support.h"

__attribute__((noinline)) void resize(int8_t *src_arr, int8_t *out_arr,
                                      uint16_t channel, uint16_t height,
                                      uint16_t width) {
  std::optional<int> max_threads = getIntEnv("TRITON_CPU_MAX_THREADS");
  if (max_threads.has_value())
    max_threads =
        std::max(1, std::min(max_threads.value(), omp_get_max_threads()));
  else
    max_threads = omp_get_max_threads();

  if (getBoolEnv("TRITON_CPU_OMP_DEBUG"))
    printf("max_threads: %d\n", max_threads.value());

  uint16_t dst_height = height * 2;
  uint16_t dst_width = width * 2;
  // fraction of factor is 7;
  size_t hw_fl = 7;

#pragma omp parallel for collapse(2) schedule(static)                          \
    num_threads(max_threads.value())
  for (uint16_t c = 0; c < channel; c++) {
    for (uint16_t h = 0; h < dst_height; h++) {
      uint16_t input_y = h << (hw_fl - 1);
      uint16_t y0 = input_y >> hw_fl;
      uint16_t y1 = std::min(y0 + (uint16_t)1, height - (uint16_t)1);

      uint32_t t0 = (uint32_t)c * width * height + (uint32_t)y0 * width;
      uint32_t t1 = (uint32_t)c * width * height + (uint32_t)y1 * width;
      uint32_t t2 =
          (uint32_t)c * dst_width * dst_height + (uint32_t)h * dst_width;
      int8_t *src_ptr0 = src_arr + t0;
      int8_t *src_ptr1 = src_arr + t1;
      int8_t *out_ptr = out_arr + t2;

#pragma omp simd
      for (size_t w = 0; w < dst_width; w++) {
        uint16_t input_x = (uint16_t)w << (hw_fl - 1);
        uint16_t x0 = (input_x >> hw_fl);
        uint16_t x1 = std::min(x0 + (uint16_t)1, width - (uint16_t)1);

        uint8_t factor = (uint8_t)1 << hw_fl;
        uint8_t w1_lambda = (uint8_t)(input_x - (x0 << hw_fl));
        uint8_t w0_lambda = factor - w1_lambda;
        uint8_t h1_lambda = (uint8_t)(input_y - (y0 << hw_fl));
        uint8_t h0_lambda = factor - h1_lambda;

        int16_t y0x0 = src_ptr0[x0];
        int16_t y0x1 = src_ptr0[x1];
        int16_t y1x0 = src_ptr1[x0];
        int16_t y1x1 = src_ptr1[x1];

        int16_t sum1 = (w0_lambda * y0x0 + w1_lambda * y0x1) >> hw_fl;
        int16_t sum2 = (w0_lambda * y1x0 + w1_lambda * y1x1) >> hw_fl;

        out_ptr[w] = (int8_t)((h0_lambda * sum1 + h1_lambda * sum2) >> hw_fl);
      }
    }
  }
}