#include "kernel/resize.h"
#include "support/omp.h"
#include "support/support.h"
#include <stdint.h>
#include <chrono>

static inline uint64_t rdcycle(void) {
  uint64_t cycles;
  // Read cycle counter to cycles variable
  // 0xc00 is the address of cycle CSR
  asm volatile ("rdcycle %0" : "=r" (cycles));
  return cycles;
}

static inline uint64_t rdinstret(void) {
  uint64_t cycles;
  // Read instruction retired counter to cycles variable
  // 0xc02 is the address of instret CSR
  asm volatile ("rdinstret %0" : "=r" (cycles));
  return cycles;
}

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

  // Allocate cycle storage array for each thread
  // Use max_threads to ensure array size is sufficient
  uint64_t *thread_cycles = new uint64_t[max_threads.value()];
  uint64_t *start_cycle = new uint64_t[max_threads.value()];
  uint64_t *end_cycle = new uint64_t[max_threads.value()];
  uint64_t *thread_instret = new uint64_t[max_threads.value()];
  uint64_t *start_instret = new uint64_t[max_threads.value()];
  uint64_t *end_instret = new uint64_t[max_threads.value()];
  for (int i = 0; i < max_threads.value(); i++) {
    thread_cycles[i] = 0;
    start_cycle[i] = 0;
    end_cycle[i] = 0;
    thread_instret[i] = 0;
    start_instret[i] = 0;
    end_instret[i] = 0;
  }

  uint16_t dst_height = height * 2;
  uint16_t dst_width = width * 2;
  // fraction of factor is 7;
  size_t hw_fl = 7;

#pragma omp parallel num_threads(max_threads.value())
  {
    // Get current thread ID
    int thread_id = omp_get_thread_num();

    // Use barrier to ensure all threads start synchronously
#pragma omp barrier

    // Read start cycle before loop starts (only count loop execution time)
    start_cycle[thread_id] = rdcycle();
    start_instret[thread_id] = rdinstret();
    // Execute actual parallel loop
#pragma omp for collapse(2) schedule(static)
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

    // Read end cycle before barrier (ensure loop completion is counted)
    end_cycle[thread_id] = rdcycle();
    end_instret[thread_id] = rdinstret();
    // Use barrier to ensure all threads complete computation
#pragma omp barrier

    // Calculate current thread's cycle count and store in array
    thread_cycles[thread_id] = end_cycle[thread_id] - start_cycle[thread_id];
    thread_instret[thread_id] = end_instret[thread_id] - start_instret[thread_id];
    printf("Thread %d cycles: %lu, instret: %lu\n", thread_id, thread_cycles[thread_id], thread_instret[thread_id]);
  }

  uint64_t total_cycles = 0;
  uint64_t total_instret = 0;
  for (int i = 0; i < max_threads.value(); i++) {
    if (thread_cycles[i] > 0) {
      if (thread_cycles[i] > total_cycles) {
        total_cycles = thread_cycles[i];
      }
      total_instret += thread_instret[i];
    }
  }

  printf("  Total cycles (sum): %lu\n", total_cycles);
  printf("  Total instret (sum): %lu\n", total_instret);

  // Clean up memory
  delete[] thread_cycles;
  delete[] start_cycle;
  delete[] end_cycle;
  delete[] thread_instret;
  delete[] start_instret;
  delete[] end_instret;
}
