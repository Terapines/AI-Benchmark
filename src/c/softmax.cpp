#include "kernel/softmax.h"
#include "support/omp.h"
#include "support/support.h"
#include <math.h>
#include <stdint.h>

static inline uint64_t rdcycle(void) {
  uint64_t cycles;
  // 读取cycle计数器到cycles变量
  // 0xc00是cycle CSR的地址
  asm volatile ("rdcycle %0" : "=r" (cycles));
  return cycles;
}

static inline uint64_t rdinstret(void) {
  uint64_t cycles;
  // 读取cycle计数器到cycles变量
  // 0xc00是cycle CSR的地址
  asm volatile ("rdinstret %0" : "=r" (cycles));
  return cycles;
}

void softmax(float *input, float *out, const int R, const int C) {

  std::optional<int> max_threads = getIntEnv("TRITON_CPU_MAX_THREADS");
  if (max_threads.has_value())
    max_threads =
        std::max(1, std::min(max_threads.value(), omp_get_max_threads()));
  else
    max_threads = omp_get_max_threads();

  if (getBoolEnv("TRITON_CPU_OMP_DEBUG"))
    printf("max_threads: %d\n", max_threads.value());

  // 为每个线程分配 cycle 存储数组
  // 使用 max_threads 确保数组大小足够
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

  #pragma omp parallel num_threads(max_threads.value())
  {
    // 获取当前线程 ID
    int thread_id = omp_get_thread_num();

    // 使用 barrier 确保所有线程同步开始
#pragma omp barrier

    // 在循环开始前读取开始 cycle（只统计循环本身的执行时间）
    start_cycle[thread_id] = rdcycle();
    start_instret[thread_id] = rdinstret();
    // 执行实际的并行循环
#pragma omp for schedule(static)
    for (int i = 0; i < R; i++) {
      // find max value in each row
      float *input_r = input + i * C;
      float max_val = input_r[0];
      for (int j = 1; j < C; j++) {
        max_val = std::fmax(input_r[j], max_val);
      }

      // sub maximum and calculate exp
      float sum_exp = 0.0;
      float *out_r = out + i * C;
      for (int j = 0; j < C; j++) {
        out_r[j] = exp(input_r[j] - max_val);
        sum_exp += out_r[j];
      }

      // normalize
      for (int j = 0; j < C; j++) {
        out_r[j] /= sum_exp;
      }
    }

    // 在 barrier 之前读取结束 cycle（确保统计到循环结束）
    end_cycle[thread_id] = rdcycle();
    end_instret[thread_id] = rdinstret();
    // 使用 barrier 确保所有线程都完成计算
#pragma omp barrier

    // 计算当前线程的 cycle 数并存储到数组中
    thread_cycles[thread_id] = end_cycle[thread_id] - start_cycle[thread_id];
    thread_instret[thread_id] = end_instret[thread_id] - start_instret[thread_id];
    printf("Thread %d cycles: %lu, instret: %lu\n", thread_id, thread_cycles[thread_id], thread_instret[thread_id]);
  }

  uint64_t total_cycles = 0;
  uint64_t total_instret = 0;
  for (int i = 0; i < max_threads.value(); i++) {
    if (thread_cycles[i] > 0) {
      total_cycles += thread_cycles[i];
      total_instret += thread_instret[i];
    }
  }

  printf("  Total cycles (sum): %lu\n", total_cycles);
  printf("  Total instret (sum): %lu\n", total_instret);

  // 清理内存
  delete[] thread_cycles;
}
