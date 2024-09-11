#ifdef C_KERNEL_ENABLE
#include "kernel/softmax.h"
#endif

#ifdef TRITON_KERNEL_ENABLE
#include "softmax_kernel_launcher.h"
#endif

#include "support/support.h"
#include <cassert>
#include <iostream>
#include <random>
#include <stdio.h>

#include <chrono>

int main(int argc, char *argv[]) {

  std::vector<int> Shape = splitStringToInts(argv[1]);

  int R = 0; // row
  int C = 0; // column
  int RUN_COUNT = 10;

  if (Shape.size()) {
    assert(Shape.size() == 3 && "Invalid shape format: RxCxRUN_COUNT\n");
    R = Shape.at(0);
    C = Shape.at(1);
    RUN_COUNT = Shape.at(2);
  }

  printf("Data shape %dx%dx%d\n", R, C, RUN_COUNT);

  assert(R != 0 && C != 0 && "Invalid shape\n");

  float *input = (float *)malloc(R * C * sizeof(float));

  // Will be used to obtain a seed for the random number engine
  std::random_device rd;
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> norm_dis(0, 1);
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) {
      input[i * C + j] = norm_dis(gen);
    }
  }

  // now let's calculate everything ourselves

  // c kernel
#ifdef C_KERNEL_ENABLE
  float *c_out = (float *)malloc(R * C * sizeof(float));

  auto c_softmax_begin_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < RUN_COUNT; i++) {
    softmax(input, c_out, R, C);
  }
  auto c_softmax_end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> c_softmax_time_interval =
      c_softmax_end_time - c_softmax_begin_time;
  /// NOTE: Format running time to generate performance report easily
  PRINT_KERNEL_RUNNING_TIME(C_KERNEL, c_softmax_time_interval.count())
  free(c_out);
#endif

  // triton kernel
#ifdef TRITON_KERNEL_ENABLE
  float *t_out = (float *)malloc(R * C * sizeof(float));

  auto triton_softmax_begin_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < RUN_COUNT; i++) {
    softmax_kernel_omp(R, 1, 1, &softmax_kernel, t_out, input, C, C, C);
  }
  auto triton_softmax_end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> triton_softmax_time_interval =
      triton_softmax_end_time - triton_softmax_begin_time;
  /// NOTE: Format running time to generate performance report easily
  PRINT_KERNEL_RUNNING_TIME(TRITON_KERNEL, triton_softmax_time_interval.count())
  free(t_out);
#endif

  // check correctness of backward pass
  // check_tensor(c_out, t_out, R * C, "out");

  free(input);
  return 0;
}