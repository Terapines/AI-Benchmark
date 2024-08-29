#include "kernel/softmax.h"
#include "softmax_kernel_launcher.h"
#include "support/support.h"
#include <cassert>
#include <iostream>
#include <random>
#include <stdio.h>

int main(int argc, char *argv[]) {

  std::vector<int> Shape = splitStringToInts(argv[1]);

  int R = 0; // row
  int C = 0; // column

  if (Shape.size()) {
    assert(Shape.size() == 2 && "Invalid shape format: RxC\n");
    R = Shape.at(0);
    C = Shape.at(1);
  }

  printf("Data shape %dx%d\n", R, C);

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
  float *c_out = (float *)malloc(R * C * sizeof(float));

  auto c_softmax_begin_time = std::chrono::high_resolution_clock::now();
  softmax(input, c_out, R, C);
  auto c_softmax_end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> c_softmax_time_interval =
      c_softmax_end_time - c_softmax_begin_time;
  /// NOTE: Format running time to generate performance report easily
  PRINT_KERNEL_RUNNING_TIME(C_KERNEL, c_softmax_time_interval.count())

  // triton kernel
  float *t_out = (float *)malloc(R * C * sizeof(float));

  auto triton_softmax_begin_time = std::chrono::high_resolution_clock::now();
  softmax_kernel_omp(R, 1, 1, &softmax_kernel, t_out, input, C, C, C);
  auto triton_softmax_end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> triton_softmax_time_interval =
      triton_softmax_end_time - triton_softmax_begin_time;
  /// NOTE: Format running time to generate performance report easily
  PRINT_KERNEL_RUNNING_TIME(TRITON_KERNEL, triton_softmax_time_interval.count())

  // check correctness of backward pass
  check_tensor(c_out, t_out, R * C, "out");

  free(input);
  free(c_out);
  free(t_out);
  return 0;
}