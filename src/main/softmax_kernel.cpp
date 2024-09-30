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
#include <cstring>
#include <memory>

int main(int argc, char *argv[]) {

  int R = 0; // row
  int C = 0; // column
  int RUN_COUNT = 10;

  if (argc >= 2) {
    std::vector<int> Shape = splitStringToInts(argv[1]);
    if (Shape.size()) {
      assert(Shape.size() == 3 && "Invalid shape format: RxCxRUN_COUNT\n");
      R = Shape.at(0);
      C = Shape.at(1);
      RUN_COUNT = Shape.at(2);
    }
  }

  printf("Data shape %dx%dx%d\n", R, C, RUN_COUNT);

  assert(R != 0 && C != 0 && "Invalid shape\n");

  float *input = (float *)malloc(R * C * sizeof(float));

  float *ref_out = (float *)malloc(R * C * sizeof(float));
  float *real_out = (float *)malloc(R * C * sizeof(float));

  memset(real_out, 0, R * C * sizeof(float));

#ifdef CHECK_ACCURACY
  std::string DB = getDB(argv[1]);
  FILE *file = fopen(DB.c_str(), "rb");
  if (file) {
    printf("File %s open for read\n", DB.c_str());

    fread(input, sizeof(float), R * C, file);
    fread(ref_out, sizeof(float), R * C, file);
  } else {
#endif
    // Will be used to obtain a seed for the random number engine
    std::random_device rd;
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::normal_distribution<float> norm_dis(0, 1);
    for (int i = 0; i < R; ++i) {
      for (int j = 0; j < C; ++j) {
        input[i * C + j] = norm_dis(gen);
      }
    }
#ifdef CHECK_ACCURACY
  }
#endif

  // now let's calculate everything ourselves

  // c kernel
#ifdef C_KERNEL_ENABLE

  auto c_softmax_begin_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < RUN_COUNT; i++) {
    softmax(input, real_out, R, C);
  }
  auto c_softmax_end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> c_softmax_time_interval =
      c_softmax_end_time - c_softmax_begin_time;
  /// NOTE: Format running time to generate performance report easily
  PRINT_KERNEL_RUNNING_TIME(C_KERNEL, c_softmax_time_interval.count())

#endif

  // triton kernel
#ifdef TRITON_KERNEL_ENABLE

  auto triton_softmax_begin_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < RUN_COUNT; i++) {
    softmax_kernel_omp(R, 1, 1, &softmax_kernel, real_out, input, C, C, C);
  }
  auto triton_softmax_end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> triton_softmax_time_interval =
      triton_softmax_end_time - triton_softmax_begin_time;
  /// NOTE: Format running time to generate performance report easily
  PRINT_KERNEL_RUNNING_TIME(TRITON_KERNEL, triton_softmax_time_interval.count())

#endif

#ifdef CHECK_ACCURACY
  if (file == nullptr) {
    file = fopen(DB.c_str(), "wb");

    printf("File %s open for write\n", DB.c_str());
    assert(file);

    fwrite(input, sizeof(float), R * C, file);

    memcpy(ref_out, real_out, R * C * sizeof(float));
    fwrite(ref_out, sizeof(float), R * C, file);
  }
  fclose(file);

  // check correctness of backward pass
  check_tensor(ref_out, real_out, R * C, "out");
#endif

  free(input);
  free(ref_out);
  free(real_out);

  return 0;
}