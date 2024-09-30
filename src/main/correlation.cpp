
#ifdef C_KERNEL_ENABLE
#include "kernel/correlation.h"
#endif

#ifdef TRITON_KERNEL_ENABLE
#include "correlation_kernel_launcher.h"
#endif

#include "support/support.h"
#include <cassert>
#include <iostream>
#include <random>
#include <stdio.h>

#include <chrono>
#include <cstring>
#include <memory>

#define OUT_SHIFT 0

int main(int argc, char *argv[]) {

  std::vector<int> Shape = splitStringToInts(argv[1]);

  int OUT_CHANNEL = 5;
  int IN_CHANNEL = 58;
  int HEIGHT = 112;
  int WIDTH = 88;
  int RUN_COUNT = 10;

  if (Shape.size()) {
    assert(Shape.size() == 5 &&
           "Invalid shape format: "
           "OUT_CHANNELxIN_CHANNELxHEIGHTxWIDTHxRUN_COUNT\n");
    OUT_CHANNEL = Shape.at(0);
    IN_CHANNEL = Shape.at(1);
    HEIGHT = Shape.at(2);
    WIDTH = Shape.at(3);
    RUN_COUNT = Shape.at(4);
  }

  printf("Data shape %dx%dx%dx%dx%d\n", OUT_CHANNEL, IN_CHANNEL, HEIGHT, WIDTH,
         RUN_COUNT);

  assert(OUT_CHANNEL != 0 && IN_CHANNEL != 0 && HEIGHT != 0 && WIDTH != 0 &&
         "Invalid shape\n");

  int IN_SIZE = HEIGHT * WIDTH * IN_CHANNEL;
  int OUT_SIZE = HEIGHT * WIDTH * OUT_CHANNEL;
  int8_t *src0_arr_global = new int8_t[IN_SIZE]; // 58x112x88
  int8_t *src1_arr_global = new int8_t[IN_SIZE]; // 58x112x88
  int8_t *ref_out = new int8_t[OUT_SIZE];        // 5x112x88
  int8_t *real_out = new int8_t[OUT_SIZE];       // 5x112x88

  memset(real_out, 0, OUT_SIZE);

#ifdef CHECK_ACCURACY
  std::string DB = getDB(argv[1]);
  FILE *file = fopen(DB.c_str(), "rb");
  if (file) {
    printf("File %s open for read\n", DB.c_str());
    fread(src0_arr_global, sizeof(int8_t), IN_SIZE, file);
    fread(src1_arr_global, sizeof(int8_t), IN_SIZE, file);
    fread(ref_out, sizeof(int8_t), OUT_SIZE, file);
  } else {
#endif
    // Will be used to obtain a seed for the random number engine
    std::random_device rd;
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> uni_dis(0, 255);
    for (int i = 0; i < IN_SIZE; ++i) {
      src0_arr_global[i] = uni_dis(gen);
      src1_arr_global[i] = uni_dis(gen);
    }
#ifdef CHECK_ACCURACY
  }
#endif

  // now let's calculate everything ourselves

  // c kernel
#ifdef C_KERNEL_ENABLE

  auto c_correlation_begin_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < RUN_COUNT; i++) {
    correlation(src0_arr_global, src1_arr_global, real_out, IN_CHANNEL, HEIGHT,
                WIDTH, OUT_CHANNEL, OUT_SHIFT);
  }
  auto c_correlation_end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> c_correlation_time_interval =
      c_correlation_end_time - c_correlation_begin_time;
  /// NOTE: Format running time to generate performance report easily
  PRINT_KERNEL_RUNNING_TIME(C_KERNEL, c_correlation_time_interval.count())

#endif

  // triton kernel
#ifdef TRITON_KERNEL_ENABLE

  // BLOCK_SHAPE 64x32x8
  // int grid = ceil((float)HEIGHT / 32) * ceil((float)WIDTH / 8);
  int gridZ = OUT_CHANNEL;
  int gridY = ceil((float)HEIGHT / correlation_kernel_BLOCK_SIZE_H);
  int gridX = ceil((float)WIDTH / correlation_kernel_BLOCK_SIZE_W);

  auto triton_correlation_begin_time =
      std::chrono::high_resolution_clock::now();
  for (int i = 0; i < RUN_COUNT; i++) {
    correlation_kernel_omp(gridX, gridY, gridZ, &correlation_kernel,
                           src0_arr_global, src1_arr_global, real_out,
                           OUT_CHANNEL, IN_CHANNEL, HEIGHT, WIDTH,
                           HEIGHT * WIDTH, OUT_SHIFT);
  }
  auto triton_correlation_end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> triton_correlation_time_interval =
      triton_correlation_end_time - triton_correlation_begin_time;
  /// NOTE: Format running time to generate performance report easily
  PRINT_KERNEL_RUNNING_TIME(TRITON_KERNEL,
                            triton_correlation_time_interval.count())

#endif

#ifdef CHECK_ACCURACY
  // check correctness of backward pass
  if (file == nullptr) {
    file = fopen(DB.c_str(), "wb");
    printf("File %s open for write\n", DB.c_str());
    assert(file);
    memcpy(ref_out, real_out, OUT_SIZE * sizeof(int8_t));

    fwrite(src0_arr_global, sizeof(int8_t), IN_SIZE, file);
    fwrite(src1_arr_global, sizeof(int8_t), IN_SIZE, file);
    fwrite(ref_out, sizeof(int8_t), OUT_SIZE, file);
  }
  fclose(file);

  check_tensor<int8_t>(ref_out, real_out, OUT_SIZE, "out");
#endif

  delete[] src0_arr_global;
  delete[] src1_arr_global;
  delete[] ref_out;  // 5x112x88
  delete[] real_out; // 5x112x88

  return 0;
}