#ifdef C_KERNEL_ENABLE
#include "kernel/matmul.h"
#endif

#ifdef TRITON_KERNEL_ENABLE
#include "matmul_kernel_launcher.h"
#endif

#include "support/support.h"
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

int main(int argc, char *argv[]) {
  int M = 179;
  int N = 321;
  int K = 167;
  int RUN_COUNT = 10;

  if (argc >= 2) {
    std::vector<int> Shape = splitStringToInts(argv[1]);

    if (Shape.size()) {
      assert(Shape.size() == 4 && "Invalid shape format: MxNxKxRUN_COUNT\n");
      M = Shape.at(0);
      N = Shape.at(1);
      K = Shape.at(2);
      RUN_COUNT = Shape.at(3);
    }
  }

  printf("Data shape %dx%dx%dx%d\n", M, N, K, RUN_COUNT);

  float *arg0 = (float *)malloc(M * K * sizeof(float));
  float *arg1 = (float *)malloc(K * N * sizeof(float));

  float *ref_out = (float *)malloc(M * N * sizeof(float));
  float *real_out = (float *)malloc(M * N * sizeof(float));

  memset(real_out, 0, M * N * sizeof(float));

#ifdef CHECK_ACCURACY
  std::string DB = getDB(argv[1]);
  FILE *file = fopen(DB.c_str(), "rb");
  if (file) {
    printf("File %s open for read\n", DB.c_str());

    fread(arg0, sizeof(float), M * K, file);
    fread(arg1, sizeof(float), K * N, file);
    fread(ref_out, sizeof(float), M * N, file);
  } else {
#endif
    // Will be used to obtain a seed for the random number engine
    std::random_device rd;
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::normal_distribution<> norm_dis(0, 1);
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        arg0[i * K + j] = norm_dis(gen);
      }
    }

    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < N; ++j) {
        arg1[i * N + j] = norm_dis(gen);
      }
    }
#ifdef CHECK_ACCURACY
  }
#endif

  // triton kernel
#ifdef TRITON_KERNEL_ENABLE
  high_resolution_clock::time_point beginTime = high_resolution_clock::now();
  for (int i = 0; i < RUN_COUNT; i++) {
    matmul_kernel_omp(ceil(1.0 * M / matmul_kernel_BLOCK_SIZE_M) *
                          ceil(1.0 * N / matmul_kernel_BLOCK_SIZE_N),
                      1, 1, matmul_kernel, arg0, arg1, real_out, M, N, K, K, 1,
                      N, 1, N, 1);
  }
  high_resolution_clock::time_point endTime = high_resolution_clock::now();
  milliseconds timeInterval =
      std::chrono::duration_cast<milliseconds>(endTime - beginTime);
  cout << "Running Time: " << timeInterval.count() << " ms" << endl;
  cout << "Triton-cpu kernel: "
       << M * N * K * RUN_COUNT / (timeInterval.count() / 1000.0) / 1e9
       << " GFLOPS" << endl;
  fprintf(stderr, "Triton-cpu kernel: %f GFLOPS\n",
          M * N * K * RUN_COUNT / (timeInterval.count() / 1000.0) / 1e9);

  std::chrono::duration<double> triton_correlation_time_interval =
      endTime - beginTime;
  /// NOTE: Format running time to generate performance report easily
  PRINT_KERNEL_RUNNING_TIME(TRITON_KERNEL,
                            triton_correlation_time_interval.count())

#endif

// c kernel
#ifdef C_KERNEL_ENABLE

  high_resolution_clock::time_point beginTime = high_resolution_clock::now();
  for (int i = 0; i < RUN_COUNT; i++) {
    matmul(arg0, arg1, real_out, M, N, K);
  }
  high_resolution_clock::time_point endTime = high_resolution_clock::now();

  milliseconds timeInterval =
      std::chrono::duration_cast<milliseconds>(endTime - beginTime);
  cout << "Running Time: " << timeInterval.count() << " ms" << endl;
  cout << "c++ matmul: "
       << M * N * K * RUN_COUNT / (timeInterval.count() / 1000.0) / 1e9
       << " GFLOPS" << endl;
  fprintf(stderr, "c++ matmul: %f GFLOPS\n",
          M * N * K * RUN_COUNT / (timeInterval.count() / 1000.0) / 1e9);

  std::chrono::duration<double> c_correlation_time_interval =
      endTime - beginTime;
  /// NOTE: Format running time to generate performance report easily
  PRINT_KERNEL_RUNNING_TIME(C_KERNEL, c_correlation_time_interval.count())
#endif

#ifdef CHECK_ACCURACY
  if (file == nullptr) {
    file = fopen(DB.c_str(), "wb");

    printf("File %s open for write\n", DB.c_str());
    assert(file);

    fwrite(arg0, sizeof(float), M * K, file);
    fwrite(arg1, sizeof(float), K * N, file);

    memcpy(ref_out, real_out, M * N * sizeof(float));
    fwrite(ref_out, sizeof(float), M * N, file);
  }
  fclose(file);

  check_tensor(ref_out, real_out, M * N, "out");
#endif

  free(arg0);
  free(arg1);

  free(ref_out);
  free(real_out);
  return 0;
}
