#ifdef C_KERNEL_ENABLE
#include "kernel/layernorm.h"
#endif

#ifdef TRITON_KERNEL_ENABLE
/// TODO: Better way to include all kernel header file
#include "_layer_norm_bwd_fused_launcher.h"
#include "_layer_norm_fwd_fused_launcher.h"
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

  int N = 0; // input token length
  int D = 0; // embedding vector dimension
  int RUN_COUNT = 10;

  if (argc >= 2) {
    std::vector<int> Shape = splitStringToInts(argv[1]);
    if (Shape.size()) {
      assert(Shape.size() == 3 && "Invalid shape format: NxDxRUN_COUNT\n");
      N = Shape.at(0);
      D = Shape.at(1);
      RUN_COUNT = Shape.at(2);
    }
  }

  printf("Data shape %dx%dx%d\n", N, D, RUN_COUNT);

  assert(N != 0 && D != 0 && "Invalid shape\n");

  float *x = (float *)malloc(N * D * sizeof(float));
  float *w = (float *)malloc(D * sizeof(float));
  float *b = (float *)malloc(D * sizeof(float));
  float *dout = (float *)malloc(N * D * sizeof(float));

  float *ref_out = (float *)malloc(N * D * sizeof(float));
  float *ref_mean = (float *)malloc(N * sizeof(float));
  float *ref_rstd = (float *)malloc(N * sizeof(float));
  float *ref_dx = (float *)calloc(N * D, sizeof(float));
  float *ref_dw = (float *)calloc(D, sizeof(float));
  float *ref_db = (float *)calloc(D, sizeof(float));

  float *real_out = (float *)malloc(N * D * sizeof(float));
  float *real_mean = (float *)malloc(N * sizeof(float));
  float *real_rstd = (float *)malloc(N * sizeof(float));
  float *real_dx = (float *)calloc(N * D, sizeof(float));
  float *real_dw = (float *)calloc(D, sizeof(float));
  float *real_db = (float *)calloc(D, sizeof(float));
  // forward pass
  memset(real_out, 0, N * D * sizeof(float));
  memset(real_mean, 0, N * sizeof(float));
  memset(real_rstd, 0, N * sizeof(float));

  // backward pass (note calloc inits grads to zero)
  memset(real_dx, 0, N * D * sizeof(float));
  memset(real_dw, 0, D * sizeof(float));
  memset(real_db, 0, D * sizeof(float));

  std::string DB = getDB(argv[1]);

  FILE *file = fopen(DB.c_str(), "rb");
  if (file) {
    printf("File %s open for read\n", DB.c_str());
    fread(x, sizeof(float), N * D, file);
    fread(w, sizeof(float), D, file);
    fread(b, sizeof(float), D, file);
    fread(dout, sizeof(float), N * D, file);

    fread(ref_out, sizeof(float), N * D, file);
    fread(ref_mean, sizeof(float), N, file);
    fread(ref_rstd, sizeof(float), N, file);
    fread(ref_dx, sizeof(float), N * D, file);
    fread(ref_dw, sizeof(float), D, file);
    fread(ref_db, sizeof(float), D, file);
  } else {
    // Will be used to obtain a seed for the random number engine
    std::random_device rd;
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> uni_dis(0, 1.0);
    for (int i = 0; i < D; ++i) {
      w[i] = uni_dis(gen);
      b[i] = uni_dis(gen);
    }

    std::normal_distribution<> norm_dis(0, 1);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < D; ++j) {
        x[i * D + j] = -2.3 + 0.5 * norm_dis(gen);
        dout[i * D + j] = .1 * norm_dis(gen);
      }
    }
  }

  // now let's calculate everything ourselves

  // c kernel
#ifdef C_KERNEL_ENABLE

  auto c_layernorm_begin_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < RUN_COUNT; i++) {
    layernorm_forward(real_out, real_mean, real_rstd, x, w, b, N, D);
    memset(real_dx, 0, N * D * sizeof(float));
    memset(real_dw, 0, D * sizeof(float));
    memset(real_db, 0, D * sizeof(float));
    layernorm_backward(real_dx, real_dw, real_db, dout, x, w, real_mean,
                       real_rstd, N, D);
  }

  auto c_layernorm_end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> c_layernorm_time_interval =
      c_layernorm_end_time - c_layernorm_begin_time;
  /// NOTE: Format running time to generate performance report easily
  PRINT_KERNEL_RUNNING_TIME(C_KERNEL, c_layernorm_time_interval.count())
#endif

  // triton kernel
#ifdef TRITON_KERNEL_ENABLE

  int *locks = (int *)calloc(D, sizeof(int));

  auto triton_layernorm_begin_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < RUN_COUNT; i++) {
    _layer_norm_fwd_fused_omp(N, 1, 1, &_layer_norm_fwd_fused, x, real_out, w,
                              b, real_mean, real_rstd, D, D, 1e-5);
    memset(real_dw, 0, D * sizeof(float));
    memset(real_db, 0, D * sizeof(float));
    // memset(locks, 0, D * sizeof(float));
    _layer_norm_bwd_fused_omp(N, 1, 1, _layer_norm_bwd_fused, real_dx, real_dw, real_db,
                                 dout, x, w, real_mean, real_rstd, locks, D, D);
  }

  auto triton_layernorm_end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> triton_layernorm_time_interval =
      triton_layernorm_end_time - triton_layernorm_begin_time;
  /// NOTE: Format running time to generate performance report easily
  PRINT_KERNEL_RUNNING_TIME(TRITON_KERNEL,
                            triton_layernorm_time_interval.count())

  free(locks);
#endif

  if (file == nullptr) {
    file = fopen(DB.c_str(), "wb");

    printf("File %s open for write\n", DB.c_str());
    assert(file);

    fwrite(x, sizeof(float), N * D, file);
    fwrite(w, sizeof(float), D, file);
    fwrite(b, sizeof(float), D, file);
    fwrite(dout, sizeof(float), N * D, file);

    memcpy(ref_out, real_out, N * D * sizeof(float));
    fwrite(ref_out, sizeof(float), N * D, file);
    memcpy(ref_mean, real_mean, N * sizeof(float));
    fwrite(ref_mean, sizeof(float), N, file);
    memcpy(ref_rstd, real_rstd, N * sizeof(float));
    fwrite(ref_rstd, sizeof(float), N, file);
    memcpy(ref_dx, real_dx, N * D * sizeof(float));
    fwrite(ref_dx, sizeof(float), N * D, file);
    memcpy(ref_dw, real_dw, D * sizeof(float));
    fwrite(ref_dw, sizeof(float), D, file);
    memcpy(ref_db, real_db, D * sizeof(float));
    fwrite(ref_db, sizeof(float), D, file);
  }
  fclose(file);

  // check correctness of backward pass
  check_tensor(ref_out, real_out, N * D, "out");
  check_tensor(ref_mean, real_mean, N, "mean");
  check_tensor(ref_rstd, real_rstd, N, "rstd");
  check_tensor(ref_dx, real_dx, N * D, "dx");
  check_tensor(ref_dw, real_dw, D, "dw");
  check_tensor(ref_db, real_db, D, "db");

  free(x);
  free(w);
  free(b);
  free(dout);

  free(ref_out);
  free(ref_mean);
  free(ref_rstd);
  free(ref_dx);
  free(ref_dw);
  free(ref_db);

  free(real_out);
  free(real_mean);
  free(real_rstd);
  free(real_dx);
  free(real_dw);
  free(real_db);
  return 0;
}