#ifdef C_KERNEL_ENABLE
#include "kernel/layernorm.h"
#endif

#ifdef TRITON_KERNEL_ENABLE
/// TODO: Better way to include all kernel header file
#include "_layer_norm_bwd_dwdb_launcher.h"
#include "_layer_norm_bwd_dx_fused_launcher.h"
#include "_layer_norm_fwd_fused_launcher.h"
#endif

#include "support/support.h"
#include <cassert>
#include <iostream>
#include <random>
#include <stdio.h>

#include <chrono>

int main(int argc, char *argv[]) {

  std::vector<int> Shape = splitStringToInts(argv[1]);

  int N = 0; // input token length
  int D = 0; // embedding vector dimension
  int RUN_COUNT = 10;

  if (Shape.size()) {
    assert(Shape.size() == 3 && "Invalid shape format: NxDxRUN_COUNT\n");
    N = Shape.at(0);
    D = Shape.at(1);
    RUN_COUNT = Shape.at(2);
  }

  printf("Data shape %dx%dx%d\n", N, D, RUN_COUNT);

  assert(N != 0 && D != 0 && "Invalid shape\n");

  float *x = (float *)malloc(N * D * sizeof(float));
  float *w = (float *)malloc(D * sizeof(float));
  float *b = (float *)malloc(D * sizeof(float));
  float *dout = (float *)malloc(N * D * sizeof(float));

  // Will be used to obtain a seed for the random number engine
  std::random_device rd;
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> uni_dis(0, 1.0);
  for (int i = 0; i < D; ++i) {
    w[i] = uni_dis(gen);
    b[i] = uni_dis(gen);
  }

  std::uniform_real_distribution<> norm_dis(0, 1);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < D; ++j) {
      x[i * D + j] = -2.3 + 0.5 * norm_dis(gen);
      dout[i * D + j] = .1 * norm_dis(gen);
    }
  }

  // now let's calculate everything ourselves

  // c kernel
#ifdef C_KERNEL_ENABLE
  // forward pass
  float *c_out = (float *)malloc(N * D * sizeof(float));
  float *c_mean = (float *)malloc(N * sizeof(float));
  float *c_rstd = (float *)malloc(N * sizeof(float));

  // backward pass (note calloc inits grads to zero)
  float *c_dx = (float *)calloc(N * D, sizeof(float));
  float *c_dw = (float *)calloc(D, sizeof(float));
  float *c_db = (float *)calloc(D, sizeof(float));

  auto c_layernorm_begin_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < RUN_COUNT; i++) {
    layernorm_forward(c_out, c_mean, c_rstd, x, w, b, N, D);
    layernorm_backward(c_dx, c_dw, c_db, dout, x, w, c_mean, c_rstd, N, D);
  }

  auto c_layernorm_end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> c_layernorm_time_interval =
      c_layernorm_end_time - c_layernorm_begin_time;
  /// NOTE: Format running time to generate performance report easily
  PRINT_KERNEL_RUNNING_TIME(C_KERNEL, c_layernorm_time_interval.count())
  free(c_out);
  free(c_mean);
  free(c_rstd);
  free(c_dx);
  free(c_dw);
  free(c_db);
#endif

  // triton kernel
#ifdef TRITON_KERNEL_ENABLE
  float *t_out = (float *)malloc(N * D * sizeof(float));
  float *t_mean = (float *)malloc(N * sizeof(float));
  float *t_rstd = (float *)malloc(N * sizeof(float));

  int GROUP_SIZE_M = 64;
  if (D <= 8192)
    GROUP_SIZE_M = 96;
  if (D <= 4096)
    GROUP_SIZE_M = 128;
  if (D <= 1024)
    GROUP_SIZE_M = 256;
  printf("GROUP_SIZE_M: %d\n", GROUP_SIZE_M);

  float *t_dx = (float *)calloc(N * D, sizeof(float));
  float *t_dw_partial = (float *)calloc(GROUP_SIZE_M * D, sizeof(float));
  float *t_db_partial = (float *)calloc(GROUP_SIZE_M * D, sizeof(float));
  int *locks = (int *)calloc(2 * GROUP_SIZE_M, sizeof(int));
  float *t_dw = (float *)calloc(D, sizeof(float));
  float *t_db = (float *)calloc(D, sizeof(float));
  uint32_t gridX = std::ceil((float)D / 128);

  auto triton_layernorm_begin_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < RUN_COUNT; i++) {
    _layer_norm_fwd_fused_omp(N, 1, 1, &_layer_norm_fwd_fused, x, t_out, w, b,
                              t_mean, t_rstd, D, D, 1e-5);
    _layer_norm_bwd_dx_fused_omp(N, 1, 1, _layer_norm_bwd_dx_fused, t_dx, dout,
                                 t_dw_partial, t_db_partial, x, w, t_mean,
                                 t_rstd, locks, D, D);
    _layer_norm_bwd_dwdb_omp(gridX, 1, 1, _layer_norm_bwd_dwdb, t_dw_partial,
                             t_db_partial, t_dw, t_db,
                             std::min(GROUP_SIZE_M, N), D);
  }

  auto triton_layernorm_end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> triton_layernorm_time_interval =
      triton_layernorm_end_time - triton_layernorm_begin_time;
  /// NOTE: Format running time to generate performance report easily
  PRINT_KERNEL_RUNNING_TIME(TRITON_KERNEL,
                            triton_layernorm_time_interval.count())
  free(t_out);
  free(t_mean);
  free(t_rstd);
  free(t_dx);
  free(t_dw_partial);
  free(t_db_partial);
  free(locks);
  free(t_dw);
  free(t_db);
#endif

  // check correctness of backward pass
  // check_tensor(c_out, t_out, N * D, "out");
  // check_tensor(c_mean, t_mean, N, "mean");
  // check_tensor(c_rstd, t_rstd, N, "rstd");
  // check_tensor(c_dx, t_dx, N * D, "dx");
  // check_tensor(c_dw, t_dw, D, "dw");
  // check_tensor(c_db, t_db, D, "db");

  free(x);
  free(w);
  free(b);
  free(dout);

  return 0;
}