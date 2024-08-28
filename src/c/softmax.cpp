#include "kernel/softmax.h"
#include <math.h>

void softmax(float *input, float *out, const int R, const int C) {
  for (int i = 0; i < R; i++) {
    // 找到该行的最大值
    float *input_r = input + i * C;
    float max_val = input_r[0];
    for (int j = 1; j < C; j++) {
      if (input_r[j] > max_val) {
        max_val = input_r[j];
      }
    }

    // 减去最大值并计算指数
    float sum_exp = 0.0;
    float *out_r = out + i * C;
    for (int j = 0; j < C; j++) {
      out_r[j] = exp(input_r[j] - max_val);
      sum_exp += out_r[j];
    }

    // 归一化
    for (int j = 0; j < C; j++) {
      out_r[j] /= sum_exp;
    }
  }
}
