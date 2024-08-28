#include "kernel/layernorm.h"
#include <math.h>

void layernorm_forward(float *out, float *mean, float *rstd, float *inp,
                       float *weight, float *bias, const int N, const int D) {
  float eps = 1e-5f;

  for (int i = 0; i < N; i++) {
    // seek to the input position inp[i,:]
    float *inp_r = inp + i * D;
    // calculate the mean
    float m = 0.0f;
    for (int j = 0; j < D; j++) {
      m += inp_r[j];
    }
    m = m / D;
    // calculate the variance (without any bias correction)
    float v = 0.0f;
    for (int j = 0; j < D; j++) {
      float xshift = inp_r[j] - m;
      v += xshift * xshift;
    }
    v = v / D;
    // calculate the rstd
    float s = 1.0f / sqrtf(v + eps);
    // seek to the output position in out[i,:]
    float *out_r = out + i * D;
    for (int j = 0; j < D; j++) {
      float n = (s * (inp_r[j] - m));    // normalized output
      float o = n * weight[j] + bias[j]; // scale and shift it
      out_r[j] = o;                      // write
    }
    // cache the mean and rstd for the backward pass later
    mean[i] = m;
    rstd[i] = s;
  }
}

void layernorm_backward(float *dinp, float *dweight, float *dbias, float *dout,
                        float *inp, float *weight, float *mean, float *rstd,
                        const int N, const int D) {
  for (int i = 0; i < N; i++) {

    float *dout_r = dout + i * D;
    float *inp_r = inp + i * D;
    float *dinp_r = dinp + i * D;
    float mean_r = mean[i];
    float rstd_r = rstd[i];

    // first: two reduce operations.
    // c1 and c2 ref: triton tutorial layernorm
    float c1 = 0.0f;
    float c2 = 0.0f;
    for (int j = 0; j < D; j++) {
      float norm = (inp_r[j] - mean_r) * rstd_r;
      float dnorm_j = weight[j] * dout_r[j]; // wdy

      c2 += dnorm_j;

      c1 += dnorm_j * norm;
    }

    c2 = c2 / D;

    c1 = c1 / D;

    // now iterate again and accumulate all the gradients
    for (int j = 0; j < D; j++) {
      float norm = (inp_r[j] - mean_r) * rstd_r;
      float dnorm_j = weight[j] * dout_r[j];

      // gradient contribution to bias
      dbias[j] += dout_r[j];
      // gradient contribution to weight
      dweight[j] += norm * dout_r[j];

      // gradient contribution to input
      float dval = 0.0f;
      dval += dnorm_j;   // term 1
      dval -= c2;        // term 2
      dval -= norm * c1; // term 3
      dval *= rstd_r;    // final scale
      dinp_r[j] += dval;
    }
  }
}
