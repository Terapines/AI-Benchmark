void layernorm_forward(float *out, float *mean, float *rstd, float *inp,
                       float *weight, float *bias, const int N, const int D);

void layernorm_backward(float *dinp, float *dweight, float *dbias, float *dout,
                        float *inp, float *weight, float *mean, float *rstd,
                        const int N, const int D);