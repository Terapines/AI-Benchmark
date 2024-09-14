#include <math.h>

void rope(float* t, float* freqs_cos, float* freqs_sin, float* output, int seq_len, int batch_num, int head_num, int head_dim) {
    int half_dim = head_dim / 2;
    
    #pragma omp parallel for collapse(3)
    for (int seq = 0; seq < seq_len; ++seq) {
        for (int batch = 0; batch < batch_num; ++batch) {
            for (int head = 0; head < head_num; ++head) {
                float* t_1 = &t[(seq * batch_num * head_num * head_dim) + (batch * head_num * head_dim) + (head * head_dim)];
                float* t_2 = &t_1[half_dim];

                float* out_1 = &output[(seq * batch_num * head_num * head_dim) + (batch * head_num * head_dim) + (head * head_dim)];
                float* out_2 = &out_1[half_dim];

                #pragma omp simd
                for (int dim = 0; dim < half_dim; ++dim) {
                    float cos_val = freqs_cos[seq * head_dim + dim];
                    float sin_val = freqs_sin[seq * head_dim + dim];
                    out_1[dim] = t_1[dim] * cos_val - t_2[dim] * sin_val;
                    out_2[dim] = t_1[dim] * sin_val + t_2[dim] * cos_val;
                }
            }
        }
    }
}