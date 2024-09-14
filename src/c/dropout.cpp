#include <omp.h>
#include <cstdlib> 
#include <ctime>

void dropout(float *input, float *output, int N, float ratio, int seed) {
    srand(seed);

    #pragma omp parallel for simd
    for (int i = 0; i < N; ++i) {
        float random_value = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        if (random_value < ratio) {
            output[i] = 0.0f;
        } else {
            output[i] = input[i] / (1.0f - ratio);
        }
    }
}
