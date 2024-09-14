#include <math.h>
#include <optional>
#include "support/support.h"
#include "support/omp.h"
#include <cstdlib>
#include <ctime>
#include "kernel/dropout.h"


void dropout(float *input, float *output, int N, float ratio, int seed)
{
    std::optional<int> max_threads = getIntEnv("TRITON_CPU_MAX_THREADS");
    if (max_threads.has_value())
        max_threads =
            std::max(1, std::min(max_threads.value(), omp_get_max_threads()));
    else
        max_threads = omp_get_max_threads();

    if (getBoolEnv("TRITON_CPU_OMP_DEBUG"))
        printf("max_threads: %d\n", max_threads.value());

    srand(seed);

#pragma omp parallel for simd schedule(static) num_threads(max_threads.value())
    for (int i = 0; i < N; ++i)
    {
        float random_value = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        if (random_value < ratio)
        {
            output[i] = 0.0f;
        }
        else
        {
            output[i] = input[i] / (1.0f - ratio);
        }
    }
}
