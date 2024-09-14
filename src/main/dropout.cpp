#ifdef C_KERNEL_ENABLE
#include "kernel/dropout.h"
#endif

#ifdef TRITON_KERNEL_ENABLE
#include "dropout_kernel_launcher.h"
#endif

#include "support/support.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <random>

int main(int argc, char *argv[])
{
    int N = 4096;
    int RUN_COUNT = 10;

    float ratio = 0.5;
    int seed = 0;

    if (argc >= 2)
    {
        std::vector<int> Shape = splitStringToInts(argv[1]);
        if (Shape.size())
        {
            assert(Shape.size() == 2 && "Invalid shape format: NxRUN_COUNT\n");
            N = Shape.at(0);
            RUN_COUNT = Shape.at(1);
        }
    }

    printf("Data shape %dx%d\n", N, RUN_COUNT);

        
    assert(N != 0 && "Invalid shape\n");

    float *input = (float *)malloc(N * sizeof(float));

    float *real_out = (float *)malloc(N * sizeof(float));

    memset(real_out, 0, N * sizeof(float));

    std::string DB = getDB(argv[1]);

    FILE *file = fopen(DB.c_str(), "rb");
    if (file)
    {
        printf("File %s open for read\n", DB.c_str());
        fread(input, sizeof(float), N, file);
    }
    else
    {
        // Will be used to obtain a seed for the random number engine
        std::random_device rd;
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(-1, 1);
        std::generate(input, input + N, [&]() { return dis(gen); });
    }

#ifdef TRITON_KERNEL_ENABLE
    // run triton kernel
    printf("Start running Triton kernel %d times.\n", RUN_COUNT);

    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUN_COUNT; i++)
    {
        dropout_kernel_omp(N, 1, 1, &dropout_kernel, input, real_out, N, ratio, seed);
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::milliseconds time_interval = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

    printf("Triton output:\n");
    for (int i = 0; i < std::min(16, N); i++)
    {
        printf("%.3f  ", real_out[i]);
        if (i == std::min(16, N) - 1)
            printf("...\n");
    }

    // check ratio
    int count = 0;
    for (int i = 0; i < N; i++)
    {
        if (real_out[i] == 0)
            count++;
    }
    printf("Triton Dropout ratio: %.3f\n", (float)count / N);

    printf("Triton kernel running time: %ld ms\n", time_interval.count());

    // NOTE: The GFLOPS calculation is not accurate, just for reference
    printf("Triton kernel: %f GFLOPS\n", N * RUN_COUNT / (time_interval.count() / 1000.0) / 1e9);
#endif

#ifdef C_KERNEL_ENABLE
    // run c++ kernel
    printf("Start running c++ kernel %d times.\n", RUN_COUNT);

    std::chrono::high_resolution_clock::time_point begin_c = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUN_COUNT; i++)
    {
        dropout(input, real_out, N, ratio, seed);
    }
    std::chrono::high_resolution_clock::time_point end_c = std::chrono::high_resolution_clock::now();
    std::chrono::milliseconds time_interval_c = std::chrono::duration_cast<std::chrono::milliseconds>(end_c - begin_c);

    printf("c++ output:\n");
    for (int i = 0; i < std::min(16, N); i++)
    {
        printf("%.3f  ", real_out[i]);
        if (i == std::min(16, N) - 1)
            printf("...\n");
    }

    // check ratio
    count = 0;
    for (int i = 0; i < N; i++)
    {
        if (real_out[i] == 0)
            count++;
    }
    printf("C++ Dropout ratio: %.3f\n", (float)count / N);

    printf("c++ kernel running time: %ld ms\n", time_interval_c.count());

    // NOTE: The GFLOPS calculation is not accurate, just for reference
    printf("c++ kernel: %f GFLOPS\n", N * RUN_COUNT / (time_interval_c.count() / 1000.0) / 1e9);
#endif

    return 0;
}