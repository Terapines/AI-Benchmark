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

#ifdef CHECK_ACCURACY
    std::string DB = getDB(argv[1]);
    FILE *file = fopen(DB.c_str(), "rb");
    if (file)
    {
        printf("File %s open for read\n", DB.c_str());
        fread(input, sizeof(float), N, file);
    }
    else
    {
#endif
        // Will be used to obtain a seed for the random number engine
        std::random_device rd;
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(-1, 1);
        std::generate(input, input + N, [&]() { return dis(gen); });
#ifdef CHECK_ACCURACY
    }
#endif

#ifdef TRITON_KERNEL_ENABLE
    // run triton kernel

    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();

    int  grid = ceil((float)N / dropout_kernel_BLOCK_SIZE);

    for (int i = 0; i < RUN_COUNT; i++)
    {
        dropout_kernel_omp(grid, 1, 1, &dropout_kernel, input, real_out, N, ratio, seed);
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

    PRINT_KERNEL_RUNNING_TIME(TRITON_KERNEL, std::chrono::duration<double>(end - begin).count())

#endif

#ifdef C_KERNEL_ENABLE
    // run c++ kernel

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
    int count_c = 0;
    for (int i = 0; i < N; i++)
    {
        if (real_out[i] == 0)
            count_c++;
    }
    printf("C++ Dropout ratio: %.3f\n", (float)count_c / N);

    PRINT_KERNEL_RUNNING_TIME(C_KERNEL, std::chrono::duration<double>(end_c - begin_c).count())

#endif

    return 0;
}