#ifdef C_KERNEL_ENABLE
#include "kernel/warp.h"
#endif

#ifdef TRITON_KERNEL_ENABLE
#include "warp_kernel_launcher.h"
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
    int H = 128;
    int W = 128;
    int C = 3;
    int RUN_COUNT = 10;

    if (argc >= 2)
    {
        std::vector<int> Shape = splitStringToInts(argv[1]);
        if (Shape.size())
        {
            assert(Shape.size() == 4 && "Invalid shape format: HxWxCxRUN_COUNT\n");
            H = Shape.at(0);
            W = Shape.at(1);
            C = Shape.at(2);
            RUN_COUNT = Shape.at(3);
        }
    }

    printf("Data shape %dx%dx%d\n", H, W, C);

    assert(H != 0 && W != 0 && C != 0 && "Invalid shape\n");

    int8_t *input = (int8_t *)malloc(H * W * C * sizeof(int8_t));
    int16_t *offset = (int16_t *)malloc(H * W * sizeof(int16_t));

    int8_t *real_out = (int8_t *)malloc(H * W * C * sizeof(int8_t));
    int8_t *ref_out = (int8_t *)malloc(H * W * C * sizeof(int8_t));

    memset(real_out, 0, H * W * C * sizeof(int8_t));

    std::string DB = getDB(argv[1]);

#ifdef CHECK_ACCURACY
    FILE *file = fopen(DB.c_str(), "rb");
    if (file)
    {
        printf("File %s open for read\n", DB.c_str());

        fread(input, sizeof(int8_t), H * W * C, file);
        fread(offset, sizeof(int16_t), H * W, file);
        fread(ref_out, sizeof(int8_t), H * W * C, file);
    }
    else
    {
#endif
        // Initialize random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis_int8(-128, 127);
        std::uniform_int_distribution<> dis_int16(-32768, 32767);

        // Generate random input data
        std::generate(input, input + H * W * C, [&]() { return dis_int8(gen); });
        std::generate(offset, offset + H * W, [&]() { return dis_int16(gen); });
#ifdef CHECK_ACCURACY
    }
#endif

    printf("Input:\n");
    for (int i = 0; i < std::min(16, H * W * C); i++)
    {
        printf("%hhd  ", input[i]);
        if (i == std::min(16, H * W * C) - 1)
            printf("...\n");
    }

#ifdef TRITON_KERNEL_ENABLE
    // Run Triton kernel
    printf("Start running Triton kernel %d times.\n", RUN_COUNT);

    int gridX = ceil((float)H / warp_kernel_BLOCK_SIZE_H);
    int gridY = ceil((float)W / warp_kernel_BLOCK_SIZE_W);
    int gridZ = C;

    auto begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUN_COUNT; i++)
    {
        warp_kernel_omp(gridX, gridY, gridZ, &warp_kernel, input, offset, real_out, C, H, W);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time_interval = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

    printf("Triton output:\n");
    for (int i = 0; i < std::min(16, H * W * C); i++)
    {
        printf("%hhd  ", real_out[i]);
        if (i == std::min(16, H * W * C) - 1)
            printf("...\n");
    }

    printf("Triton kernel running time: %d ms\n", (int)time_interval.count());
    PRINT_KERNEL_RUNNING_TIME(TRITON_KERNEL, std::chrono::duration<double>(end - begin).count())

    // NOTE: The GFLOPS calculation is approximate
    printf("Triton kernel: %f GFLOPS\n", (H * W * C * RUN_COUNT) / (time_interval.count() / 1000.0) / 1e9);
#endif

#ifdef C_KERNEL_ENABLE
    // Run C++ kernel
    printf("Start running C++ kernel %d times.\n", RUN_COUNT);

    auto begin_c = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUN_COUNT; i++)
    {
        warp(input, offset, real_out, C, H, W);
    }
    auto end_c = std::chrono::high_resolution_clock::now();
    auto time_interval_c = std::chrono::duration_cast<std::chrono::milliseconds>(end_c - begin_c);

    printf("C++ output:\n");
    for (int i = 0; i < std::min(16, H * W * C); i++)
    {
        printf("%hhd  ", real_out[i]);
        if (i == std::min(16, H * W * C) - 1)
            printf("...\n");
    }

    printf("C++ kernel running time: %d ms\n", (int)time_interval_c.count());
    PRINT_KERNEL_RUNNING_TIME(C_KERNEL, std::chrono::duration<double>(end_c - begin_c).count())

    printf("C++ kernel: %f GFLOPS\n", (H * W * C * RUN_COUNT) / (time_interval_c.count() / 1000.0) / 1e9);
#endif

#ifdef CHECK_ACCURACY
    // Check correctness of output
    if (file == nullptr) {
        file = fopen(DB.c_str(), "wb");
        printf("File %s open for write\n", DB.c_str());
        assert(file);
        memcpy(ref_out, real_out, H * W * C * sizeof(int8_t));

        fwrite(input, sizeof(int8_t), H * W * C, file);
        fwrite(offset, sizeof(int16_t), H * W, file);
        fwrite(ref_out, sizeof(int8_t), H * W * C, file);
    }
    fclose(file);

    check_tensor(ref_out, real_out, H * W * C, "out");
#endif

    free(input);
    free(offset);
    free(ref_out);
    free(real_out);

    return 0;
}
