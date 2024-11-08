#ifdef C_KERNEL_ENABLE
#include "kernel/rope.h"
#endif

#ifdef TRITON_KERNEL_ENABLE
#include "rope_kernel_fw_launcher.h"
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
    int SEQ_LEN = 128;
    int BATCH_NUM = 32;
    int HEAD_NUM = 12;
    int HEAD_DIM = 64;
    int RUN_COUNT = 10;
    float theta = 10000.0;

    // parse args
    if (argc >= 2)
    {
        std::vector<int> Shape = splitStringToInts(argv[1]);
        if (Shape.size())
        {
            assert(Shape.size() == 5 && "Invalid shape format: SEQ_LENxBATCH_NUMxHEAD_NUMxHEAD_DIMxRUN_COUNT\n");
            SEQ_LEN = Shape.at(0);
            BATCH_NUM = Shape.at(1);
            HEAD_NUM = Shape.at(2);
            HEAD_DIM = Shape.at(3);
            RUN_COUNT = Shape.at(4);
        }
    }

    printf("Data shape %dx%dx%dx%dx%d\n", SEQ_LEN, BATCH_NUM, HEAD_NUM, HEAD_DIM, RUN_COUNT);

    assert(SEQ_LEN != 0 && BATCH_NUM != 0 && HEAD_NUM != 0 && HEAD_DIM != 0 && "Invalid shape\n");

    // Initialize input tensors
    float *t = (float *)malloc(SEQ_LEN * BATCH_NUM * HEAD_NUM * HEAD_DIM * sizeof(float));
    float *freq_cos = (float *)malloc(SEQ_LEN * HEAD_DIM * sizeof(float));
    float *freq_sin = (float *)malloc(SEQ_LEN * HEAD_DIM * sizeof(float));

    float *real_out = (float *)malloc(SEQ_LEN * BATCH_NUM * HEAD_NUM * HEAD_DIM * sizeof(float));
    float *ref_out = (float *)malloc(SEQ_LEN * BATCH_NUM * HEAD_NUM * HEAD_DIM * sizeof(float));

    memset(real_out, 0, SEQ_LEN * BATCH_NUM * HEAD_NUM * HEAD_DIM * sizeof(float));

#ifdef CHECK_ACCURACY
    std::string DB = getDB(argv[1]);
    FILE *file = fopen(DB.c_str(), "rb");
    if (file)
    {
        printf("File %s open for read\n", DB.c_str());

        fread(t, sizeof(float), SEQ_LEN * BATCH_NUM * HEAD_NUM * HEAD_DIM, file);
        fread(freq_cos, sizeof(float), SEQ_LEN * HEAD_DIM, file);
        fread(freq_sin, sizeof(float), SEQ_LEN * HEAD_DIM, file);
        fread(ref_out, sizeof(float), SEQ_LEN * BATCH_NUM * HEAD_NUM * HEAD_DIM, file);
    }
    else
    {
#endif
        // Will be used to obtain a seed for the random number engine
        std::random_device rd;
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(-1, 1);
        std::generate(t, t + SEQ_LEN * BATCH_NUM * HEAD_NUM * HEAD_DIM, [&]() { return dis(gen); });
        std::generate(freq_cos, freq_cos + SEQ_LEN * HEAD_DIM, [&]() { return dis(gen); });
        std::generate(freq_sin, freq_sin + SEQ_LEN * HEAD_DIM, [&]() { return dis(gen); });
#ifdef CHECK_ACCURACY
    }
#endif

#ifdef TRITON_KERNEL_ENABLE
    // run triton kernel

    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUN_COUNT; i++)
    {
        rope_kernel_fw_omp(HEAD_NUM, BATCH_NUM, SEQ_LEN, &rope_kernel_fw, t, BATCH_NUM * HEAD_NUM * HEAD_DIM, HEAD_NUM * HEAD_DIM, real_out, freq_cos, freq_sin, HEAD_DIM, HEAD_DIM, SEQ_LEN, HEAD_DIM);
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::milliseconds time_interval = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

    PRINT_KERNEL_RUNNING_TIME(TRITON_KERNEL, std::chrono::duration<double>(end - begin).count())

#endif

#ifdef C_KERNEL_ENABLE
    // run c++ kernel

    std::chrono::high_resolution_clock::time_point begin_c = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUN_COUNT; i++)
    {
        rope(t, freq_cos, freq_sin, real_out, SEQ_LEN, BATCH_NUM, HEAD_NUM, HEAD_DIM);
    }
    std::chrono::high_resolution_clock::time_point end_c = std::chrono::high_resolution_clock::now();
    std::chrono::milliseconds time_interval_c = std::chrono::duration_cast<std::chrono::milliseconds>(end_c - begin_c);

    PRINT_KERNEL_RUNNING_TIME(C_KERNEL, std::chrono::duration<double>(end_c - begin_c).count())

#endif

#ifdef CHECK_ACCURACY
    if (file == nullptr)
    {
        file = fopen(DB.c_str(), "wb");

        printf("File %s open for write\n", DB.c_str());
        assert(file);

        fwrite(t, sizeof(float), SEQ_LEN * BATCH_NUM * HEAD_NUM * HEAD_DIM, file);
        fwrite(freq_cos, sizeof(float), SEQ_LEN * HEAD_DIM, file);
        fwrite(freq_sin, sizeof(float), SEQ_LEN * HEAD_DIM, file);

        memcpy(ref_out, real_out, SEQ_LEN * BATCH_NUM * HEAD_NUM * HEAD_DIM * sizeof(float));
        fwrite(ref_out, sizeof(float), SEQ_LEN * BATCH_NUM * HEAD_NUM * HEAD_DIM, file);
    }
    fclose(file);

    // test the correctness of the kernel
    check_tensor(ref_out, real_out, SEQ_LEN * BATCH_NUM * HEAD_NUM * HEAD_DIM, "out");
#endif

    // free memory
    free(t);
    free(freq_cos);
    free(freq_sin);
    free(ref_out);
    free(real_out);

    return 0;
}