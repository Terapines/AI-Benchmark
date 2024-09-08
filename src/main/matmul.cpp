#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <chrono>
#include <cassert>
#include "support/support.h"
#include "kernel/matmul.h"
#include "matmul_kernel_launcher.h"

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;


static void _launch(int gridX, int gridY, int gridZ, void* arg0, void* arg1, void* arg2, int32_t arg3, int32_t arg4, int32_t arg5, int32_t arg6, int32_t arg7, int32_t arg8, int32_t arg9, int32_t arg10, int32_t arg11) {
    matmul_kernel_omp(gridX, gridY, gridZ, matmul_kernel, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11);
}

int main(int argc, char *argv[]) {
    int cnt = 10;

    int M = 179;
    int N = 321;
    int K = 167;
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_N = 64;

    if (argc >= 2) {
        std::vector<int> Shape = splitStringToInts(argv[1]);
        if (Shape.size()) {
            assert(Shape.size() == 4 && "Invalid shape format: MxNxKxCNT\n");
            M = Shape.at(0);
            N = Shape.at(1);
            K = Shape.at(2);
            cnt = Shape.at(3);
        }
    }

    int count;
    srand(time(0));
    printf("%s %s\n", __DATE__, __TIME__);

    count = M * K;
    float *arg0 = (float *) malloc(count * sizeof(float));
    if (arg0 == NULL){
        printf("arg0 == NULL\n");
        return -1;
    } else {
        for (int i = 0; i < count; i++) {
            arg0[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)) - 0.5;
            if (i < 16 && i < K)
                printf("%.3f  ", arg0[i]);
            if (i == count - 1)
                printf("...\n");
        }
    }

    count = K * N;
    float *arg1 = (float *) malloc(count * sizeof(float));
    if (arg1 == NULL){
        printf("arg1 == NULL\n");
        return -1;
    } else {
        for (int i = 0; i < count; i++) {
            arg1[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)) - 0.5;
            if (i < 16 && i < N)
                printf("%.3f  ", arg1[i]);
            if (i  == count - 1)
                printf("...\n");
        }
    }

    count = M * N;
    float *arg2 = (float *) malloc(count * sizeof(float));
    if (arg2 == NULL){
        printf("arg2 == NULL\n");
        return -1;
    }

    count = M * N;
    float *buf = (float *) malloc(count * sizeof(float));
    if (buf == NULL){
        printf("buf == NULL\n");
        return -1;
    }

    printf("Run kernel %d times.\n", cnt);
    high_resolution_clock::time_point beginTime = high_resolution_clock::now();
    for (int i = 0; i < cnt; i++) {
        _launch(ceil(1.0*M/BLOCK_SIZE_M)*ceil(1.0*N/BLOCK_SIZE_N), 1, 1, arg0, arg1, arg2, M, N, K, K, 1, N, 1, N, 1);
    }
    high_resolution_clock::time_point endTime = high_resolution_clock::now();
    milliseconds timeInterval = std::chrono::duration_cast<milliseconds>(endTime - beginTime);
    cout << "Running Time: " << timeInterval.count()  << " ms" << endl;
    cout << "Triton-cpu kernel: " << M*N*K*cnt/(timeInterval.count()/1000.0)/1e9  << " GFLOPS" << endl;
    fprintf(stderr, "Triton-cpu kernel: %f GFLOPS\n", M*N*K*cnt/(timeInterval.count()/1000.0)/1e9);

    count = M * N;
    for (int i = 0; i < count; i++) {
        if (i < 16 && i < N)
            printf("%.3f  ", arg2[i]);
        if (i == count - 1)
            printf("...\n");
    }

    printf("Run matmul %d times.\n", cnt);

    beginTime = high_resolution_clock::now();
    for (int i = 0; i < cnt; i++) {
        matmul(arg0, arg1, buf, M, N, K);
    }
    endTime = high_resolution_clock::now();

    timeInterval = std::chrono::duration_cast<milliseconds>(endTime - beginTime);
    cout << "Running Time: " << timeInterval.count()  << " ms" << endl;
    cout << "c++ matmul: " << M*N*K*cnt/(timeInterval.count()/1000.0)/1e9  << " GFLOPS" << endl;
    fprintf(stderr, "c++ matmul: %f GFLOPS\n", M*N*K*cnt/(timeInterval.count()/1000.0)/1e9);

    count = M * N;
    for (int i = 0; i < count; i++) {
        if (i < 16 && i < N)
            printf("%.3f  ", buf[i]);
        if (i == count - 1)
            printf("...\n");
    }

    /*count = M * N;
    for (int i = 0; i < count; i++) {
        if (arg2[i] != buf[i]) {
            printf("Error !\n");
            printf("arg2[%d]: %.9f, buf[%d]: %.9f\n", i, arg2[i], i, buf[i]);
            return -1;
        }
    }*/

    //printf("Pass\n");
    return 0;
}
