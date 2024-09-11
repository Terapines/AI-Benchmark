void matmul(float *arg0, float *arg1, float *arg2, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      arg2[i * N + j] = 0;
      for (int k = 0; k < K; k++) {
        arg2[i * N + j] += arg0[i * K + k] * arg1[k * N + j];
      }
    }
  }
}
