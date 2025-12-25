#include <omp.h>
#include <stdio.h>

int main() {
#pragma omp parallel for
  for (int i = 0; i < 8; i++) {
    printf("i=%d thread=%d\n", i, omp_get_thread_num());
  }
}
