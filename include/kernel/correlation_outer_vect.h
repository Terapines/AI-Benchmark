#ifndef __CORRELATION_KERNEL_H__
#define __CORRELATION_KERNEL_H__

#include <stdint.h>
#include <cstddef>

void correlation(const int8_t *__restrict__ src0_arr,
                 const int8_t *__restrict__ src1_arr,
                 int8_t *__restrict__ out_arr,
                 size_t in_channel,
                 size_t height,
                 size_t width,
                 size_t out_channel,
                 size_t out_shift);

#endif