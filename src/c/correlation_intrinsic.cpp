// #include "kernel.h"
// #include <riscv_vector.h>

// __attribute__((noinline)) void correlation(struct correlation_param *param) {
//   size_t in_channel = param->in_channel;
//   size_t height = param->height;
//   size_t width = param->width;
//   size_t out_channel = param->out_channel;
//   size_t out_shift = param->out_shift;
//   int8_t *src0_arr = param->src0_ptr;
//   int8_t *src1_arr = param->src1_ptr;
//   int8_t *out_arr = param->out_ptr;

//   int8_t *src0_ptr = src0_arr;
//   int8_t *src1_ptr = src1_arr;
//   int8_t *out_ptr = out_arr;

//   size_t fm_size = height * width;
//   size_t vl = 0;
//   size_t vl0 = 0;
//   size_t vl_cnt = 0;

//   for (size_t w = width; (vl = __riscv_vsetvl_e8m2(w)); w -= vl) {
//     for (size_t h = 0; h < height; h++) {
//       for (size_t d = 0; d < out_channel; d++) {
//         src0_ptr = h * width + vl_cnt + d + src0_arr;
//         src1_ptr = h * width + vl_cnt + src1_arr;
//         out_ptr = h * width + vl_cnt + d * fm_size + out_arr;
//         vl0 = __riscv_vsetvl_e8m2(w - d);
//         vint16m4_t acc = __riscv_vmv_v_x_i16m4(0, vl0);
//         vint8m2_t vx, vy;
//         for (size_t c = 0; c < in_channel; c++) {
//           vx = __riscv_vle8_v_i8m2(src0_ptr, vl0);
//           vy = __riscv_vle8_v_i8m2(src1_ptr, vl0);
//           acc = __riscv_vwmacc_vv_i16m4(acc, vy, vx, vl0);
//           src0_ptr += fm_size;
//           src1_ptr += fm_size;
//         }
//         vint8m2_t acc_sra = __riscv_vnsra_wx_i8m2(acc, out_shift, vl0);
//         __riscv_vse8_v_i8m2(out_ptr + d, acc_sra, vl0);
//       }
//     }
//     vl_cnt += vl;
//   }
// }
