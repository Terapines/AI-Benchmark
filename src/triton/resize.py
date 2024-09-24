import torch

import triton
import triton.language as tl

import os

USE_GPU = False
triton.runtime.driver.set_active_to_cpu()

def get_resize_kernel_autotune_config():
    configs = [
        triton.Config({'BLOCK_SIZE_H': 1, 'BLOCK_SIZE_W': 2}),
        triton.Config({'BLOCK_SIZE_H': 1, 'BLOCK_SIZE_W': 4}),
        triton.Config({'BLOCK_SIZE_H': 1, 'BLOCK_SIZE_W': 8}),
        triton.Config({'BLOCK_SIZE_H': 1, 'BLOCK_SIZE_W': 16}),
        triton.Config({'BLOCK_SIZE_H': 1, 'BLOCK_SIZE_W': 32}),
        triton.Config({'BLOCK_SIZE_H': 1, 'BLOCK_SIZE_W': 64}),
        triton.Config({'BLOCK_SIZE_H': 2, 'BLOCK_SIZE_W': 2}),
        triton.Config({'BLOCK_SIZE_H': 2, 'BLOCK_SIZE_W': 4}),
        triton.Config({'BLOCK_SIZE_H': 2, 'BLOCK_SIZE_W': 8}),
        triton.Config({'BLOCK_SIZE_H': 2, 'BLOCK_SIZE_W': 16}),
        triton.Config({'BLOCK_SIZE_H': 2, 'BLOCK_SIZE_W': 32}),
        triton.Config({'BLOCK_SIZE_H': 2, 'BLOCK_SIZE_W': 64}),
        triton.Config({'BLOCK_SIZE_H': 4, 'BLOCK_SIZE_W': 64}),
        triton.Config({'BLOCK_SIZE_H': 4, 'BLOCK_SIZE_W': 2}),
        triton.Config({'BLOCK_SIZE_H': 4, 'BLOCK_SIZE_W': 4}),
        triton.Config({'BLOCK_SIZE_H': 4, 'BLOCK_SIZE_W': 8}),
        triton.Config({'BLOCK_SIZE_H': 4, 'BLOCK_SIZE_W': 16}),
        triton.Config({'BLOCK_SIZE_H': 4, 'BLOCK_SIZE_W': 32}),
        triton.Config({'BLOCK_SIZE_H': 8, 'BLOCK_SIZE_W': 8}),
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 16}),
        triton.Config({'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_W': 32}),
    ]
    if(os.getenv("ENABLE_AUTOTUNING") == "resize_kernel"):
      assert (len(configs) > 1), "Autotuning config size need be larger than 1"
      return configs

    return [configs[0]]

@triton.autotune(
    configs=get_resize_kernel_autotune_config(),
    key=[],
)

@triton.jit
def resize_kernel(
    src_ptr,
    out_ptr,
    channel,
    height,
    width,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid_h = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)
    pid_c = tl.program_id(axis=2)

    dst_height = 2 * height  # 2x upsample
    dst_width = 2 * width

    hw_fl = 7
    
    h_idx = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)[:, None]  # [BLOCK_SIZE_H, 1]
    w_idx = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)[None, :]  # [1, BLOCK_SIZE_W]

    mask = (h_idx < dst_height) & (w_idx < dst_width) & (pid_c < channel)

    input_y = h_idx << (hw_fl - 1)
    input_x = w_idx << (hw_fl - 1)

    y0 = input_y >> hw_fl
    x0 = input_x >> hw_fl

    y1 = tl.minimum(y0 + 1, height - 1)
    x1 = tl.minimum(x0 + 1, width - 1)

    factor = 1 << hw_fl

    w1_lambda = input_x - (x0 << hw_fl)
    w0_lambda = factor - w1_lambda
    h1_lambda = input_y - (y0 << hw_fl)
    h0_lambda = factor - h1_lambda

    src_offset = pid_c * height * width
    y0_offset = src_offset + y0 * width
    y1_offset = src_offset + y1 * width

    y0x0_offset = y0_offset + x0
    y0x1_offset = y0_offset + x1
    y1x0_offset = y1_offset + x0
    y1x1_offset = y1_offset + x1

    y0x0 = tl.load(src_ptr + y0x0_offset, mask=mask, other=0).to(tl.int16)
    y0x1 = tl.load(src_ptr + y0x1_offset, mask=mask, other=0).to(tl.int16)
    y1x0 = tl.load(src_ptr + y1x0_offset, mask=mask, other=0).to(tl.int16)
    y1x1 = tl.load(src_ptr + y1x1_offset, mask=mask, other=0).to(tl.int16)

    sum1 = (y0x0 * w0_lambda + y0x1 * w1_lambda) >> hw_fl
    sum2 = (y1x0 * w0_lambda + y1x1 * w1_lambda) >> hw_fl
    sum = (sum1 * h0_lambda + sum2 * h1_lambda) >> hw_fl

    sum = sum.to(tl.int8)

    tl.store(out_ptr + (pid_c * dst_height * dst_width + h_idx * dst_width + w_idx), sum, mask=mask)


def resize(src_arr, out_arr):
    src_arr = src_arr.contiguous()
    out_arr = out_arr.contiguous()

    # Get dimensions
    channel, height, width = src_arr.shape

    # BLOCK_H = 32
    # BLOCK_W = 32

    # Compute grid dimensions
    grid = lambda meta: (triton.cdiv(height * 2, meta['BLOCK_SIZE_H']), triton.cdiv(width * 2, meta['BLOCK_SIZE_W']), channel)

    # Launch the Triton kernel
    resize_kernel[grid](
        src_arr, out_arr, channel, height, width
    )

C, H, W = 3, 512, 512
src = torch.ones((C, H, W), dtype=torch.int8, device='cpu')
out = torch.empty((C, 2 * H, 2 * W), dtype=torch.int8, device='cpu')

resize(src, out)

# print(src)
# print(out)




