import torch
import triton
import triton.language as tl
import os

USE_GPU = False
triton.runtime.driver.set_active_to_cpu()

def get_warp_kernel_autotune_config():
    configs = [
        triton.Config({'BLOCK_SIZE_W': 1}),
        triton.Config({'BLOCK_SIZE_W': 4}),
        triton.Config({'BLOCK_SIZE_W': 8}),
        triton.Config({'BLOCK_SIZE_W': 16}),
        triton.Config({'BLOCK_SIZE_W': 32}),
        triton.Config({'BLOCK_SIZE_W': 64}),
        triton.Config({'BLOCK_SIZE_W': 128}),
        triton.Config({'BLOCK_SIZE_W': 256}),
    ]
    if(os.getenv("ENABLE_AUTOTUNING") == "warp_kernel"):
        assert (len(configs) > 1), "Autotuning config size need be larger than 1"
        return configs

    return [triton.Config({'BLOCK_SIZE_W': 32})]

@triton.autotune(
    configs=get_warp_kernel_autotune_config(),
    key=[],
)

@triton.jit
def warp_kernel(
    src_ptr,        # *int8, shape [C, H, W]
    offset_ptr,     # *int16, shape [H, W]
    out_ptr,        # *int8, shape [C, H, W]
    channel,        # int
    height,         # int
    width,          # int
    BLOCK_SIZE_W: tl.constexpr,
):

    pid_h = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)

    # Compute the indices
    h_idx = pid_h

    for off in range(0, width, BLOCK_SIZE_W):
        w_idx = off + tl.arange(0, BLOCK_SIZE_W)
        # Create a mask to avoid out-of-bounds accesses
        mask = (w_idx < width)

        # Compute offset indices
        offset_idx = h_idx * width + w_idx  # [BLOCK_SIZE_H, BLOCK_SIZE_W]

        # Load offset values
        offset_val = tl.load(offset_ptr + offset_idx, mask=mask, other=0).to(tl.int16)

        # Decompose offset_val into integer and fractional parts
        offset_int = (offset_val >> 8).to(tl.int8)
        offset_fraction = ((offset_val << 8) >> 8).to(tl.int8)

        # Compute indvar (w_idx)
        indvar = w_idx.to(tl.int8)

        # Compute right_idx and left_idx
        right_idx = (indvar - offset_int).to(tl.int8)
        left_idx = (right_idx - 1).to(tl.int8)

        # Compute src indices
        src_base = pid_c * height * width + h_idx * width  # [BLOCK_SIZE_H, 1]
        right_src_idx = src_base + right_idx  # [BLOCK_SIZE_H, BLOCK_SIZE_W]
        left_src_idx = src_base + left_idx

        # Load values
        right_val = tl.load(src_ptr + right_src_idx, mask=mask, other=0).to(tl.int8)
        left_val = tl.load(src_ptr + left_src_idx, mask=mask, other=0).to(tl.int8)

        right_val = tl.where(right_idx < 0, 0, right_val)
        left_val = tl.where(left_idx < 0, 0, left_val)

        # Compute output
        out = (right_val.to(tl.int16) << 8)
        out += (left_val - right_val).to(tl.int16) * offset_fraction.to(tl.int16)
        out = (out >> 8).to(tl.int8)

        # Compute output indices
        out_idx = pid_c * height * width + h_idx * width + w_idx

        # Store the result
        tl.store(out_ptr + out_idx, out, mask=mask)

def warp(src_arr, offset_arr, out_arr):
    src_arr = src_arr.contiguous()
    offset_arr = offset_arr.contiguous()
    out_arr = out_arr.contiguous()

    # Get dimensions
    channel, height, width = src_arr.shape

    # Compute grid dimensions
    grid = lambda meta: (height, channel, 1)

    # Launch the Triton kernel
    warp_kernel[grid](
        src_arr, offset_arr, out_arr, channel, height, width
    )

# Example usage:
C, H, W = 3, 512, 512
src = torch.ones((C, H, W), dtype=torch.int8, device='cpu')
offset = torch.zeros((H, W), dtype=torch.int16, device='cpu')  # Example offset values
out = torch.empty((C, H, W), dtype=torch.int8, device='cpu')

warp(src, offset, out)

# Now `out` contains the warped image.
