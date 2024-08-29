import torch

import triton
import triton.language as tl


@triton.jit
def correlation_kernel(
        # Pointers to matrices
        src0_ptr, src1_ptr, out_ptr,
        # Matrix dimensions
        out_channel, in_channel, height, width, hw,
        # Normalize
        out_shift,
        # Meta-parameters
        BLOCK_SIZE_OC: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr, BLOCK_SIZE_IC: tl.constexpr
):

    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.

    height_idx = tl.arange(0, BLOCK_SIZE_H)
    width_idx = tl.arange(0, BLOCK_SIZE_W)


    # Create a mask to guard memory operations against out-of-bounds accesses.
    bound_mask = ((height_idx[:, None] < height) & (width_idx[None, :] < width)) & (width_idx[None, :] >= pid)
    # channel_mask = width_idx[None, :] >= pid

    offsets = (height_idx[:, None] * width) + width_idx[None, :]

    sum_data = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.int16)

    src0_ptrs = src0_ptr + offsets
    src1_ptrs = src1_ptr + offsets

    for k in range(in_channel):

        # src0_val = tl.load(src0_ptr + in_idx1)
        src0_val = tl.load(src0_ptrs, mask=bound_mask, other=0)
        # src0_val = tl.where(channel_mask, src0_val, 0).to(tl.int8)

        # src1_val = tl.load(src1_ptr + in_idx2)
        src1_val = tl.load(src1_ptrs - pid, mask=bound_mask, other=0)
        # src1_val = tl.where(channel_mask, src1_val, 0).to(tl.int8)

        sum_data += src0_val * src1_val

        src0_ptrs += hw
        src1_ptrs += hw

    out_idx = pid * hw + offsets
    out_val = (sum_data >> out_shift).to(tl.int8)
    tl.store(out_ptr + out_idx, out_val, mask=bound_mask)


def correlation(src0_arr, src1_arr, out_arr, out_shift):
    out_channel = out_arr.shape[0]

    # Define grid size
    in_channel, height, width = src0_arr.shape


    grid = (out_channel,)
    block_ic = triton.next_power_of_2(in_channel)
    block_oc = triton.next_power_of_2(out_channel)

    # Launch the kernel
    correlation_kernel[grid](
        src0_arr, src1_arr, out_arr,
        out_channel, in_channel, height, width, height* width, out_shift,
        BLOCK_SIZE_OC=block_oc, BLOCK_SIZE_H = 128, BLOCK_SIZE_W = 128, BLOCK_SIZE_IC = block_ic
    )

# %%
# Unit Test
# ---------
#
triton.runtime.driver.set_active_to_cpu()

IN_C = 58
OUT_C = 5
H = 112
W = 88

RUN_COUNT=100
IN_SIZE = IN_C * H * W
OUT_SIZE = OUT_C * H * W

src0_arr_global = torch.ones((IN_SIZE), dtype=torch.int8, device='cpu')
src1_arr_global = torch.ones((IN_SIZE), dtype=torch.int8, device='cpu')
out_arr_global = torch.zeros((OUT_C, H, W), dtype=torch.int8, device='cpu')

for i in range(IN_SIZE):
    src0_arr_global[i] = i % 16
    src1_arr_global[i] = i % 35

src0_arr_global = torch.reshape(src0_arr_global, (IN_C, H, W))
src1_arr_global = torch.reshape(src1_arr_global, (IN_C, H, W))


correlation(src0_arr_global, src1_arr_global, out_arr_global, 0)

output = torch.flatten(out_arr_global)
