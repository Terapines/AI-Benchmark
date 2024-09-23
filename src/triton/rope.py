import torch
import triton
import triton.language as tl
from typing import Tuple, Union
import os

USE_GPU = False
triton.runtime.driver.set_active_to_cpu()

def get_rope_kernel_autotune_config():
    configs = [
        triton.Config({'BLOCK_SIZE': 16}),
        triton.Config({'BLOCK_SIZE': 4}),
        triton.Config({'BLOCK_SIZE': 2}),
        triton.Config({'BLOCK_SIZE': 8}),
        triton.Config({'BLOCK_SIZE': 32})
    ]
    if(os.getenv("ENABLE_AUTOTUNING") == "rope_kernel"):
      assert (len(configs) > 1), "Autotuning config size need be larger than 1"
      return configs

    return [configs[0]]

@triton.autotune(
    configs=get_rope_kernel_autotune_config(),
    key=[],
)
@triton.jit
def rope_kernel_fw(input_ptr, # [seq_len, batch_num, head_num, head_dim]
                   in_seq_len_stride, 
                   in_batch_stride,
                   output_ptr, 
                   cos_ptr, # [seq_len, head_dim]
                   sin_ptr, # [seq_len, head_dim]
                   cos_stride, 
                   sin_stride,
                   seq_len, 
                   head_dim,
                   BLOCK_SIZE: tl.constexpr):
    
    pid_seq = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)

    head_dim_mid = head_dim // 2

    for off in range(0, head_dim_mid, BLOCK_SIZE):
        head_dim_offset = off + tl.arange(0, BLOCK_SIZE)  # [0:head_dim/2]

        mask = head_dim_offset < head_dim_mid

        cos_offset = (pid_seq % seq_len) * cos_stride + head_dim_offset
        sin_offset = (pid_seq % seq_len) * sin_stride + head_dim_offset

        cos = tl.load(cos_ptr + cos_offset, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + sin_offset, mask=mask, other=0.0)

        x1_offset = pid_seq * in_seq_len_stride + pid_batch * \
            in_batch_stride + pid_head * head_dim + head_dim_offset
        x2_offset = pid_seq * in_seq_len_stride + pid_batch * in_batch_stride + \
            pid_head * head_dim + head_dim_mid + head_dim_offset

        x1 = tl.load(input_ptr + x1_offset, mask=mask, other=0.0)
        x2 = tl.load(input_ptr + x2_offset, mask=mask, other=0.0)

        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        tl.store(output_ptr + x1_offset, y1, mask=mask)
        tl.store(output_ptr + x2_offset, y2, mask=mask)

    return

@triton.autotune(
    configs=get_rope_kernel_autotune_config(),
    key=[],
)
@triton.jit
def rope_kernel_bw(input_ptr, 
                   in_seq_len_stride, 
                   in_batch_stride,
                   output_ptr, 
                   cos_ptr, 
                   sin_ptr, 
                   cos_stride, 
                   sin_stride,
                   seq_len, 
                   head_dim,
                   BLOCK_SIZE: tl.constexpr):
    
    pid_seq = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)

    head_dim_mid = head_dim // 2

    for off in range(0, head_dim_mid, BLOCK_SIZE):
        head_dim_offset = off + tl.arange(0, BLOCK_SIZE)  # [0:head_dim/2]

        mask = head_dim_offset < head_dim_mid

        cos_offset = (pid_seq % seq_len) * cos_stride + head_dim_offset
        sin_offset = (pid_seq % seq_len) * sin_stride + head_dim_offset

        cos = tl.load(cos_ptr + cos_offset, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + sin_offset, mask=mask, other=0.0)

        x1_offset = pid_seq * in_seq_len_stride + pid_batch * \
            in_batch_stride + pid_head * head_dim + head_dim_offset
        x2_offset = pid_seq * in_seq_len_stride + pid_batch * in_batch_stride + \
            pid_head * head_dim + head_dim_mid + head_dim_offset

        x1 = tl.load(input_ptr + x1_offset, mask=mask, other=0.0)
        x2 = tl.load(input_ptr + x2_offset, mask=mask, other=0.0)

        y1 = x1 * cos - x2 * -sin
        y2 = x1 * -sin + x2 * cos

        tl.store(output_ptr + x1_offset, y1, mask=mask)
        tl.store(output_ptr + x2_offset, y2, mask=mask)
    
    return


class FusedRoPEFucnTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor, # [seq_len, batch_num, head_num, head_dim]
        freqs: torch.Tensor, # [seq_len, head_dim]
        tensor_format: str = "sbhd",
        cu_seqlens: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if tensor_format == "bshd":
            t = t.transpose(0, 1)
        elif tensor_format != "sbhd":
            raise ValueError(f"Unsupported tensor_format: {tensor_format}.")

        seq_len, batch_num, head_num, head_dim = t.shape
        output = torch.empty_like(t)

        # BLOCK_SIZE = 16

        grid = (seq_len, head_num, batch_num)

        freqs = freqs[:seq_len]
        cos = torch.cos(freqs).to(t.dtype)
        sin = torch.sin(freqs).to(t.dtype)

        rope_kernel_fw[grid](t,
                             t.stride(0),
                             t.stride(1),
                             output,
                             cos,
                             sin,
                             cos.stride(0),
                             sin.stride(0),
                             seq_len,
                             head_dim)

        ctx.cos = cos
        ctx.sin = sin
        # ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.tensor_format = tensor_format

        if tensor_format == "bshd":
            return output.transpose(0, 1)
        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        if ctx.tensor_format == "bshd":
            grad_output = grad_output.transpose(0, 1)
        elif ctx.tensor_format != "sbhd":
            raise ValueError(
                f"Unsupported tensor_format: {ctx.tensor_format}.")

        seq_len, batch_num, head_num, head_dim = grad_output.shape
        grad_input = torch.empty_like(grad_output)

        grid = (seq_len, head_num, batch_num)

        rope_kernel_bw[grid](grad_output.clone(),
                             grad_input.stride(0),
                             grad_input.stride(1),
                             grad_input,
                             ctx.cos,
                             ctx.sin,
                             ctx.cos.stride(0),
                             ctx.sin.stride(0),
                             seq_len,
                             head_dim)

        if ctx.tensor_format == "bshd":
            return grad_input.transpose(0, 1), None, None, None, None

        return grad_input, None, None, None, None


# def apply_rotary_pos_emb(
#     t: torch.Tensor,
#     freqs: torch.Tensor,
#     tensor_format: str = "sbhd",
#     fused: bool = False,
#     cu_seqlens: Union[torch.Tensor, None] = None,
# ) -> torch.Tensor:
#     if fused:
#         return FusedRoPEFucnTriton.apply(t, freqs, tensor_format, cu_seqlens)
#     else:
#         "Only fused option is supported"

rope_triton = FusedRoPEFucnTriton.apply
device = 'cpu'
dtype = torch.float32 if device == 'cpu' else torch.float16

# NOTE: this is generated by chatgpt
def rope_pytorch(t, freqs):
    """
    Applies Rotary Positional Embeddings (RoPE) to the tensor `t` with given `freqs`.
    
    Args:
        t (torch.Tensor): The input tensor with shape [seq_len, batch_num, head_num, head_dim].
        freqs (torch.Tensor): The precomputed frequencies for RoPE with shape [seq_len, head_dim].
    
    Returns:
        torch.Tensor: The output tensor with RoPE applied.
    """
    seq_len, batch_num, head_num, head_dim = t.shape
    output = torch.empty_like(t)
    
    half_dim = head_dim // 2

    freqs = freqs[:seq_len]
    cos = torch.cos(freqs).to(t.dtype).unsqueeze(1).unsqueeze(1)
    sin = torch.sin(freqs).to(t.dtype).unsqueeze(1).unsqueeze(1)

    t_1, t_2 = t[..., :half_dim], t[..., half_dim:]
    y1 = t_1 * cos - t_2 * sin
    y2 = t_1 * sin + t_2 * cos

    output[..., :half_dim] = y1
    output[..., half_dim:] = y2

    return output

def rotary_pos_emb(dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # [dim // 2]
    t = torch.arange(seq_len, dtype=freqs.dtype, device=freqs.device) # [seq_len]
    freqs = torch.outer(t, freqs).float() # [seq_len, dim // 2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis # [seq_len, dim]


def test_rope_with_pytorch(seq_len=128, batch_num=16, head_num=12, head_dim=64, theta=10000.0):
    t = torch.randn(seq_len, batch_num, head_num, head_dim, device=device, dtype=dtype)
    freqs = rotary_pos_emb(head_dim, seq_len, theta=theta)
    t_ = t.clone()
    freqs_ = freqs.clone()

    t.requires_grad = True
    freqs.requires_grad = True
    t_.requires_grad = True
    freqs_.requires_grad = True

    out = rope_triton(t, freqs)
    out.sum().backward()

    out_pytorch = rope_pytorch(t_, freqs_)
    out_pytorch.sum().backward()

    # compare the results
    # print("triton: ", out[0, 0, 0, :10])
    # print("pytorch: ", out_pytorch[0, 0, 0, :10])

    # assert(torch.allclose(out, out_pytorch, atol=1e-4))
    # assert(torch.allclose(t.grad, t_.grad, atol=1e-4))

def test_rope(seq_len=128, batch_num=16, head_num=12, head_dim=64):
    t = torch.randn(seq_len, batch_num, head_num, head_dim, device=device, dtype=dtype)
    freqs = torch.randn(seq_len, head_dim, device=device, dtype=dtype)
    t.requires_grad = True
    freqs.requires_grad = True

    out = rope_triton(t, freqs)
    out.sum().backward()
    return out

# NOTE: the `batch_num` should be identical to C kernel, its a constexpr
test_rope_with_pytorch(seq_len=128, batch_num=16, head_num=12, head_dim=64, theta=10000.0)
test_rope(seq_len=128, batch_num=16, head_num=12, head_dim=64)