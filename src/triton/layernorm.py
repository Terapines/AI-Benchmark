"""
Layer Normalization
====================
In this tutorial, you will write a high-performance layer normalization
kernel that runs faster than the PyTorch implementation.

In doing so, you will learn about:

* Implementing backward pass in Triton.

* Implementing parallel reduction in Triton.

"""

# %%
# Motivations
# -----------
#
# The *LayerNorm* operator was first introduced in [BA2016]_ as a way to improve the performance
# of sequential models (e.g., Transformers) or neural networks with small batch size.
# It takes a vector :math:`x` as input and produces a vector :math:`y` of the same shape as output.
# The normalization is performed by subtracting the mean and dividing by the standard deviation of :math:`x`.
# After the normalization, a learnable linear transformation with weights :math:`w` and biases :math:`b` is applied.
# The forward pass can be expressed as follows:
#
# .. math::
#    y = \frac{ x - \text{E}[x] }{ \sqrt{\text{Var}(x) + \epsilon} } * w + b
#
# where :math:`\epsilon` is a small constant added to the denominator for numerical stability.
# Let’s first take a look at the forward pass implementation.

import torch

import triton
import triton.language as tl

import os

USE_GPU = False
triton.runtime.driver.set_active_to_cpu()


def get_layer_norm_fwd_fused_autotune_config():
    configs = [
        triton.Config({'BLOCK_SIZE': 4}),
        triton.Config({'BLOCK_SIZE': 8}),
        triton.Config({'BLOCK_SIZE': 16}),
        triton.Config({'BLOCK_SIZE': 32}),
        triton.Config({'BLOCK_SIZE': 64})
    ]
    if(os.getenv("ENABLE_AUTOTUNING") == "_layer_norm_fwd_fused"):
      assert (len(configs) > 1), "Autotuning config size need be larger than 1"
      return configs

    return [triton.Config({'BLOCK_SIZE': 16})]

@triton.autotune(
    configs=get_layer_norm_fwd_fused_autotune_config(),
    key=[],
)
@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


# %%
# Backward pass
# -------------
#
# The backward pass for the layer normalization operator is a bit more involved than the forward pass.
# Let :math:`\hat{x}` be the normalized inputs :math:`\frac{ x - \text{E}[x] }{ \sqrt{\text{Var}(x) + \epsilon} }` before the linear transformation,
# the Vector-Jacobian Products (VJP) :math:`\nabla_{x}` of :math:`x` are given by:
#
# .. math::
#    \nabla_{x} = \frac{1}{\sigma}\Big( \nabla_{y} \odot w - \underbrace{ \big( \frac{1}{N} \hat{x} \cdot (\nabla_{y} \odot w) \big) }_{c_1} \odot \hat{x} - \underbrace{ \frac{1}{N} \nabla_{y} \cdot w }_{c_2} \Big)
#
# where :math:`\odot` denotes the element-wise multiplication, :math:`\cdot` denotes the dot product, and :math:`\sigma` is the standard deviation.
# :math:`c_1` and :math:`c_2` are intermediate constants that improve the readability of the following implementation.
#
# For the weights :math:`w` and biases :math:`b`, the VJPs :math:`\nabla_{w}` and :math:`\nabla_{b}` are more straightforward:
#
# .. math::
#    \nabla_{w} = \nabla_{y} \odot \hat{x} \quad \text{and} \quad \nabla_{b} = \nabla_{y}
#
# Since the same weights :math:`w` and biases :math:`b` are used for all rows in the same batch, their gradients need to sum up.
# To perform this step efficiently, we use a parallel reduction strategy: each kernel instance accumulates
# partial :math:`\nabla_{w}` and :math:`\nabla_{b}` across certain rows into one of :math:`\text{GROUP_SIZE_M}` independent buffers.
# These buffers stay in the L2 cache and then are further reduced by another function to compute the actual :math:`\nabla_{w}` and :math:`\nabla_{b}`.
#
# Let the number of input rows :math:`M = 4` and :math:`\text{GROUP_SIZE_M} = 2`,
# here's a diagram of the parallel reduction strategy for :math:`\nabla_{w}` (:math:`\nabla_{b}` is omitted for brevity):
#
#   .. image:: parallel_reduction.png
#
# In Stage 1, the rows of X that have the same color share the same buffer and thus a lock is used to ensure that only one kernel instance writes to the buffer at a time.
# In Stage 2, the buffers are further reduced to compute the final :math:`\nabla_{w}` and :math:`\nabla_{b}`.
# In the following implementation, Stage 1 is implemented by the function :code:`_layer_norm_bwd_dx_fused` and Stage 2 is implemented by the function :code:`_layer_norm_bwd_dwdb`.

def get_layer_norm_bwd_fused_autotune_config():
    configs = [
        # triton.Config({'BLOCK_SIZE_N': 1}),
        triton.Config({'BLOCK_SIZE_N': 2}),
        triton.Config({'BLOCK_SIZE_N': 4}),
        triton.Config({'BLOCK_SIZE_N': 8}),
        triton.Config({'BLOCK_SIZE_N': 16}),
        triton.Config({'BLOCK_SIZE_N': 32}),
        triton.Config({'BLOCK_SIZE_N': 64}),
    ]
    if(os.getenv("ENABLE_AUTOTUNING") == "_layer_norm_bwd_fused"):
      assert (len(configs) > 1), "Autotuning config size need be larger than 1"
      return configs

    return [triton.Config({'BLOCK_SIZE_N': 16})]

@triton.autotune(
    configs=get_layer_norm_bwd_fused_autotune_config(),
    key=[],
)
@triton.jit
def _layer_norm_bwd_fused(DX,  # pointer to the input gradient
                          DW,  # pointer to the partial sum of weights gradient
                          DB,  # pointer to the partial sum of biases gradient
                          DY,  # pointer to the output gradient
                          X,  # pointer to the input
                          W,  # pointer to the weights
                          Mean,  # pointer to the mean
                          Rstd,  # pointer to the 1/std
                          Lock,  # pointer to the lock
                          stride,  # how much to increase the pointer when moving by 1 row
                          N,  # number of columns in X
                          BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)

    X += row * stride
    DY += row * stride
    DX += row * stride

    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    # Load data to SRAM
    c1 = 0.0
    c2 = 0.0
    for off in range(0, N, BLOCK_SIZE_N):
      cols = off + tl.arange(0, BLOCK_SIZE_N)
      mask = cols < N
      x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
      dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
      w = tl.load(W + cols, mask=mask).to(tl.float32)
      # Compute dx
      xhat = (x - mean) * rstd
      wdy = w * dy
      xhat = tl.where(mask, xhat, 0.)
      wdy = tl.where(mask, wdy, 0.)
      c1 += tl.sum(xhat * wdy, axis=0)
      c2 += tl.sum(wdy, axis=0)

    c1 /= N
    c2 /= N

    for off in range(0, N, BLOCK_SIZE_N):
      # Offset locks and weights/biases gradient pointer for parallel reduction
      off = tl.multiple_of(off, BLOCK_SIZE_N)

      cols = off + tl.arange(0, BLOCK_SIZE_N)
      mask = cols < N
      x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
      dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
      w = tl.load(W + cols, mask=mask).to(tl.float32)
      # Compute dx
      xhat = (x - mean) * rstd
      wdy = w * dy
      xhat = tl.where(mask, xhat, 0.)
      wdy = tl.where(mask, wdy, 0.)
      dx = (wdy - (xhat * c1 + c2)) * rstd

      # Write dx
      tl.store(DX + cols, dx, mask=mask)

      partial_dw = (dy * xhat).to(w.dtype)
      partial_db = (dy).to(w.dtype)

      while tl.atomic_cas(Lock + (off / BLOCK_SIZE_N).to(tl.int32), 0, 1) == 1:
        pass
      partial_dw += tl.load(DW + cols, mask=mask)
      partial_db += tl.load(DB + cols , mask=mask)
      tl.store(DW + cols,  partial_dw, mask=mask)
      tl.store(DB + cols , partial_db, mask=mask)

      # Release the lock
      tl.atomic_xchg(Lock + (off / BLOCK_SIZE_N).to(tl.int32), 0)


# %%
# Benchmark
# ---------
#
# We can now compare the performance of our kernel against that of PyTorch.
# Here we focus on inputs that have Less than 64KB per feature.
# Specifically, one can set :code:`'mode': 'backward'` to benchmark the backward pass.


class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = 8

        if N > MAX_FUSED_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        _layer_norm_fwd_fused[(M, )](  #
            x_arg, y, weight, bias, mean, rstd,  #
            x_arg.stride(0), N, eps
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, m, v = ctx.saved_tensors
        # heuristics for amount of parallel reduction stream for DW/DB
        N = w.shape[0]

        GROUP_SIZE_M = 4

        # allocate output
        dw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        db = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)

        locks = torch.zeros(N, dtype=torch.int32, device=w.device)

        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _layer_norm_bwd_fused[(M, )](  #
            dx, dw, db, dy, x, w, m, v, locks, #
            x_arg.stride(0), N,  #
        )

        return dx, None, dw, db, None


layer_norm = LayerNorm.apply
device = 'cpu'
# Torch doesn't support operations in float16 on CPU so use float32 instead
dtype = torch.float32 if device == 'cpu' else torch.float16


def test_layer_norm(M, N, dtype, eps=1e-5, device='cpu'):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = layer_norm(x, w_shape, weight, bias, eps)

    # backward pass (triton)
    y_tri.backward(dy)
    # dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]


test_layer_norm(1151, 8192, dtype, device=device)





# %%
# References
# ----------
#
# .. [BA2016] Jimmy Lei Ba and Jamie Ryan Kiros and Geoffrey E. Hinton, "Layer Normalization", Arxiv 2016
