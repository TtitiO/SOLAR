from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


METHOD_LABEL = "triton-fused-softcap-softmax"


@triton.jit
def _softcap_softmax_kernel(
    x_ptr,
    y_ptr,
    n_cols: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < n_cols
    base = row_id * n_cols

    x = tl.load(x_ptr + base + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    scaled = x * 0.03333333333333333
    x = (2.0 / (1.0 + tl.exp(-2.0 * scaled)) - 1.0) * 30.0
    x = tl.where(mask, x, -float("inf"))

    row_max = tl.max(x, axis=0)
    numerator = tl.exp(x - row_max)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator

    tl.store(y_ptr + base + offsets, y, mask=mask)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn_weights: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(attn_weights)
        n_cols = attn_weights.shape[-1]
        n_rows = attn_weights.numel() // n_cols
        _softcap_softmax_kernel[(n_rows,)](
            attn_weights,
            output,
            n_cols,
            BLOCK_N=4096,
            num_warps=8,
            num_stages=4,
        )
        return output


def get_inputs():
    torch.manual_seed(0)
    B, H, S = 1, 32, 4096
    attn_weights = torch.randn(B, H, S, S, dtype=torch.bfloat16)
    return [attn_weights]


def get_init_inputs():
    return []
