"""Attention softmax with softcapping (sol_execbench_l1_046) — optimized kernel.

DSL chosen: PyTorch — reason: Elementwise softcapping + softmax; PyTorch's built-in
  CUDA kernels are already highly optimized for tanh+softmax on RTX 4090.
Operator: Gemma-2 style attention: tanh(logits/30)*30 then softmax normalization.
Benchmark shapes: [1, 32, 4096, 4096] bf16 → input is 1024 MB (>>L2 72 MB)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Softcapping + softmax: tanh(logits / 30.0) * 30.0, then softmax."""

    def __init__(self):
        super().__init__()

    def forward(self, attn_weights: torch.Tensor) -> torch.Tensor:
        SOFTCAP = 30.0
        scaled = attn_weights / SOFTCAP
        clamped = torch.tanh(scaled)
        softcapped = clamped * SOFTCAP
        output = F.softmax(softcapped, dim=-1, dtype=torch.float32).to(attn_weights.dtype)
        return output


def get_inputs():
    """Return all forward() inputs as CPU tensors at benchmark shapes."""
    torch.manual_seed(0)
    B, H, S = 1, 32, 4096
    attn_weights = torch.randn(B, H, S, S, dtype=torch.bfloat16)
    return [attn_weights]


def get_init_inputs():
    return []
