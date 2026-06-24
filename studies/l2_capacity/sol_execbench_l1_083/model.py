"""Attention score @ V matmul (sol_execbench_l1_083) — optimized kernel.

DSL chosen: PyTorch — reason: native PyTorch matmul+transpose is highly optimized on RTX 4090
Operator: Fused attention output: attention_weights @ V → transpose → reshape.
Benchmark shapes: attn_weights [1,20,2048,2048] bf16, value [1,20,2048,64] bf16
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """attention_weights @ V matmul + transpose + reshape (second half of attention)."""

    def __init__(self):
        super().__init__()

    def forward(self, attention_weights: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size = attention_weights.shape[0]
        seq_len_q = attention_weights.shape[2]
        num_heads = attention_weights.shape[1]
        head_dim = value.shape[-1]
        hidden_size = num_heads * head_dim

        # [B, H, Q, K] @ [B, H, K, D] -> [B, H, Q, D]
        attn_output = torch.matmul(attention_weights, value)
        # [B, H, Q, D] -> [B, Q, H, D] -> [B, Q, H*D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len_q, hidden_size)
        return attn_output


def get_inputs():
    torch.manual_seed(0)
    B, H, S, D = 1, 20, 2048, 64
    attention_weights = torch.randn(B, H, S, S, dtype=torch.bfloat16)
    value = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    return [attention_weights, value]


def get_init_inputs():
    return []
