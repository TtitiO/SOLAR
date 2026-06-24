"""VAE attention block with GroupNorm (sol_execbench_l1_089) — optimized kernel.

DSL chosen: PyTorch — reason: native PyTorch for SOLAR traceability
Operator: VAE single-head spatial attention: GroupNorm → QKV proj → Q@K^T → softmax → @V → output
Benchmark shapes: x [1,512,64,64] → S=4096 single-head attn, fp32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """VAE spatial attention: GroupNorm → single-head QKV → Q@K^T → softmax → @V → proj_out."""

    def __init__(self):
        super().__init__()

    def forward(self, x, group_norm_weight, group_norm_bias,
                query_weight, query_bias, key_weight, key_bias,
                value_weight, value_bias, proj_out_weight, proj_out_bias, eps):
        batch, channels, height, width = x.shape
        num_groups = 32
        residual = x

        # GroupNorm
        channels_per_group = channels // num_groups
        x_grouped = x.view(batch, num_groups, channels_per_group, height, width)
        mean = x_grouped.mean(dim=(2, 3, 4), keepdim=True)
        var = x_grouped.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        x_norm = (x_grouped - mean) / torch.sqrt(var + eps)
        x_norm = x_norm.view(batch, channels, height, width)
        x_norm = x_norm * group_norm_weight.view(1, channels, 1, 1) + group_norm_bias.view(1, channels, 1, 1)

        # Reshape to sequence
        seq_len = height * width
        x_seq = x_norm.view(batch, channels, seq_len).permute(0, 2, 1).contiguous()

        # Q, K, V projections
        q = torch.matmul(x_seq, query_weight.t()) + query_bias
        k = torch.matmul(x_seq, key_weight.t()) + key_bias
        v = torch.matmul(x_seq, value_weight.t()) + value_bias

        # Single-head attention
        scale = channels ** -0.5
        attn_scores = torch.bmm(q, k.transpose(1, 2)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.bmm(attn_weights, v)

        # Output projection
        attn_output = torch.matmul(attn_output, proj_out_weight.t()) + proj_out_bias

        # Residual connection
        attn_output = attn_output.permute(0, 2, 1).contiguous().view(batch, channels, height, width)
        output = residual + attn_output
        return output


def get_inputs():
    torch.manual_seed(0)
    B, C, H, W = 1, 512, 64, 64
    num_groups = 32
    x = torch.randn(B, C, H, W, dtype=torch.float32)
    group_norm_weight = torch.ones(C, dtype=torch.float32)
    group_norm_bias = torch.zeros(C, dtype=torch.float32)
    query_weight = torch.randn(C, C, dtype=torch.float32) * 0.02
    query_bias = torch.zeros(C, dtype=torch.float32)
    key_weight = torch.randn(C, C, dtype=torch.float32) * 0.02
    key_bias = torch.zeros(C, dtype=torch.float32)
    value_weight = torch.randn(C, C, dtype=torch.float32) * 0.02
    value_bias = torch.zeros(C, dtype=torch.float32)
    proj_out_weight = torch.randn(C, C, dtype=torch.float32) * 0.02
    proj_out_bias = torch.zeros(C, dtype=torch.float32)
    eps = 1e-5
    return [x, group_norm_weight, group_norm_bias,
            query_weight, query_bias, key_weight, key_bias,
            value_weight, value_bias, proj_out_weight, proj_out_bias,
            torch.tensor(eps, dtype=torch.float32)]


def get_init_inputs():
    return []
