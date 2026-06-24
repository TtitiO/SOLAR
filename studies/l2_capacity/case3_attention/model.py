"""Full attention with QK LayerNorm (sol_execbench_l1_082) for L2 capacity demo.

QKV_proj -> QK LayerNorm -> QK^T -> softmax -> AV -> o_proj
Fixed dims: num_heads=24, head_dim=64, dim=1536, batch=1, seq_len=2048, fp16

The attn_scores [1,24,2048,2048] fp32 = 402 MB is a TRUE INTERMEDIATE
between QK^T and softmax/AV. This makes the spill large enough to flip
the roofline bottleneck from compute to memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Full attention: QKV_proj -> QK LN -> QK^T -> softmax -> AV -> o_proj."""

    def __init__(self):
        super().__init__()
        self.num_heads = 24
        self.head_dim = 64
        self.dim = self.num_heads * self.head_dim  # 1536

    def forward(self, hidden_states, qkv_weight, qkv_bias,
                q_norm_weight, q_norm_bias, k_norm_weight, k_norm_bias,
                out_proj_weight, out_proj_bias):
        B, S, _ = hidden_states.shape
        dim = self.dim
        num_heads = self.num_heads
        head_dim = self.head_dim
        eps = 1e-5

        # 1. QKV projection: [B,S,dim] @ [3*dim,dim]^T -> [B,S,3*dim]
        qkv = F.linear(hidden_states, qkv_weight, qkv_bias)

        # 2. Split into Q, K, V: each [B,S,dim]
        q, k, v = qkv.split(dim, dim=-1)

        # 3. Reshape to multi-head: [B,S,dim] -> [B,S,H,D] -> [B,H,S,D]
        q = q.view(B, S, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, S, num_heads, head_dim).transpose(1, 2)
        v = v.view(B, S, num_heads, head_dim).transpose(1, 2)

        # 4. QK LayerNorm (per-head, across head_dim=64)
        q = F.layer_norm(q, [head_dim], weight=q_norm_weight, bias=q_norm_bias, eps=eps)
        k = F.layer_norm(k, [head_dim], weight=k_norm_weight, bias=k_norm_bias, eps=eps)

        # 5. QK^T: attn_scores in fp32 for softmax stability
        #    [B,H,S,D] @ [B,H,D,S] -> [B,H,S,S] fp32  (402 MB at S=2048!)
        q_fp32 = q.float()
        k_fp32 = k.float()
        attn_scores = torch.matmul(q_fp32, k_fp32.transpose(-2, -1))
        scale = head_dim ** -0.5
        attn_scores = attn_scores * scale

        # 6. Softmax (keeps the tensor in fp32)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # 7. AV: [B,H,S,S] @ [B,H,S,D] -> [B,H,S,D] fp32
        v_fp32 = v.float()
        attn_output = torch.matmul(attn_probs, v_fp32)

        # 8. Back to fp16 and reshape: [B,H,S,D] -> [B,S,H,D] -> [B,S,dim]
        attn_output = attn_output.to(torch.float16)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, dim)

        # 9. Output projection: [B,S,dim] @ [dim,dim]^T -> [B,S,dim]
        output = F.linear(attn_output, out_proj_weight, out_proj_bias)

        return output


def get_inputs():
    """Return all inputs for forward() at shapes above."""
    torch.manual_seed(0)
    B, S = 1, 2048
    dim = 1536
    three_dim = 3 * dim
    head_dim = 64

    return [
        torch.randn(B, S, dim, dtype=torch.float16),          # hidden_states
        torch.randn(three_dim, dim, dtype=torch.float16),      # qkv_weight
        torch.randn(three_dim, dtype=torch.float16),           # qkv_bias
        torch.ones(head_dim, dtype=torch.float16),             # q_norm_weight
        torch.zeros(head_dim, dtype=torch.float16),            # q_norm_bias
        torch.ones(head_dim, dtype=torch.float16),             # k_norm_weight
        torch.zeros(head_dim, dtype=torch.float16),            # k_norm_bias
        torch.randn(dim, dim, dtype=torch.float16),            # out_proj_weight
        torch.zeros(dim, dtype=torch.float16),                 # out_proj_bias
    ]


def get_init_inputs():
    """No extra init args needed."""
    return []
