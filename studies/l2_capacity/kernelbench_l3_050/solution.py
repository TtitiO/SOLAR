from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


METHOD_LABEL = "triton-fused-relu-attention"


@triton.jit
def _relu_attn_fwd_kernel(q, k, v, y, T: tl.constexpr, D: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    base = pid_bh * T * D

    q_blk = tl.load(q + base + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < T, other=0.0)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    scale = 0.125  # 1 / sqrt(64)

    for n0 in range(0, T, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        k_blk = tl.load(k + base + offs_n[:, None] * D + offs_d[None, :], mask=offs_n[:, None] < T, other=0.0)
        scores = tl.dot(q_blk, tl.trans(k_blk), allow_tf32=False) * scale
        causal = offs_n[None, :] <= offs_m[:, None]
        valid = (offs_m[:, None] < T) & (offs_n[None, :] < T) & causal
        scores = tl.where(valid & (scores > 0.0), scores, 0.0)
        v_blk = tl.load(v + base + offs_n[:, None] * D + offs_d[None, :], mask=offs_n[:, None] < T, other=0.0)
        acc += tl.dot(scores, v_blk, allow_tf32=False)

    tl.store(y + base + offs_m[:, None] * D + offs_d[None, :], acc, mask=offs_m[:, None] < T)


class Model(nn.Module):
    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        D = C // self.n_head
        q = q.view(B, T, self.n_head, D).transpose(1, 2).contiguous()
        k = k.view(B, T, self.n_head, D).transpose(1, 2).contiguous()
        v = v.view(B, T, self.n_head, D).transpose(1, 2).contiguous()
        y = torch.empty_like(q)
        grid = (triton.cdiv(T, 16), B * self.n_head)
        _relu_attn_fwd_kernel[grid](q, k, v, y, T, D, BLOCK_M=16, BLOCK_N=32, num_warps=4, num_stages=3)
        return y.transpose(1, 2).contiguous().view(B, T, C)


batch_size = 16
max_seqlen = 1024
n_embd = 768
n_head = 12


def get_inputs():
    return [torch.rand(batch_size, max_seqlen, n_embd)]


def get_init_inputs():
    return [n_embd, n_head, max_seqlen]
