from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


METHOD_LABEL = "triton-flash-causal-attention"


@triton.jit
def _flash_causal_attn_fwd_kernel(q, k, v, y, T: tl.constexpr, D: tl.constexpr,
                                  SM_SCALE: tl.constexpr, BLOCK_M: tl.constexpr,
                                  BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    base = pid_bh * T * D

    q_blk = tl.load(
        q + base + offs_m[:, None] * D + offs_d[None, :],
        mask=(offs_m[:, None] < T) & d_mask[None, :],
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for n0 in range(0, T, BLOCK_N):
        if n0 <= pid_m * BLOCK_M + BLOCK_M - 1:
            offs_n = n0 + offs_n_base
            k_blk = tl.load(
                k + base + offs_n[:, None] * D + offs_d[None, :],
                mask=(offs_n[:, None] < T) & d_mask[None, :],
                other=0.0,
            )
            scores = tl.dot(q_blk, tl.trans(k_blk), allow_tf32=False) * SM_SCALE
            valid = (offs_m[:, None] < T) & (offs_n[None, :] < T) & (offs_n[None, :] <= offs_m[:, None])
            scores = tl.where(valid, scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            l_new = l_i * alpha + tl.sum(p, axis=1)

            v_blk = tl.load(
                v + base + offs_n[:, None] * D + offs_d[None, :],
                mask=(offs_n[:, None] < T) & d_mask[None, :],
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p, v_blk, allow_tf32=False)
            m_i = m_new
            l_i = l_new

    acc = acc / l_i[:, None]
    tl.store(
        y + base + offs_m[:, None] * D + offs_d[None, :],
        acc,
        mask=(offs_m[:, None] < T) & d_mask[None, :],
    )


class Model(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
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
        _flash_causal_attn_fwd_kernel[grid](
            q, k, v, y, T, D, 1.0 / math.sqrt(D),
            BLOCK_M=16, BLOCK_N=32, BLOCK_D=triton.next_power_of_2(D),
            num_warps=4, num_stages=2,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0


def get_inputs():
    return [torch.rand(batch_size, seq_len, n_embd)]


def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
