# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused gated MLP (sol_execbench_l1_074) for SOLAR capacity model analysis.

Llama-8B MLP shapes, fp16, B*S=2048 (batch=1, seq=2048):
  hidden_size H=4096, intermediate_size I=14336

Reference:
  up_states = F.linear(hidden_states, gate_up_weight)   # [B,S,2I]
  gate, up_states = up_states.chunk(2, dim=-1)           # each [B,S,I]
  up_states = up_states * (gate * torch.sigmoid(gate))   # SiLU, [B,S,I]
  output = F.linear(up_states, down_weight)              # [B,S,H]

The intermediate up_states [2048, 14336] fp16 = 58.7 MB,
peak live (gate+up) ~117 MB >> L2 (50 MB), making this a true test
of the L2/SRAM capacity model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Fused gated MLP: gate_up GEMM -> SiLU gate -> down GEMM."""

    def __init__(self):
        super().__init__()
        # Llama-8B shapes: gate_up_proj [2*14336, 4096], down_proj [4096, 14336]
        self.gate_up_weight = nn.Parameter(torch.empty(28672, 4096, dtype=torch.float16))
        self.down_weight = nn.Parameter(torch.empty(4096, 14336, dtype=torch.float16))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.gate_up_weight, std=0.02)
        nn.init.normal_(self.down_weight, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # gate_up GEMM: [B, S, H] @ [2I, H]^T -> [B, S, 2I]
        up_states = F.linear(hidden_states, self.gate_up_weight)
        # chunk into gate and up: each [B, S, I]
        gate, up_states = up_states.chunk(2, dim=-1)
        # SiLU gating: up * gate * sigmoid(gate)
        up_states = up_states * (gate * torch.sigmoid(gate))
        # down GEMM: [B, S, I] @ [H, I]^T -> [B, S, H]
        output = F.linear(up_states, self.down_weight)
        return output


def get_inputs():
    """Return hidden_states [1, 2048, 4096] fp16."""
    torch.manual_seed(0)
    return [torch.randn(1, 2048, 4096, dtype=torch.float16)]


def get_init_inputs():
    """No extra init args needed."""
    return []
