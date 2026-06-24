"""VAE residual block (sol_execbench_l1_002) for SOLAR capacity model analysis.

Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> add (residual)

Shapes: B=8, C=256, H=W=128, fp16
Each feature map: 8*256*128*128*2 = 67,108,864 bytes ≈ 64 MB
With 5+ simultaneous intermediates, peak live >> 75 MB L2 (RTX 4090).
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """VAE residual block: Conv3x3->GN->SiLU->Conv3x3->GN->SiLU->residual_add."""

    def __init__(self):
        super().__init__()
        C = 256
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=C)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = torch.nn.functional.silu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = torch.nn.functional.silu(out)
        return out + identity


def get_inputs():
    """Return input [8, 256, 128, 128] fp16."""
    torch.manual_seed(0)
    return [torch.randn(8, 256, 128, 128, dtype=torch.float16)]


def get_init_inputs():
    """No extra init args needed."""
    return []
