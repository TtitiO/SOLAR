# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the L2/SRAM capacity-aware fused SOL model.

Covers the fix for docs/ISSUE_L2_CAPACITY_UNMODELED.md:

1. A graph whose intermediate working set FITS in SRAM_capacity reports
   unchanged fused results (no spill).
2. A graph whose intermediate working set EXCEEDS SRAM_capacity reports a
   strictly higher fused memory cost than the capacity-blind model.
3. The `capacity_aware=False` switch reproduces the original optimistic numbers.
4. The perf output exposes the working-set vs capacity diagnostic.
5. The analysis exposes peak-live (not lifetime-sum) intermediate elements.
"""

from pathlib import Path
from textwrap import dedent

from solar.common.types import ProcessingConfig
from solar.graph import PyTorchProcessor
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.perf import EinsumGraphPerfModel


# Small MLP-like graph: matmul -> relu -> matmul.  The intermediate activation
# between the two matmuls is the tensor whose residency we care about.  With
# small dims the working set trivially fits in L2.
SMALL_MLP_SOURCE = """\
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(64, 256))
        self.w2 = nn.Parameter(torch.randn(256, 64))

    def forward(self, x):
        h = torch.matmul(x, self.w1)
        h = F.relu(h)
        return torch.matmul(h, self.w2)

def get_inputs():
    torch.manual_seed(0)
    return [torch.randn(4, 32, 64)]

def get_init_inputs():
    return []
"""

# Large MLP-like graph: same structure, but the intermediate activation
# [B, S, 4096] in fp16 is ~134 MB for B*S=16384 — well over H100's 50 MB L2.
LARGE_MLP_SOURCE = """\
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(64, 4096))
        self.w2 = nn.Parameter(torch.randn(4096, 64))

    def forward(self, x):
        h = torch.matmul(x, self.w1)
        h = F.relu(h)
        return torch.matmul(h, self.w2)

def get_inputs():
    torch.manual_seed(0)
    return [torch.randn(8, 2048, 64)]

def get_init_inputs():
    return []
"""


def _analyze(tmp_path: Path, source: str):
    """Run extraction + einsum + analysis. Returns (analysis dict, analysis path)."""
    model_file = tmp_path / "model.py"
    model_file.write_text(dedent(source))

    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    config = ProcessingConfig(
        save_graph=False, force_rerun=True, debug=False, safe_mode=False,
    )
    processor = PyTorchProcessor(config)
    assert processor.process_model_file(str(model_file), str(graph_dir))

    einsum_dir = tmp_path / "einsum"
    einsum_dir.mkdir()
    converter = PyTorchToEinsum()
    result = converter.convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir))
    assert result is not None

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    analyzer = EinsumGraphAnalyzer()
    analysis = analyzer.analyze_graph(
        str(einsum_dir / "einsum_graph_renamed.yaml"), str(analysis_dir),
        precision="fp16", copy_graph=False,
    )
    assert analysis is not None
    return analysis, analysis_dir / "analysis.yaml"


def _predict(tmp_path: Path, analysis_path: Path, *, capacity_aware: bool, tag: str):
    perf_dir = tmp_path / f"perf_{tag}"
    perf_dir.mkdir(exist_ok=True)
    model = EinsumGraphPerfModel()
    perf = model.predict(
        str(analysis_path), str(perf_dir),
        arch_config="H100_PCIe", precision="fp16",
        capacity_aware=capacity_aware, copy_analysis=False,
    )
    assert perf is not None
    return perf


class TestAnalysisPeakLive:
    def test_peak_live_field_emitted(self, tmp_path):
        analysis, _ = _analyze(tmp_path, SMALL_MLP_SOURCE)
        assert "intermediate_peak_live_elements" in analysis["total"]

    def test_peak_live_not_exceeding_lifetime_sum(self, tmp_path):
        """Peak simultaneously-live <= lifetime sum of intermediates."""
        analysis, _ = _analyze(tmp_path, SMALL_MLP_SOURCE)
        peak = analysis["total"]["intermediate_peak_live_elements"]
        lifetime = analysis["total"]["intermediate_elements"]
        assert 0 <= peak <= lifetime


class TestCapacityModelFits:
    """Small graph fits in L2 -> capacity model is a no-op."""

    def test_no_spill_when_fits(self, tmp_path):
        _, analysis_path = _analyze(tmp_path, SMALL_MLP_SOURCE)
        perf = _predict(tmp_path, analysis_path, capacity_aware=True, tag="aware")
        cache = perf["cache"]
        assert cache["fits_in_l2"] is True
        assert cache["spilled_bytes"] == 0
        assert cache["spill_fraction"] == 0.0

    def test_aware_equals_blind_when_fits(self, tmp_path):
        """When the working set fits, capacity-aware == capacity-blind."""
        _, analysis_path = _analyze(tmp_path, SMALL_MLP_SOURCE)
        aware = _predict(tmp_path, analysis_path, capacity_aware=True, tag="aware")
        blind = _predict(tmp_path, analysis_path, capacity_aware=False, tag="blind")
        assert aware["fused"]["memory_bytes"] == blind["fused"]["memory_bytes"]
        assert aware["fused"]["runtime_ms"] == blind["fused"]["runtime_ms"]


class TestCapacityModelSpills:
    """Large graph overflows L2 -> spill increases fused cost."""

    def test_spill_detected(self, tmp_path):
        _, analysis_path = _analyze(tmp_path, LARGE_MLP_SOURCE)
        perf = _predict(tmp_path, analysis_path, capacity_aware=True, tag="aware")
        cache = perf["cache"]
        assert cache["fits_in_l2"] is False
        assert cache["spilled_bytes"] > 0
        assert 0.0 < cache["spill_fraction"] < 1.0
        assert cache["intermediate_peak_live_bytes"] > cache["sram_capacity_bytes"]

    def test_aware_strictly_higher_than_blind(self, tmp_path):
        """Capacity-aware fused cost must exceed the optimistic blind cost."""
        _, analysis_path = _analyze(tmp_path, LARGE_MLP_SOURCE)
        aware = _predict(tmp_path, analysis_path, capacity_aware=True, tag="aware")
        blind = _predict(tmp_path, analysis_path, capacity_aware=False, tag="blind")
        assert aware["fused"]["memory_bytes"] > blind["fused"]["memory_bytes"]

    def test_blind_reproduces_original(self, tmp_path):
        """capacity_aware=False reports no spill regardless of overflow."""
        _, analysis_path = _analyze(tmp_path, LARGE_MLP_SOURCE)
        blind = _predict(tmp_path, analysis_path, capacity_aware=False, tag="blind")
        assert blind["cache"]["spilled_bytes"] == 0
        assert blind["cache"]["capacity_aware"] is False

    def test_spill_bounded_by_unfused(self, tmp_path):
        """Spilled fused cost never exceeds the unfused upper bound."""
        _, analysis_path = _analyze(tmp_path, LARGE_MLP_SOURCE)
        aware = _predict(tmp_path, analysis_path, capacity_aware=True, tag="aware")
        assert aware["fused"]["memory_bytes"] <= aware["unfused"]["memory_bytes"]
