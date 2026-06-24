<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# L2 SRAM Capacity Model Validation Report

Experimental validation of the `--no-capacity-model` toggle on `solar.cli.predict_perf_model`,
using two numerically-verified real benchmarks on RTX 4090 hardware.

## Summary

The L2/SRAM capacity model correctly **detects and quantifies** intermediate-tensor spills
when the peak live working set exceeds on-chip memory. However, for the two tensor-core-heavy
operators tested on RTX 4090, the fused bottleneck remains **compute**, so the capacity toggle
does not change `T_SOL` or the SOL score.

## Motivation

In capacity-blind mode (`--no-capacity-model`), Solar assumes all intermediate tensors stay
on-chip (never touch DRAM). This yields an optimistic `T_SOL` that may be **physically
unreachable**. The capacity-aware model adds DRAM traffic for spilled intermediates, raising
`T_SOL` toward a more realistic bound and potentially producing a higher (truer) SOL score.

## Protocol

For each operator, four steps:

1. **Tb** — Reference eager PyTorch on real GPU (≈200 iters, CUDA events, median reported).
2. **Tk** — Optimized kernel (`torch.compile`, median reported). Must satisfy `Tk < Tb`.
3. **T_SOL** — Solar pipeline (Stages 1–4), run twice: `--no-capacity-model` (blind) and default (aware). Extract `fused.runtime_ms` from `perf_RTX_4090.yaml`.
4. **SOL score** — `S(Tk) = 1 / (1 + (Tk − T_SOL) / (Tb − T_SOL))` for both blind and aware.

GPU: NVIDIA GeForce RTX 4090 (2×, CUDA 8.9, 24 GB), arch config `configs/arch/RTX4090.yaml`.

---

## Case 1 — Fused Gated MLP (`sol_execbench_l1_074`)

**Operator**: `gate_up GEMM → chunk → SiLU gate → down GEMM`  
**Shapes**: Llama-8B MLP — `B=1, S=2048, H=4096, I=14336`, fp16  
**Model file**: `model_048.py`

### Geometry

| Tensor | Shape | Size (fp16) |
|--------|-------|-------------|
| `hidden_states` | [1, 2048, 4096] | 16.8 MB |
| `gate_up_weight` | [28672, 4096] | 235.0 MB |
| `up_states` (intermediate) | [2048, 28672] | 117.4 MB |
| `gate` (intermediate) | [2048, 14336] | 58.7 MB |
| SiLU output (intermediate) | [2048, 14336] | 58.7 MB |
| `down_weight` | [4096, 14336] | 117.4 MB |
| `output` | [1, 2048, 4096] | 16.8 MB |

**Peak live intermediates**: 176.2 MB (168.0 MB reported)  
**L2 capacity**: 75.5 MB (72.0 MB reported)  

### Benchmark

| Quantity | Value |
|----------|-------|
| Tb (eager) | 6.8577 ms |
| Tk (`torch.compile`) | 4.9671 ms |
| Speedup Tb/Tk | 1.38× |

Numerical verification: max abs error 7.8e-3 (within fp16 tolerance).

### Solar Pipeline

```
Stage 1 (process_model)   → 7 layers extracted
Stage 2 (toeinsum_model)  → 7/7 layers with einsum equations
Stage 3 (analyze_model)   → 6 layers, 360.8G MACs, 192.9M fused elements
Stage 4 (predict_perf)    → perf_RTX_4090.yaml (blind + aware)
```

### Capacity Diagnostics (aware)

| Field | Value |
|-------|-------|
| `capacity_aware` | true |
| `intermediate_peak_live_bytes` | 176,160,768 (168.0 MB) |
| `sram_capacity_bytes` | 75,497,472 (72.0 MB) |
| `fits_in_l2` | **false** |
| `spill_fraction` | 0.5714 (57.1%) |
| `spilled_bytes` | 369,098,752 (352.0 MB) |

### Roofline Details

| Field | Blind | Aware |
|-------|-------|-------|
| `fused.memory_elements` | 192,937,984 | 377,487,360 |
| `fused.memory_bytes` | 385,875,968 | 754,974,720 |
| `fused.compute_cycles` | 5,505,024 | 5,505,024 |
| `fused.memory_cycles` | 964,689 | 1,887,436 |
| `fused.arithmetic_intensity` | 934.96 | 477.87 |
| `fused.bottleneck` | **compute** | **compute** |

The spill correctly adds 184.5M elements (369 MB) to DRAM traffic, but memory cycles
(1.89M) remain far below compute cycles (5.50M). Both modes have identical `total_cycles`
and therefore identical `runtime_ms`.

### SOL Scores

| Quantity | Value |
|----------|-------|
| T_SOL_blind (ms) | 2.1845333333 |
| T_SOL_aware (ms) | 2.1845333333 |
| S_blind | 0.626788 |
| S_aware | 0.626788 |
| **Δ (aware − blind)** | **0.000000** |

---

## Case 2 — VAE Residual Block (`sol_execbench_l1_002`)

**Operator**: `Conv3×3 → GroupNorm → SiLU → Conv3×3 → GroupNorm → SiLU → residual add`  
**Shapes**: `B=8, C=256, H=W=128`, fp16 (64.0 MB per feature map)  
**Model file**: `model_049.py`

### Geometry

| Tensor | Shape | Size (fp16) |
|--------|-------|-------------|
| Input | [8, 256, 128, 128] | 64.0 MB |
| Conv1 weight | [256, 256, 3, 3] | 1.1 MB |
| Conv1 output | [8, 256, 128, 128] | 64.0 MB |
| GN1 output | [8, 256, 128, 128] | 64.0 MB |
| SiLU1 output | [8, 256, 128, 128] | 64.0 MB |
| Conv2 weight | [256, 256, 3, 3] | 1.1 MB |
| Conv2 output | [8, 256, 128, 128] | 64.0 MB |
| GN2 output | [8, 256, 128, 128] | 64.0 MB |
| SiLU2 output + identity add | [8, 256, 128, 128] | 64.0 MB |

**Peak live intermediates**: 134.2 MB (128.0 MB reported)  
**L2 capacity**: 75.5 MB (72.0 MB reported)  

### Benchmark

| Quantity | Value |
|----------|-------|
| Tb (eager) | 5.7990 ms |
| Tk (`torch.compile`) | 4.9062 ms |
| Speedup Tb/Tk | 1.18× |

Numerical verification: max abs error 7.8e-3 (within fp16 tolerance).

### Solar Pipeline

```
Stage 1 (process_model)   → 8 layers extracted
Stage 2 (toeinsum_model)  → 8/8 layers with einsum equations
Stage 3 (analyze_model)   → 7 layers, 154.6G MACs, 68.3M fused elements
Stage 4 (predict_perf)    → perf_RTX_4090.yaml (blind + aware)
```

### Capacity Diagnostics (aware)

| Field | Value |
|-------|-------|
| `capacity_aware` | true |
| `intermediate_peak_live_bytes` | 134,217,728 (128.0 MB) |
| `sram_capacity_bytes` | 75,497,472 (72.0 MB) |
| `fits_in_l2` | **false** |
| `spill_fraction` | 0.4375 (43.8%) |
| `spilled_bytes` | 352,321,536 (336.0 MB) |

### Roofline Details

| Field | Blind | Aware |
|-------|-------|-------|
| `fused.memory_elements` | 68,288,512 | 244,449,280 |
| `fused.memory_bytes` | 136,577,024 | 488,898,560 |
| `fused.compute_cycles` | 2,359,296 | 2,359,296 |
| `fused.memory_cycles` | 341,442 | 1,222,246 |
| `fused.arithmetic_intensity` | 1132.10 | 316.26 |
| `fused.bottleneck` | **compute** | **compute** |

Spill adds 176.2M elements (336 MB). Memory cycles rise 3.6× but still below compute.

### SOL Scores

| Quantity | Value |
|----------|-------|
| T_SOL_blind (ms) | 0.9362285714 |
| T_SOL_aware (ms) | 0.9362285714 |
| S_blind | 0.550539 |
| S_aware | 0.550539 |
| **Δ (aware − blind)** | **0.000000** |

---

## Analysis

### The capacity model works correctly

In both cases, the aware mode:

- Correctly identifies `fits_in_l2 == false` when peak live (168 / 128 MB) exceeds L2 (72 MB).
- Computes a spill fraction (57.1% / 43.8%) proportional to the overflow.
- Adds the corresponding spilled bytes to `fused.memory_elements` and `fused.memory_bytes`.
- Increases `memory_cycles` proportionally (2.0× / 3.6×).

The blind mode reports the same `capacity_aware: false` and `spilled_bytes: 0`, but still
records the diagnostic fields (`fits_in_l2`, `spill_fraction`) for comparison visibility.

### Why Δ = 0

The roofline model computes `total_cycles = max(compute_cycles, memory_cycles)`.
For both operators on RTX 4090:

| Case | compute_cycles | memory_cycles (blind) | memory_cycles (aware) | headroom |
|------|---------------|----------------------|----------------------|----------|
| MLP | 5,505,024 | 964,689 | 1,887,436 | 2.9× |
| VAE | 2,359,296 | 341,442 | 1,222,246 | 1.9× |

Even with 100% spill (the worst-case capacity miss), both operators would remain
compute-bound. Ridge point = `MAC_per_cycle_fp16_tc / DRAM_byte_per_cycle` = 65536 / 400 = 163.84;
the aware arithmetic intensities (478 / 316) are well above this threshold.

### When Δ ≠ 0

The capacity model changes T_SOL only when the spill pushes memory_cycles above
compute_cycles. This requires:

1. **Memory-bound operators** — AI below the ridge point (element-wise ops, small
   reductions, layer norms, attention softmax patterns, embedding lookups).
2. **Hardware with lower compute:bandwidth ratio** — lower ridge point makes it
   easier for the spill to flip the bottleneck (e.g., GPUs with fewer tensor cores
   or higher memory bandwidth).
3. **Higher spill fractions** — spill close to 100% with AI near the ridge point.

### The toggle's diagnostic value

Even when Δ = 0, the `cache` block provides actionable information:

- `fits_in_l2 == false` signals that fusion alone is insufficient — the intermediate
  working set physically cannot fit on-chip.
- `spill_fraction` quantifies how much intermediate traffic must traverse DRAM,
  independent of whether compute dominates.
- These fields enable **kernel designers** to reason about memory pressure and
  justify techniques like tiling, recomputation, or operator fusion that reduce
  the peak live working set.

---

## Files Produced

```
model_048.py                          # Fused gated MLP model
model_049.py                          # VAE residual model
bench_048.py                          # MLP GPU benchmark
bench_049.py                          # VAE GPU benchmark

output/                               # MLP Solar pipeline
├── graph/pytorch_graph.yaml
├── einsum/einsum_graph_renamed.yaml
├── analysis/analysis.yaml
├── perf_aware/perf_RTX_4090.yaml
└── perf_blind/perf_RTX_4090.yaml

output_vae/                           # VAE Solar pipeline
├── graph/pytorch_graph.yaml
├── einsum/einsum_graph_renamed.yaml
├── analysis/analysis.yaml
├── perf_aware/perf_RTX_4090.yaml
└── perf_blind/perf_RTX_4090.yaml
```

## Reproducing

```bash
# Case 1: Fused Gated MLP
python3 bench_048.py

python -m solar.cli.process_model --model-file model_048.py --output-dir output/graph
python -m solar.cli.toeinsum_model --graph-path output/graph/pytorch_graph.yaml --output-dir output/einsum --no-copy-graph
python -m solar.cli.analyze_model --einsum-graph-path output/einsum/einsum_graph_renamed.yaml --output-dir output/analysis --precision fp16
python -m solar.cli.predict_perf_model --analysis-path output/analysis/analysis.yaml --output-dir output/perf_aware --arch-config configs/arch/RTX4090.yaml --precision fp16
python -m solar.cli.predict_perf_model --analysis-path output/analysis/analysis.yaml --output-dir output/perf_blind --arch-config configs/arch/RTX4090.yaml --precision fp16 --no-capacity-model

# Case 2: VAE Residual
python3 bench_049.py

python -m solar.cli.process_model --model-file model_049.py --output-dir output_vae/graph
python -m solar.cli.toeinsum_model --graph-path output_vae/graph/pytorch_graph.yaml --output-dir output_vae/einsum --no-copy-graph
python -m solar.cli.analyze_model --einsum-graph-path output_vae/einsum/einsum_graph_renamed.yaml --output-dir output_vae/analysis --precision fp16
python -m solar.cli.predict_perf_model --analysis-path output_vae/analysis/analysis.yaml --output-dir output_vae/perf_aware --arch-config configs/arch/RTX4090.yaml --precision fp16
python -m solar.cli.predict_perf_model --analysis-path output_vae/analysis/analysis.yaml --output-dir output_vae/perf_blind --arch-config configs/arch/RTX4090.yaml --precision fp16 --no-capacity-model
```

## Conclusion

1. **T_SOL_aware == T_SOL_blind** for both operators on RTX 4090 — the capacity model adds realistic DRAM time, but compute dominates so total cycles are unchanged.
2. **S_aware == S_blind** — Δ = 0; the capacity toggle does not change the SOL score for these compute-bound operators.
3. **fits_in_l2 == false** and **spill_fraction > 0** are confirmed — the overflow is real and correctly quantified.
4. The capacity model's value is in **identifying when fusion alone is insufficient** (via `fits_in_l2` / `spill_fraction`), even when the runtime bottleneck is unchanged. These diagnostics guide kernel optimization: tiling, recomputation, or operator fusion that shrinks the peak live set below `sram_capacity_bytes`.
