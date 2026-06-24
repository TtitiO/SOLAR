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

---

## Case 3 — Full Attention (`sol_execbench_l1_082`) ✅ Bottleneck Flips

**Operator**: `QKV_proj → QK LayerNorm → QK^T → softmax → AV → o_proj`  
**Shapes**: `H=24, D=64, dim=1536, B=1, S=2048`, fp16 (attn scores in fp32)  
**Model file**: `model_082.py`

### Geometry

| Tensor | Shape | Size |
|--------|-------|------|
| `hidden_states` | [1, 2048, 1536] fp16 | 6.3 MB |
| `qkv_weight` | [4608, 1536] fp16 | 14.2 MB |
| `q` / `k` / `v` (per head) | [1, 24, 2048, 64] fp16 | 3.1 MB each |
| `attn_scores` (intermediate) | [1, 24, 2048, 2048] fp32 | **390 MB** |
| `attn_probs` (intermediate) | [1, 24, 2048, 2048] fp32 | **390 MB** |
| `out_proj_weight` | [1536, 1536] fp16 | 4.7 MB |
| `output` | [1, 2048, 1536] fp16 | 6.3 MB |

**Peak live intermediates**: 408.9 MB (390 MB reported)  
**L2 capacity**: 75.5 MB (72.0 MB reported)

### Benchmark

| Quantity | Value |
|----------|-------|
| Tb (eager) | 4.1078 ms |
| Tk (`torch.compile`) | 1.7336 ms |
| Speedup Tb/Tk | 2.37× |

### Solar Pipeline

```
Stage 1 (process_model)   → 30 layers extracted
Stage 2 (toeinsum_model)  → 30/30 layers with einsum equations
Stage 3 (analyze_model)   → 25 layers, 32.2G MACs, 6.3M fused elements
Stage 4 (predict_perf)    → perf_RTX_4090.yaml (blind + aware)
```

### Capacity Diagnostics

| Field | Blind | Aware |
|-------|-------|-------|
| `capacity_aware` | false | true |
| `spilled_bytes` | 0 | 1,128,590,414 (1.05 GB) |
| `spill_fraction` | 0.8154 | 0.8154 |
| `peak_live_bytes` | 408,944,640 | 408,944,640 |
| `fits_in_l2` | **false** | **false** |

### Roofline Details

| Field | Blind | Aware |
|-------|-------|-------|
| `fused.memory_bytes` | 12,582,912 (12 MB) | 1,141,173,326 (1.09 GB) |
| `fused.compute_cycles` | 491,520 | 491,520 |
| `fused.memory_cycles` | 31,457 | **2,852,933** |
| `fused.bottleneck` | **compute** | **memory** |
| `fused.runtime_ms` | **0.1950** | **1.1321** |

Memory cycles jump **91×** (31K → 2,853K), flipping the bottleneck compute→memory.
T_SOL increases **5.8×**.

### SOL Scores

| Quantity | Value |
|----------|-------|
| Tb (ms) | 4.1078 |
| Tk (ms) | 1.7336 |
| T_SOL_blind (ms) | 0.1950 |
| T_SOL_aware (ms) | 1.1321 |
| blind bottleneck | compute |
| aware bottleneck | **memory** (flipped) |
| spill_fraction | 0.8154 (81.5%) |
| S_blind | 0.7178 |
| S_aware | 0.8319 |
| **Δ (aware − blind)** | **+0.1141 (+11.4 pp)** |

### Interpretation

- **T_SOL_blind = 0.195 ms** is physically unreachable: it assumes the 390 MB `attn_scores`
  tensor stays in 72 MB L2. Only 12.6 MB of model I/O is counted, so the model thinks
  the graph is compute-bound (491K > 31K cycles).
- **T_SOL_aware = 1.132 ms** adds 1.05 GB of spilled DRAM traffic. Memory cycles swamp
  compute (2,853K > 491K), and the true bottleneck is memory.
- **S_blind = 0.7178** undervalues the kernel because its denominator `Tb − T_SOL_blind`
  is inflated by an impossibly low T_SOL.
- **S_aware = 0.8319** gives the correct score: the optimized kernel achieves 83.2%
  of the achievable memory-bandwidth roofline.
- **Δ = +11.4 pp**: a kernel ranked by S_blind would be undervalued by 11.4 percentage
  points relative to its true capability.

---

## Analysis

### Why Cases 1–2 showed Δ = 0

The roofline model computes `total_cycles = max(compute_cycles, memory_cycles)`.
For the MLP and VAE operators on RTX 4090:

| Case | compute_cycles | memory_cycles (blind) | memory_cycles (aware) | headroom |
|------|---------------|----------------------|----------------------|----------|
| MLP | 5,505,024 | 964,689 | 1,887,436 | 2.9× |
| VAE | 2,359,296 | 341,442 | 1,222,246 | 1.9× |

Even with 100% spill, both GEMM-heavy operators remain compute-bound. Ridge point
= 163.84; their aware arithmetic intensities (478 / 316) are well above this.

### Why Case 3 shows Δ = +11.4 pp

The attention operator has a much lower arithmetic intensity because the `[B,H,S,S]`
intermediate dwarfs the compute:

| Case | compute_cycles | memory_cycles (blind) | memory_cycles (aware) | headroom |
|------|---------------|----------------------|----------------------|----------|
| Attention | 491,520 | 31,457 | 2,852,933 | **flipped** |

In blind mode, the fused model counts only 12.6 MB of model I/O (the attention scores
are "free" intermediates). Memory cycles are negligible and the graph appears
compute-bound. In aware mode, the 81.5% spill adds 1.05 GB of DRAM traffic, pushing
memory cycles to 91× their blind value and overwhelming compute. The bottleneck flips.

### When Δ ≠ 0

The capacity model changes T_SOL only when the spill pushes memory_cycles above
compute_cycles. This requires:

1. **Memory-bound operators or near-ridge operators** — AI below or near the ridge
   point (attention with large `[B,H,S,S]` intermediates, element-wise ops, layer norms,
   reduction ops on large tensors).
2. **Intermediate-dominated graphs** — where intermediate traffic is orders of
   magnitude larger than model I/O (the attention scores alone are 390 MB vs 12.6 MB
   of model weights + I/O).
3. **Large enough spill fraction** — the [B,H,S,S] tensor grows as O(S²), so any
   reasonable sequence length quickly overflows L2.

### Diagnostic value in all cases

Even when Δ = 0, the `cache` block provides actionable information:

- `fits_in_l2 == false` signals that fusion alone is insufficient.
- `spill_fraction` quantifies how much intermediate traffic must traverse DRAM.
- These fields guide kernel optimization (tiling, recomputation, FlashAttention-style
  fusion) to shrink the peak live set below `sram_capacity_bytes`.

---

## Files Produced

All study artifacts live under `studies/l2_capacity/`:

```
studies/l2_capacity/
├── README.md
├── case1_mlp/
│   ├── model.py                    # Fused gated MLP model
│   ├── bench.py                    # GPU benchmark
│   └── output/                     # Solar pipeline output
├── case2_vae/
│   ├── model.py                    # VAE residual model
│   ├── bench.py                    # GPU benchmark
│   └── output/
└── case3_attention/
    ├── model.py                    # Full attention model
    ├── bench.py                    # GPU benchmark
    └── output/
```

## Reproducing

All commands run from the repository root.

```bash
# Case 1: Fused Gated MLP
cd studies/l2_capacity/case1_mlp && python3 bench.py

python -m solar.cli.process_model --model-file studies/l2_capacity/case1_mlp/model.py --output-dir studies/l2_capacity/case1_mlp/output/graph
python -m solar.cli.toeinsum_model --graph-path studies/l2_capacity/case1_mlp/output/graph/pytorch_graph.yaml --output-dir studies/l2_capacity/case1_mlp/output/einsum --no-copy-graph
python -m solar.cli.analyze_model --einsum-graph-path studies/l2_capacity/case1_mlp/output/einsum/einsum_graph_renamed.yaml --output-dir studies/l2_capacity/case1_mlp/output/analysis --precision fp16
python -m solar.cli.predict_perf_model --analysis-path studies/l2_capacity/case1_mlp/output/analysis/analysis.yaml --output-dir studies/l2_capacity/case1_mlp/output/perf_aware --arch-config configs/arch/RTX4090.yaml --precision fp16
python -m solar.cli.predict_perf_model --analysis-path studies/l2_capacity/case1_mlp/output/analysis/analysis.yaml --output-dir studies/l2_capacity/case1_mlp/output/perf_blind --arch-config configs/arch/RTX4090.yaml --precision fp16 --no-capacity-model

# Case 2: VAE Residual
cd studies/l2_capacity/case2_vae && python3 bench.py

python -m solar.cli.process_model --model-file studies/l2_capacity/case2_vae/model.py --output-dir studies/l2_capacity/case2_vae/output/graph
python -m solar.cli.toeinsum_model --graph-path studies/l2_capacity/case2_vae/output/graph/pytorch_graph.yaml --output-dir studies/l2_capacity/case2_vae/output/einsum --no-copy-graph
python -m solar.cli.analyze_model --einsum-graph-path studies/l2_capacity/case2_vae/output/einsum/einsum_graph_renamed.yaml --output-dir studies/l2_capacity/case2_vae/output/analysis --precision fp16
python -m solar.cli.predict_perf_model --analysis-path studies/l2_capacity/case2_vae/output/analysis/analysis.yaml --output-dir studies/l2_capacity/case2_vae/output/perf_aware --arch-config configs/arch/RTX4090.yaml --precision fp16
python -m solar.cli.predict_perf_model --analysis-path studies/l2_capacity/case2_vae/output/analysis/analysis.yaml --output-dir studies/l2_capacity/case2_vae/output/perf_blind --arch-config configs/arch/RTX4090.yaml --precision fp16 --no-capacity-model

# Case 3: Full Attention (sol_execbench_l1_082)
cd studies/l2_capacity/case3_attention && python3 bench.py

python -m solar.cli.process_model --model-file studies/l2_capacity/case3_attention/model.py --output-dir studies/l2_capacity/case3_attention/output/graph
python -m solar.cli.toeinsum_model --graph-path studies/l2_capacity/case3_attention/output/graph/pytorch_graph.yaml --output-dir studies/l2_capacity/case3_attention/output/einsum --no-copy-graph
python -m solar.cli.analyze_model --einsum-graph-path studies/l2_capacity/case3_attention/output/einsum/einsum_graph_renamed.yaml --output-dir studies/l2_capacity/case3_attention/output/analysis --precision fp16
python -m solar.cli.predict_perf_model --analysis-path studies/l2_capacity/case3_attention/output/analysis/analysis.yaml --output-dir studies/l2_capacity/case3_attention/output/perf_aware --arch-config configs/arch/RTX4090.yaml --precision fp16
python -m solar.cli.predict_perf_model --analysis-path studies/l2_capacity/case3_attention/output/analysis/analysis.yaml --output-dir studies/l2_capacity/case3_attention/output/perf_blind --arch-config configs/arch/RTX4090.yaml --precision fp16 --no-capacity-model
```

## Conclusion

1. **The capacity model works correctly** — all three cases show correct spill detection
   and byte accounting. `fits_in_l2 == false` when peak live exceeds L2 capacity.

2. **Δ = 0 for compute-bound operators (Cases 1–2)** — the MLP and VAE operators are
   GEMM-heavy; the spill adds DRAM traffic but compute cycles still dominate. The
   capacity toggle does not change T_SOL or the SOL score.

3. **Δ = +0.114 (+11.4 pp) for the attention operator (Case 3)** — the `[B,H,S,S]`
   attention scores intermediate creates a 91× memory cycle increase, flipping the
   bottleneck compute→memory. T_SOL rises 5.8×. The kernel's true SOL score is
   11.4 percentage points higher than the capacity-blind score would suggest.

4. **Condition for Δ ≠ 0**: the operator must be memory-bound (or near the ridge)
   in blind mode, OR the spill must be large enough to push `mem_cycles > compute_cycles`.
   This requires intermediate-dominated graphs where intermediate traffic dwarfs model I/O
   (e.g., attention scores, large feature map chains in non-GEMM ops).

5. **Diagnostic value is universal** — even when Δ = 0, `fits_in_l2` and `spill_fraction`
   guide kernel optimization: they tell you whether fusion alone is sufficient, and
   quantify the memory pressure that tiling, recomputation, or fused kernels must address.
