<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# L2 SRAM Capacity Model Validation Report

This report records the corrected status of the L2-capacity bottleneck-flip
validation study under `studies/l2_capacity/`.

The previous Case 4 run is invalid as a SOL-score validation because the
"optimized" path was effectively the reference path (`Tk ~= Tb`). In that
condition,

```text
S(Tk) = 1 / (1 + (Tk - T_SOL) / (Tb - T_SOL))
```

is pinned near `0.5`, so changing `T_SOL` with the capacity model has almost no
visible effect on the score. The repaired harness now rejects copied-reference
solutions and requires `Tk < Tb` as a hard gate.

## Current corrected status

| Item | Status |
|------|--------|
| Benchmark harness | Shared harness only: `studies/l2_capacity/bench.py` + `bench_utils.py` |
| Per-problem `bench.py` wrappers | Removed; they duplicated stale labels and are no longer used |
| `torch.compile` as optimized kernel | Forbidden for this validation |
| Baseline provenance | Every kept `model.py` is byte-identical to `agentkernelbench_v0.json["torch code"]` |
| Passing real handwritten optimized kernels | 7 problems |
| Included problem directories | Passing problems only |
| Latest shared-harness sweep | 7 passed / 0 failed on local RTX 4090 (`--warmup 20 --iters 200`) |
| Latest SOLAR regeneration | Forced rerun for graph/einsum/analysis/perf artifacts; perf predicted with `configs/arch/RTX4090.yaml` |

## Protocol used by the repaired harness

For each problem directory:

1. `model.py` is the dataset baseline copied byte-for-byte from the matching
   `agentkernelbench_v0.json` item's `"torch code"` field. SOLAR traces this file
   and eager PyTorch benchmarks it as `Tb`.
2. `solution.py` must provide a distinct optimized implementation using a real
   target DSL such as Triton or CUDA C++.
3. The shared harness imports `model.py` and `solution.py` from separate files,
   shares compatible weights/state, validates numerical correctness, measures
   CUDA-event median runtime, and fails unless `Tk < Tb`.
4. Only passing problems are kept in this study directory.

Shared benchmark command:

```bash
python studies/l2_capacity/bench.py <problem_id> --warmup 20 --iters 200
python studies/l2_capacity/bench.py --all
```

SOLAR still runs on the baseline only. For a full regeneration, force the graph
step so stale CPU-only trace artifacts are not reused:

```bash
DIR=studies/l2_capacity/<problem_id>
python -m solar.cli.process_model --model-file $DIR/model.py --output-dir $DIR/output/graph --force-rerun
python -m solar.cli.toeinsum_model --graph-path $DIR/output/graph/pytorch_graph.yaml --output-dir $DIR/output/einsum --no-copy-graph
python -m solar.cli.analyze_model --einsum-graph-path $DIR/output/einsum/einsum_graph_renamed.yaml --output-dir $DIR/output/analysis --precision fp16
python -m solar.cli.predict_perf_model --analysis-path $DIR/output/analysis/analysis.yaml --output-dir $DIR/output/perf_aware --arch-config configs/arch/RTX4090.yaml --precision fp16
python -m solar.cli.predict_perf_model --analysis-path $DIR/output/analysis/analysis.yaml --output-dir $DIR/output/perf_blind --arch-config configs/arch/RTX4090.yaml --precision fp16 --no-capacity-model
```

`sol_execbench_l1_046` is the one function-only baseline (`run()` rather than a
`Model` class), so the latest rerun regenerated its einsum, analysis, and perf
outputs from the existing checked-in `output/graph/pytorch_graph.yaml`.

## Corrected benchmark/SOL results with min-tile spill gate

All rows have a real optimized kernel and pass the `Tk < Tb` gate. With the
reduction-aware min-tile spill gate, the attention and softcap-softmax cases do
not force SOL spill: their whole-tensor peak live values are large, but their
minimal resident tiles fit in L2. The eager baseline `Tb` is slow because it does
not fuse these streams; `T_SOL` is the fusable I/O floor.

| PID | Method | Tb (ms) | Tk (ms) | Speedup | T_SOL blind | T_SOL aware | Bottleneck blind→aware | S_blind | S_aware | Δ |
|-----|--------|---------|---------|---------|-------------|-------------|-------------------------|---------|---------|---|
| `kernelbench_l3_043` | Triton flash causal attention | 22.9676 | 20.6362 | 1.113× | 1.2483 | 1.2483 | compute→compute | 0.5284 | 0.5284 | +0.0000 |
| `kernelbench_l3_044` | Triton flash causal attention block | 57.5804 | 55.9862 | 1.028× | 3.1208 | 3.1208 | compute→compute | 0.5074 | 0.5074 | +0.0000 |
| `kernelbench_l3_050` | Triton fused ReLU causal attention | 11.3551 | 9.6962 | 1.171× | 0.3316 | 0.3316 | compute→compute | 0.5407 | 0.5407 | +0.0000 |
| `multikernelbench_multikernel_064` | Triton flash causal attention block | 58.4246 | 56.7967 | 1.029× | 3.1208 | 3.1208 | compute→compute | 0.5075 | 0.5075 | +0.0000 |
| `multikernelbench_multikernel_073` | Triton fused ReLU causal attention | 11.3951 | 9.7363 | 1.170× | 0.3316 | 0.3316 | compute→compute | 0.5405 | 0.5405 | +0.0000 |
| `multikernelbench_multikernel_104` | Triton flash causal attention | 23.5274 | 21.4742 | 1.096× | 1.2483 | 1.2483 | compute→compute | 0.5242 | 0.5242 | +0.0000 |
| `sol_execbench_l1_046` | Triton fused softcap softmax | 18.6778 | 2.3613 | 7.910× | 2.1304 | 2.1304 | memory→memory | 0.9862 | 0.9862 | +0.0000 |

These seven rows demonstrate the corrected lower-bound behavior: all reported
scores satisfy `S≤1`. When `Tk ~= Tb`, the denominator of
`S(Tk) = 1 / (1 + (Tk − T_SOL) / (Tb − T_SOL))` is near zero, collapsing the
score toward 0.5 regardless of T_SOL; the harness still rejects copied-reference
solutions to avoid that invalid validation mode.

## SOLAR bottleneck diagnostics from regenerated min-tile runs

The old whole-tensor peak-live diagnostics were useful for identifying the
overcharge, but they are no longer the spill gate. The regenerated runs report
`gate_metric: min_tile`; all seven min tiles fit in the RTX 4090's 75,497,472 B
L2 and therefore charge no spill even when whole-tensor peak live is much larger:

| PID | T_SOL blind | T_SOL aware | Blind bottleneck | Aware bottleneck | Flip | Gate | Min tile bytes | Peak live bytes | Spill |
|-----|-------------|-------------|------------------|------------------|------|------|----------------|-----------------|-------|
| `kernelbench_l3_043` | 1.2483 | 1.2483 | compute | compute | NO | min_tile | 6,144 | 1,174,405,120 | 0.0% |
| `kernelbench_l3_044` | 3.1208 | 3.1208 | compute | compute | NO | min_tile | 7,680 | 1,610,612,736 | 0.0% |
| `kernelbench_l3_050` | 0.3316 | 0.3316 | compute | compute | NO | min_tile | 6,144 | 830,472,192 | 0.0% |
| `multikernelbench_multikernel_064` | 3.1208 | 3.1208 | compute | compute | NO | min_tile | 7,680 | 1,610,612,736 | 0.0% |
| `multikernelbench_multikernel_073` | 0.3316 | 0.3316 | compute | compute | NO | min_tile | 6,144 | 830,472,192 | 0.0% |
| `multikernelbench_multikernel_104` | 1.2483 | 1.2483 | compute | compute | NO | min_tile | 6,144 | 1,174,405,120 | 0.0% |
| `sol_execbench_l1_046` | 2.1304 | 2.1304 | memory | memory | NO | min_tile | 8,194 | 2,147,483,648 | 0.0% |

## Interpretation

The corrected results support two conclusions:

1. Whole-tensor peak live is not a valid spill gate for fused tileable kernels;
   it can make `T_SOL` slower than a real fused kernel and produce `S>1`.
2. SOL-score validation requires a genuine optimized kernel. When `Tk ~= Tb`, the
   score collapses toward `0.5`; when `Tk < Tb` is real, the aware/blind score
   remains meaningful. For these seven rows, min-tile capacity accounting keeps
   the fusable lower bound below the measured Triton kernels (`S≤1`).
