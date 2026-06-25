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
| Final shared-harness sweep | 7 passed / 0 failed (`--warmup 20 --iters 200`) |

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

SOLAR still runs on the baseline only:

```bash
DIR=studies/l2_capacity/<problem_id>
python -m solar.cli.process_model --model-file $DIR/model.py --output-dir $DIR/output/graph
python -m solar.cli.toeinsum_model --graph-path $DIR/output/graph/pytorch_graph.yaml --output-dir $DIR/output/einsum --no-copy-graph
python -m solar.cli.analyze_model --einsum-graph-path $DIR/output/einsum/einsum_graph_renamed.yaml --output-dir $DIR/output/analysis --precision fp16
python -m solar.cli.predict_perf_model --analysis-path $DIR/output/analysis/analysis.yaml --output-dir $DIR/output/perf_aware --arch-config configs/arch/RTX4090.yaml --precision fp16
python -m solar.cli.predict_perf_model --analysis-path $DIR/output/analysis/analysis.yaml --output-dir $DIR/output/perf_blind --arch-config configs/arch/RTX4090.yaml --precision fp16 --no-capacity-model
```

## Corrected benchmark/SOL results

All rows have a real optimized kernel and pass the `Tk < Tb` gate. The original
harness allowed `solution.py` to be a copy of `model.py` (`Tk ~= Tb`), which
pinned the SOL score near 0.5 with negligible blind/aware Δ (all |Δ| < 10⁻³).
The corrected harness enforces `Tk < Tb`, letting the score move away from 0.5
and the Δ become measurable.

| PID | Method | Tb (ms) | Tk (ms) | Speedup | T_SOL blind | T_SOL aware | Bottleneck blind→aware | S_blind | S_aware | Δ |
|-----|--------|---------|---------|---------|-------------|-------------|-------------------------|---------|---------|---|
| `kernelbench_l3_043` | Triton flash causal attention | 25.6957 | 22.8490 | 1.125× | 1.2483 | 6.8701 | compute→memory | 0.5309 | 0.5409 | **+0.0100** |
| `kernelbench_l3_044` | Triton flash causal attention block | 64.5526 | 62.7004 | 1.030× | 3.1208 | 16.8074 | compute→memory | 0.5077 | 0.5099 | **+0.0022** |
| `kernelbench_l3_050` | Triton fused ReLU causal attention | 11.4063 | 9.7454 | 1.170× | 0.3316 | 3.2309 | compute→memory | 0.5405 | 0.5565 | **+0.0160** |
| `multikernelbench_multikernel_064` | Triton flash causal attention block | 58.9169 | 57.2212 | 1.030× | 3.1208 | 16.8074 | compute→memory | 0.5077 | 0.5103 | **+0.0026** |
| `multikernelbench_multikernel_073` | Triton fused ReLU causal attention | 11.4150 | 9.7504 | 1.171× | 0.3316 | 3.2309 | compute→memory | 0.5406 | 0.5566 | **+0.0160** |
| `multikernelbench_multikernel_104` | Triton flash causal attention | 23.6211 | 21.5836 | 1.094× | 1.2483 | 6.8701 | compute→memory | 0.5239 | 0.5324 | **+0.0085** |
| `sol_execbench_l1_046` | Triton fused softcap softmax | 18.6740 | 2.3613 | 7.908× | 2.1304 | 10.3526 | memory→memory | 0.9862 | 25.2089† | **+24.2226†** |

These seven rows demonstrate the intended behavior: once `Tk < Tb` is real, the
capacity-aware model raises `T_SOL` for spilled attention intermediates and the
SOL-score delta becomes non-zero. When `Tk ~= Tb`, the denominator of
`S(Tk) = 1 / (1 + (Tk − T_SOL) / (Tb − T_SOL))` is near zero, collapsing the
score toward 0.5 regardless of T_SOL.

† `sol_execbench_l1_046` is an out-of-range score case: the handwritten Triton
softcap+softmax kernel is faster than SOLAR's capacity-aware lower-bound estimate
(`Tk < T_SOL_aware`), so the score formula exceeds 1. This is reported explicitly
rather than clipped.

## Preserved SOLAR bottleneck diagnostics from the previous baseline runs

The old `Tk ~= Tb` scores are invalid, but the SOLAR baseline capacity diagnostics
for the kept passing problems remain useful because they are produced from
`model.py` only. The previous run showed these blind→aware bottleneck
classifications:

| PID | T_SOL blind | T_SOL aware | Blind bottleneck | Aware bottleneck | Flip | Spill |
|-----|-------------|-------------|------------------|------------------|------|-------|
| `kernelbench_l3_043` | 1.2483 | 6.8701 | compute | memory | YES | 93.6% |
| `kernelbench_l3_044` | 3.1208 | 16.8074 | compute | memory | YES | 95.3% |
| `kernelbench_l3_050` | 0.3316 | 3.2310 | compute | memory | YES | 90.9% |
| `multikernelbench_multikernel_064` | 3.1208 | 16.8074 | compute | memory | YES | 95.3% |
| `multikernelbench_multikernel_073` | 0.3316 | 3.2310 | compute | memory | YES | 90.9% |
| `multikernelbench_multikernel_104` | 1.2483 | 6.8701 | compute | memory | YES | 93.6% |
| `sol_execbench_l1_046` | 2.1304 | 10.3526 | memory | memory | NO | 96.5% |

## Interpretation

The corrected results support two conclusions:

1. The capacity model is still load-bearing for classification: many attention
   baselines flip from compute-bound to memory-bound once L2 spill traffic is
   counted.
2. SOL-score validation requires a genuine optimized kernel. When `Tk ~= Tb`, the
   score collapses toward `0.5`; when `Tk < Tb` is real, the aware/blind score
   difference becomes measurable, as shown by the seven Triton rows.
