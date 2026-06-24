# L2 SRAM Capacity Model Validation Study

Three operators tested against Solar's `--no-capacity-model` toggle
to measure whether the L2/SRAM capacity model meaningfully changes
the SOL score. See the full report at `../../docs/L2_CAPACITY_VALIDATION.md`.

## Cases

| Case | Directory | Operator | Δ score |
|------|-----------|----------|---------|
| 1 | `case1_mlp/` | Fused Gated MLP (l1_074) | 0.000 (compute-bound) |
| 2 | `case2_vae/` | VAE Residual (l1_002) | 0.000 (compute-bound) |
| 3 | `case3_attention/` | Full Attention (l1_082) | **+0.114 (bottleneck flips)** |

## Reproducing

Each case directory is self-contained:

```bash
# Case 3 (attention — shows the effect):
cd studies/l2_capacity/case3_attention

# Benchmark
python3 bench.py

# SOLAR pipeline
python -m solar.cli.process_model --model-file model.py --output-dir output/graph
python -m solar.cli.toeinsum_model --graph-path output/graph/pytorch_graph.yaml --output-dir output/einsum --no-copy-graph
python -m solar.cli.analyze_model --einsum-graph-path output/einsum/einsum_graph_renamed.yaml --output-dir output/analysis --precision fp16
python -m solar.cli.predict_perf_model --analysis-path output/analysis/analysis.yaml --output-dir output/perf_aware --arch-config configs/arch/RTX4090.yaml --precision fp16
python -m solar.cli.predict_perf_model --analysis-path output/analysis/analysis.yaml --output-dir output/perf_blind --arch-config configs/arch/RTX4090.yaml --precision fp16 --no-capacity-model
```

## Key Finding

The capacity model only changes T_SOL when the spill pushes memory cycles
above compute cycles. This requires operators where intermediate traffic
dwarfs model I/O (e.g., attention with O(S^2) scores). GEMM-heavy operators
are compute-bound regardless of spill size.
