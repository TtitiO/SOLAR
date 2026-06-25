<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Issue: Fused SOL ignores on-chip (L2/SRAM) capacity — `SRAM_capacity` is never used

- **Component:** `solar/analysis/graph_analyzer.py`, `solar/perf/perf_model.py`
- **Severity:** High (systematically optimistic upper bound)
- **Type:** Modeling correctness gap
- **Status:** Resolved (capacity-aware model on by default; `--no-capacity-model` reproduces the original behavior)
- **Affects models:** `fused`, `fused_prefetched` SOL estimates (the `unfused` model is unaffected)

---

## Summary

The `fused` and `fused_prefetched` Speed-of-Light (SOL) models assume **every
intermediate tensor stays on-chip (L2/SRAM) and is therefore free of DRAM
traffic**. This assumption is applied *unconditionally* — there is no check that
the working set of live intermediate tensors actually fits in the on-chip
capacity.

The architecture configs already declare the relevant capacity
(`SRAM_capacity`, `SRAM_byte_per_cycle`), but **these fields are never read by
any code in the repository**. As a result, when the intermediate working set
exceeds L2 (which is the common case for real models — activations are often
hundreds of MB while H100 L2 is 50 MB), the spilled DRAM traffic is not counted.
The predicted memory-bound runtime is too low, so the SOL upper bound is
systematically optimistic, and the error grows with working-set size.

---

## Evidence

### 1. Capacity fields are declared in every arch config

`configs/arch/H100_PCIe.yaml:5-6`
```yaml
SRAM_capacity: 52428800  # 50MB
SRAM_byte_per_cycle: 10000  # 20TB/s
```

`configs/arch/B200.yaml:5-6`
```yaml
SRAM_capacity: 201326592  # L2 192MB
SRAM_byte_per_cycle: 43691  # 64TB/s @ 1.5GHz
```

### 2. The capacity fields are referenced nowhere in the codebase

```
$ grep -rn --include="*.py" "SRAM_capacity\|SRAM_byte_per_cycle" .
(no matches)

$ grep -rn --include="*.py" "DRAM_capacity" .
(no matches)
```

`perf_model.py` reads only four architecture fields
(`perf_model.py:176-202`):

| Field | Line | Used for |
|-------|------|----------|
| `freq_GHz` | `perf_model.py:176` | cycles → ms |
| `DRAM_byte_per_cycle` | `perf_model.py:177` | memory_cycles |
| `MAC_per_cycle_*` | `perf_model.py:184-197` | compute_cycles |
| `MAC_per_cycle_fp32_sm` | `perf_model.py:202` | SM cycles (informational) |

`SRAM_capacity`, `SRAM_byte_per_cycle`, and `DRAM_capacity` never enter the
computation.

### 3. The "stays on-chip" assumption — exact locations

The assumption is not a single explicit statement; it is embedded in the
intermediate-tensor classification logic, in three places.

**(a) Per-layer: intermediate tensors are excluded from DRAM traffic**
`graph_analyzer.py:581-599`
```python
# Intermediate output elems: written to cache (fused) not DRAM
intermediate_output_elems = output_elems if output_is_intermediate else 0
...
model_output_elems = output_elems if not output_is_intermediate else 0
# Per-op model I/O: external inputs + model outputs (no intermediates)
model_io_elems = model_input_elems + model_output_elems
...
# Per-op fused elements: only non-intermediate DRAM traffic
fused_elems = int(model_io_elems)
```
The comment `written to cache (fused) not DRAM` is the origin of the
unconditional assumption: once a tensor is classified `intermediate`, it is
treated as cache-resident, regardless of its size.

**(b) Graph-level total drops all intermediate traffic**
`graph_analyzer.py:646-651`
```python
total_fused_prefetched_elems = int(
    sum(unique_external_inputs.values())
    + sum(unique_external_outputs.values())   # no intermediate term
)
total_fused_elems = total_fused_prefetched_elems
```
`total_intermediate_elems` is computed (`graph_analyzer.py:641`) and emitted to
`analysis.yaml` (`graph_analyzer.py:673`), but it is **purely diagnostic** — it
never feeds the fused totals.

**(c) Roofline consumes the under-counted byte total**
`perf_model.py:218,223`
```python
fused_mem_cycles = total_fused_bytes / dram_bw if dram_bw > 0 else 0.0
...
fused_total_cycles = max(compute_cycles, fused_mem_cycles)
```
`total_fused_bytes` excludes all intermediates, so a memory-bound graph whose
activations spill out of L2 still reports a memory cost as if they never
touched DRAM.

---

## Impact

- **Direction:** The bound is always optimistic (predicted runtime ≤ true SOL).
  Excluding real DRAM traffic can only lower `fused_mem_cycles`.
- **Magnitude scales with working set:** The larger the intermediate activations
  relative to `SRAM_capacity`, the larger the under-count. For a model whose
  intermediate working set is, e.g., 500 MB on an H100 (50 MB L2), almost all of
  that traffic is wrongly treated as free.
- **Worst for memory-bound, fusion-friendly graphs** (LLM activations, large
  feature maps, attention intermediates) — exactly the regime where `fused` is
  the recommended model (`SOL_GUIDE.md` §"Practical Guidance").
- **Silent:** there is no warning when the working set exceeds capacity; the
  config value that would catch it is never consulted.

This is distinct from, and compounds with, the gaps already documented in
`SOL_GUIDE.md` §6 (whole-graph vs per-op roofline; `fused` == `fused_prefetched`).

---

## Root cause

The fused model conflates "tensor is an intermediate (data-flow internal)" with
"tensor is resident on-chip (cost-free)". These are different properties: a
tensor is on-chip only if the **live working set at that point in execution
fits in `SRAM_capacity`**. The model never tests the second property because the
capacity input is never loaded.

---

## Suggested fix (sketch)

Introduce an L2 capacity check so that intermediate traffic exceeding on-chip
capacity spills back to DRAM and is counted.

1. **Load the capacity in the perf model.** In `perf_model.py` (near
   `perf_model.py:176-177`):
   ```python
   sram_capacity = float(arch.get("SRAM_capacity", 0))  # bytes
   ```

2. **Account for the spilled bytes** before computing `fused_mem_cycles`.
   Two options, in increasing fidelity:

   - **Coarse (graph-level):** if the peak live intermediate working set in
     bytes exceeds `SRAM_capacity`, add the excess (or the full intermediate
     traffic, as a conservative bound) to `total_fused_bytes`:
     ```python
     intermediate_bytes = total_intermediate_elems * bytes_per_element
     if intermediate_bytes > sram_capacity:
         spilled = intermediate_bytes - sram_capacity
         total_fused_bytes += spilled
         total_fused_prefetched_bytes += spilled
     ```
     This requires propagating a *peak live working set* estimate rather than
     the simple sum `total_intermediate_elems` (which is a lifetime total, not a
     simultaneously-live figure) to avoid over-counting.

   - **Precise (liveness-based):** in `graph_analyzer.py`, walk the graph in
     execution order, track the set of currently-live intermediate tensors and
     its byte size, and at each step spill the bytes that do not fit in
     `SRAM_capacity`. Accumulate spilled bytes into the fused total. The
     producer/consumer maps already built at `graph_analyzer.py:179-199` provide
     the liveness information needed.

3. **Differentiate `fused` vs `fused_prefetched`** while here: `fused` can
   assume only adjacent-op reuse (smaller effective cache window), whereas
   `fused_prefetched` can assume the full `SRAM_capacity` plus prefetch overlap.
   This would also close `SOL_GUIDE.md` §6 Gap 2.

4. **Optionally model on-chip bandwidth.** Once spill is handled, the
   non-spilled intermediate traffic could be charged at `SRAM_byte_per_cycle`
   rather than treated as entirely free, using the currently-unused second
   capacity field.

5. **Emit a diagnostic** in `perf_<arch>.yaml` (e.g. `l2_working_set_bytes`,
   `l2_capacity_bytes`, `spilled_bytes`, `fits_in_l2: true/false`) so the
   assumption is visible to the user.

---

## Acceptance criteria

- `SRAM_capacity` is read and influences `fused` / `fused_prefetched` results.
- A graph whose intermediate working set exceeds `SRAM_capacity` reports a
  strictly higher `fused` memory cost than today (regression test with a
  synthetic large-activation graph).
- A graph whose working set fits in L2 reports unchanged `fused` results.
- `perf_<arch>.yaml` exposes the working-set vs capacity comparison.
- `SOL_GUIDE.md` updated to describe the capacity-aware fused model and to
  remove/qualify the "intermediates stay in cache" claim.

---

## Resolution

Implemented as the default-on, reduction-aware capacity model for fused
intermediate spill.

**Code:**
- `solar/analysis/graph_analyzer.py` — liveness pass still computes
  `intermediate_peak_live_elements` (peak simultaneously-live full tensors via
  `[producer_step, last_consumer_step]` interval sweep), but the spill gate now
  uses `intermediate_min_tile_elements`: the peak live sum of each
  intermediate's minimal per-output-granule resident tile. Resident axes are
  derived from consumer reductions (einsum operands, keepdim reductions, and a
  small table for shape-preserving reductions such as softmax/layernorm).
- `solar/perf/perf_model.py` — reads `SRAM_capacity`, computes
  `spill_fraction = max(0, 1 − SRAM_capacity / min_tile_bytes)` and adds
  `spilled_bytes = intermediate_traffic_bytes × spill_fraction` to the
  `fused`/`fused_prefetched` DRAM totals. New `capacity_aware` parameter
  (default `True`); a `cache` diagnostic block (`sram_capacity_bytes`,
  `gate_metric: min_tile`, `intermediate_peak_live_bytes`,
  `intermediate_min_tile_bytes`, `fits_in_l2`, `spill_fraction`, `spilled_bytes`)
  is emitted in every run.
- `solar/cli/predict_perf_model.py` — `--no-capacity-model` flag reproduces the
  original optimistic numbers for before/after comparison.

**Tests:** `tests/test_perf_l2_capacity.py` — min-tile ≤ peak-live ≤ lifetime-sum;
no spill / unchanged result when the min tile fits; strictly higher `fused`
cost for a true non-tileable overflow; `--no-capacity-model` reproduces the
original; spilled `fused` cost bounded by the unfused total; softmax lower-bound
case remains spill-free.

**Measured effect:** the earlier whole-tensor peak-live gate was too pessimistic
for tileable ops and could break the lower-bound property (for example,
`sol_execbench_l1_046` reported `S=25.2`, i.e. a SOL estimate slower than the
real Triton fused softcap-softmax kernel). With the min-tile gate, the seven
`studies/l2_capacity` rows all fit (`spill_fraction = 0`) and land at `S≤1`;
`sol_execbench_l1_046` returns to `T_SOL=2.1304 ms`, `S=0.9862`.

**Remaining work:** `fused` and `fused_prefetched` still share the same spill
charge (does not yet close §6 Gap 2); non-spilled intermediate traffic is not
charged at `SRAM_byte_per_cycle`; full-extent reduction axes can still overcharge
when a single streamable reduction axis alone exceeds SRAM (streaming-block
accounting is future work).

---

## Original acceptance criteria status

- [x] `SRAM_capacity` is read and influences `fused` / `fused_prefetched`.
- [x] Overflowing min tile reports strictly higher `fused` cost (regression test).
- [x] Min tile that fits reports unchanged `fused` results.
- [x] `perf_<arch>.yaml` exposes both peak-live and min-tile capacity diagnostics.
- [x] `SOL_GUIDE.md` updated.

---

## Related

- `docs/SOL_GUIDE.md` §6 "Known Implementation Gaps" (whole-graph roofline;
  `fused` == `fused_prefetched`).
- Per-layer fields already available for a liveness pass:
  `intermediate_elements`, `model_io_elements`, `connections`
  (`graph_analyzer.py:601-635`).
