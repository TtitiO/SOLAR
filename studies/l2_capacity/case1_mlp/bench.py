"""Benchmark Tb (reference eager) and Tk (optimized) for the fused gated MLP.

Shapes: Llama-8B MLP, fp16, B=1, S=2048, H=4096, I=14336
Reference: gate_up GEMM -> chunk -> SiLU gate -> down GEMM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

WARMUP = 20
ITERS = 200


def run_eager(hidden_states, gate_up_weight, down_weight):
    """Reference: unfused eager execution."""
    up_states = F.linear(hidden_states, gate_up_weight)
    gate, up_states = up_states.chunk(2, dim=-1)
    up_states = up_states * (gate * torch.sigmoid(gate))
    return F.linear(up_states, down_weight)


def bench(fn, *args, name="fn", warmup=WARMUP, iters=ITERS):
    """Time a function using CUDA events."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(iters):
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times = np.array(times)
    median = np.median(times)
    mean = np.mean(times)
    print(f"  {name}: median={median:.4f} ms, mean={mean:.4f} ms, min={times.min():.4f} ms, max={times.max():.4f} ms")
    return median, mean


def verify(opt_out, ref_out, tol=1e-2):
    max_abs = (opt_out - ref_out).abs().max().item()
    rel_err = ((opt_out - ref_out).abs() / (ref_out.abs() + 1e-8)).max().item()
    print(f"  Max abs error: {max_abs:.6e}")
    print(f"  Max rel error: {rel_err:.6e}")
    if max_abs > tol:
        print(f"  ⚠️  WARNING: Large absolute error (>{tol})!")
        return False
    return True


def main():
    print("=" * 60)
    print("BENCHMARK: Fused Gated MLP (sol_execbench_l1_074)")
    print("=" * 60)

    device = torch.device("cuda")
    B, S, H = 1, 2048, 4096
    I = 14336  # intermediate_size
    twoI = 2 * I

    torch.manual_seed(42)
    gate_up_weight = nn.Parameter(torch.empty(twoI, H, dtype=torch.float16, device=device))
    down_weight = nn.Parameter(torch.empty(H, I, dtype=torch.float16, device=device))
    nn.init.normal_(gate_up_weight, std=0.02)
    nn.init.normal_(down_weight, std=0.02)

    hidden_states = torch.randn(B, S, H, dtype=torch.float16, device=device)

    # Reference output for verification
    with torch.no_grad():
        ref_out = run_eager(hidden_states, gate_up_weight, down_weight)
    print(f"  Reference output shape: {ref_out.shape}, dtype: {ref_out.dtype}")

    # --- Tb: Reference eager ---
    print("\n[STEP 1] Reference eager (Tb)")
    Tb, _ = bench(run_eager, hidden_states, gate_up_weight, down_weight, name="eager")

    # --- Tk: torch.compile variants ---
    print("\n[STEP 2] Optimized kernels (Tk)")
    candidates = {}

    # Option A: torch.compile default
    try:
        cf = torch.compile(run_eager, fullgraph=True)
        _ = cf(hidden_states, gate_up_weight, down_weight)  # trigger compile
        t, _ = bench(cf, hidden_states, gate_up_weight, down_weight, name="torch.compile(default)")
        candidates[t] = ("torch.compile(default)", cf)
    except Exception as e:
        print(f"  torch.compile(default) failed: {e}")

    # Option B: torch.compile reduce-overhead
    try:
        cf2 = torch.compile(run_eager, mode="reduce-overhead", fullgraph=True)
        _ = cf2(hidden_states, gate_up_weight, down_weight)
        t, _ = bench(cf2, hidden_states, gate_up_weight, down_weight, name="torch.compile(reduce-overhead)")
        candidates[t] = ("torch.compile(reduce-overhead)", cf2)
    except Exception as e:
        print(f"  torch.compile(reduce-overhead) failed: {e}")

    # Option C: torch.compile max-autotune
    try:
        cf3 = torch.compile(run_eager, mode="max-autotune", fullgraph=True)
        _ = cf3(hidden_states, gate_up_weight, down_weight)
        t, _ = bench(cf3, hidden_states, gate_up_weight, down_weight, name="torch.compile(max-autotune)")
        candidates[t] = ("torch.compile(max-autotune)", cf3)
    except Exception as e:
        print(f"  torch.compile(max-autotune) failed: {e}")

    if not candidates:
        print("  ❌ No optimized kernel available!")
        return None, None

    Tk = min(candidates.keys())
    best_name, best_fn = candidates[Tk]

    print(f"\n  Best Tk = {Tk:.4f} ms ({best_name})")

    # Verify numerical correctness
    print("\n[VERIFY] Numerical correctness")
    with torch.no_grad():
        opt_out = best_fn(hidden_states, gate_up_weight, down_weight)
    ok = verify(opt_out, ref_out)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Tb (reference eager)    = {Tb:.4f} ms")
    print(f"  Tk (best optimized)     = {Tk:.4f} ms")
    print(f"  Speedup                 = {Tb / Tk:.4f}x")
    print(f"  Best method: {best_name}")
    if Tk < Tb:
        print(f"  ✅ Tk < Tb (real speedup of {Tb/Tk:.4f}x)")
    else:
        print(f"  ❌ No speedup! Tk >= Tb. Need better kernel.")

    return Tb, Tk


if __name__ == "__main__":
    Tb, Tk = main()
    if Tb is not None:
        print(f"\nFINAL: Tb={Tb:.4f} ms, Tk={Tk:.4f} ms")
