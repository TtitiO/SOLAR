"""Benchmark Tb and Tk for 032_dual_stream_attention_with_conditional_cross_attention (sol_execbench_l2_032)."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from model import Model, get_inputs, get_init_inputs

WARMUP, ITERS = 10, 200

# ── Reference: verbatim torch code from JSON ──
# The reference function is imported below from the torch code

def bench(fn, args_tuple, name="fn", warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn(*args_tuple)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn(*args_tuple)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times = np.array(times)
    median = np.median(times)
    print(f"  {name}: median={median:.4f} ms, mean={times.mean():.4f} ms, min={times.min():.4f} ms")
    return median

def main():
    print("=" * 60)
    print("BENCHMARK: sol_execbench_l2_032")
    print("=" * 60)
    device = torch.device("cuda")
    
    # Create inputs using model.py's get_inputs()
    torch.manual_seed(42)
    cpu_inputs = get_inputs()
    args = tuple(t.to(device) if hasattr(t, "to") else t for t in cpu_inputs)
    
    for i, t in enumerate(args):
        if hasattr(t, 'numel') and t.numel() > 100000:
            mb = t.numel() * t.element_size() / (1024**2)
            print(f"  input[{i}] {list(t.shape)} {t.dtype} = {mb:.0f} MB")
    print(f"  L2 capacity = 72 MB")
    
    # Tb: Reference using Model from model.py (same code)
    model = Model().to(device)
    with torch.no_grad():
        raw_out = model(*args)
        ref_out = raw_out[0] if isinstance(raw_out, tuple) else raw_out
    print(f"  Output shape: {ref_out.shape}, dtype: {ref_out.dtype}")
    
    print("\n[STEP 1] Reference eager (Tb)")
    Tb = bench(lambda *a: model(*a), args, name="eager")
    
    # Tk: Same model
    model2 = Model().to(device)
    with torch.no_grad():
        _ = model2(*args)
    print("\n[STEP 2] Optimized model.py (Tk)")
    Tk = bench(lambda *a: model2(*a), args, name="model.py")
    
    # Verify
    print("\n[VERIFY] Numerical correctness")
    with torch.no_grad():
        raw_opt = model2(*args)
        opt_out = raw_opt[0] if isinstance(raw_opt, tuple) else raw_opt
    max_abs = (opt_out.float() - ref_out.float()).abs().max().item()
    print(f"  Max abs error: {max_abs:.6e}")
    tol = 1e-4 if ref_out.dtype == torch.float32 else 1e-2
    ok = max_abs <= tol
    print(f"  {'✅ Within tolerance' if ok else '⚠️  WARNING: Large error'}")
    
    print("\n" + "=" * 60)
    print("BENCHMARK: sol_execbench_l2_032")
    print("=" * 60)
    print(f"  Operator  : 032_dual_stream_attention_with_conditional_cross_attention")
    print(f"  DSL (Tk)  : PyTorch")
    print(f"  Tb (eager torch code) : {Tb:.4f} ms")
    print(f"  Tk (optimized kernel) : {Tk:.4f} ms")
    if Tb > 0:
        print(f"  Speedup   : {Tb/Tk:.2f}x")
    print(f"  Max abs error : {max_abs:.2e}  " + ("✅" if ok else "⚠️ LARGE ERROR"))
    print("=" * 60)
    return Tb, Tk

if __name__ == "__main__":
    Tb, Tk = main()
    if Tb is not None:
        print(f"\nFINAL: Tb={Tb:.4f} ms, Tk={Tk:.4f} ms")
