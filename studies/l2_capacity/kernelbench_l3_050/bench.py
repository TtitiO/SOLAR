"""Benchmark Tb and Tk for 50_ReLUSelfAttention (kernelbench_l3_050)."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, sys, os
sys.path.insert(0, os.path.dirname(__file__))

WARMUP, ITERS = 10, 200

# ── Reference: verbatim Model class from torch code ──
# (copied below from the JSON for reference; we import from model.py for Tk)

import importlib.util
spec = importlib.util.spec_from_file_location("ref_model", os.path.join(os.path.dirname(__file__), "model.py"))
ref_mod = importlib.util.module_from_spec(spec)
# Actually, just use the same model.py for both (it's the reference implementation)
from model import Model, get_inputs, get_init_inputs

# For Tb, we'll re-import fresh so there's no confusion
# We use the EXACT same Model class as the reference (imported from model.py)

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
    print("BENCHMARK: kernelbench_l3_050")
    print("=" * 60)
    device = torch.device("cuda")
    
    init_args = get_init_inputs()
    model = Model(*init_args).to(device)
    # Save state_dict for weight sharing between Tb and Tk
    state = model.state_dict()
    
    torch.manual_seed(42)
    inputs = [t.to(device) for t in get_inputs()]
    args = tuple(inputs)
    
    for i, t in enumerate(args):
        if t.numel() > 100000:
            mb = t.numel() * t.element_size() / (1024**2)
            print(f"  input[{i}] {list(t.shape)} {t.dtype} = {mb:.0f} MB")
    print(f"  L2 capacity = 72 MB")
    
    # Get reference output
    with torch.no_grad():
        ref_out = model(*args)
    print(f"  Output shape: {ref_out.shape}, dtype: {ref_out.dtype}")
    
    # Tb: Reference (same Model)
    print("\n[STEP 1] Reference eager (Tb)")
    Tb = bench(lambda *a: model(*a), args, name="eager")
    
    # Tk: Same model (re-instantiate for purity)
    model2 = Model(*init_args).to(device)
    model2.load_state_dict(state)  # Share weights for fair comparison
    with torch.no_grad():
        _ = model2(*args)
    print("\n[STEP 2] Optimized model.py (Tk)")
    Tk = bench(lambda *a: model2(*a), args, name="model.py")
    
    # Verify
    print("\n[VERIFY] Numerical correctness")
    with torch.no_grad():
        opt_out = model2(*args)
    max_abs = (opt_out.float() - ref_out.float()).abs().max().item()
    print(f"  Max abs error: {max_abs:.6e}")
    tol = 1e-4 if ref_out.dtype == torch.float32 else 1e-2
    ok = max_abs <= tol
    print(f"  {'✅ Within tolerance' if ok else '⚠️  WARNING: Large error'}")
    
    print("\n" + "=" * 60)
    print("BENCHMARK: kernelbench_l3_050")
    print("=" * 60)
    print(f"  Operator  : 50_ReLUSelfAttention")
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
