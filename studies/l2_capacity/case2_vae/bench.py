"""Benchmark Tb and Tk for VAE residual block (sol_execbench_l1_002).

Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN -> SiLU -> residual add.
B=8, C=256, H=W=128, fp16.
"""

import torch
import torch.nn as nn
import numpy as np

WARMUP = 20
ITERS = 200


class VAEResidual(nn.Module):
    def __init__(self):
        super().__init__()
        C = 256
        self.conv1 = nn.Conv2d(C, C, 3, padding=1, bias=False, dtype=torch.float16)
        self.gn1 = nn.GroupNorm(32, C, dtype=torch.float16)
        self.conv2 = nn.Conv2d(C, C, 3, padding=1, bias=False, dtype=torch.float16)
        self.gn2 = nn.GroupNorm(32, C, dtype=torch.float16)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = torch.nn.functional.silu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = torch.nn.functional.silu(out)
        return out + identity


def bench(fn, *args, name="fn", warmup=WARMUP, iters=ITERS):
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
    print(f"  {name}: median={median:.4f} ms, mean={times.mean():.4f} ms, min={times.min():.4f} ms")
    return median


def main():
    print("=" * 60)
    print("BENCHMARK: VAE Residual (sol_execbench_l1_002)")
    print("=" * 60)

    device = torch.device("cuda")
    torch.manual_seed(42)
    model = VAEResidual().to(device)
    x = torch.randn(8, 256, 128, 128, dtype=torch.float16, device=device, requires_grad=False)

    print(f"  Input shape: {x.shape}, dtype: {x.dtype}")
    print(f"  Feature map size: {x.numel() * 2 / 1024 / 1024:.1f} MB")

    # Reference output
    with torch.no_grad():
        ref_out = model(x)
    print(f"  Output shape: {ref_out.shape}")

    # Tb: eager
    print("\n[STEP 1] Reference eager (Tb)")
    Tb = bench(model, x, name="eager")

    # Tk: torch.compile
    print("\n[STEP 2] Optimized (Tk)")
    candidates = {}

    for mode_name, mode_val in [("default", "default"), ("reduce-overhead", "reduce-overhead")]:
        try:
            cf = torch.compile(model, mode=mode_val, fullgraph=False)
            _ = cf(x)
            t = bench(cf, x, name=f"torch.compile({mode_name})")
            candidates[t] = (f"torch.compile({mode_name})", cf)
        except Exception as e:
            print(f"  torch.compile({mode_name}) failed: {e}")

    if not candidates:
        print("  ❌ No optimized kernel available!")
        return None

    Tk = min(candidates.keys())
    best_name, best_fn = candidates[Tk]

    print(f"\n  Best Tk = {Tk:.4f} ms ({best_name})")

    # Verify
    with torch.no_grad():
        opt_out = best_fn(x)
    max_abs = (opt_out - ref_out).abs().max().item()
    rel_err = ((opt_out - ref_out).abs() / (ref_out.abs() + 1e-8)).max().item()
    print(f"\n[VERIFY] Max abs error: {max_abs:.6e}, Max rel error: {rel_err:.6e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Tb = {Tb:.4f} ms")
    print(f"  Tk = {Tk:.4f} ms")
    print(f"  Speedup = {Tb/Tk:.4f}x")
    print(f"  Best method: {best_name}")

    return Tb, Tk


if __name__ == "__main__":
    main()
