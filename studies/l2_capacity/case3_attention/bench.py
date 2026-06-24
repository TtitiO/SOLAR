"""Benchmark Tb and Tk for full attention (sol_execbench_l1_082).

Full attention: QKV_proj -> QK LN -> QK^T -> softmax -> AV -> o_proj
H=24, D=64, dim=1536, B=1, S=2048, fp16 (QK^T/AV in fp32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

WARMUP = 10
ITERS = 100


def run_attention(hidden_states, qkv_weight, qkv_bias,
                  q_norm_weight, q_norm_bias, k_norm_weight, k_norm_bias,
                  out_proj_weight, out_proj_bias,
                  num_heads=24, head_dim=64, eps=1e-5):
    """Reference unfused attention."""
    B, S, dim = hidden_states.shape

    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    q, k, v = qkv.split(dim, dim=-1)
    q = q.view(B, S, num_heads, head_dim).transpose(1, 2)
    k = k.view(B, S, num_heads, head_dim).transpose(1, 2)
    v = v.view(B, S, num_heads, head_dim).transpose(1, 2)

    q = F.layer_norm(q, [head_dim], weight=q_norm_weight, bias=q_norm_bias, eps=eps)
    k = F.layer_norm(k, [head_dim], weight=k_norm_weight, bias=k_norm_bias, eps=eps)

    q_fp32 = q.float()
    k_fp32 = k.float()
    attn_scores = torch.matmul(q_fp32, k_fp32.transpose(-2, -1))
    scale = head_dim ** -0.5
    attn_scores = attn_scores * scale
    attn_probs = torch.softmax(attn_scores, dim=-1)

    v_fp32 = v.float()
    attn_output = torch.matmul(attn_probs, v_fp32)
    attn_output = attn_output.to(torch.float16)
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, dim)

    output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    return output


class AttentionModel(nn.Module):
    """Wrapped as nn.Module for torch.compile."""
    def __init__(self, num_heads=24, head_dim=64, eps=1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps

    def forward(self, hidden_states, qkv_weight, qkv_bias,
                q_norm_weight, q_norm_bias, k_norm_weight, k_norm_bias,
                out_proj_weight, out_proj_bias):
        return run_attention(hidden_states, qkv_weight, qkv_bias,
                             q_norm_weight, q_norm_bias, k_norm_weight, k_norm_bias,
                             out_proj_weight, out_proj_bias,
                             self.num_heads, self.head_dim, self.eps)


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
    print("BENCHMARK: Full Attention (sol_execbench_l1_082)")
    print("=" * 60)

    device = torch.device("cuda")
    B, S = 1, 2048
    dim = 1536
    three_dim = 3 * dim
    head_dim = 64

    # Report sizes
    attn_score_mb = B * 24 * S * S * 4 / (1024 * 1024)
    print(f"  attn_scores [1,24,{S},{S}] fp32 = {attn_score_mb:.0f} MB")
    print(f"  L2 capacity = 72 MB")

    # Create inputs on GPU
    torch.manual_seed(42)
    hidden_states = torch.randn(B, S, dim, dtype=torch.float16, device=device)
    qkv_weight = torch.randn(three_dim, dim, dtype=torch.float16, device=device)
    qkv_bias = torch.randn(three_dim, dtype=torch.float16, device=device)
    q_norm_weight = torch.ones(head_dim, dtype=torch.float16, device=device)
    q_norm_bias = torch.zeros(head_dim, dtype=torch.float16, device=device)
    k_norm_weight = torch.ones(head_dim, dtype=torch.float16, device=device)
    k_norm_bias = torch.zeros(head_dim, dtype=torch.float16, device=device)
    out_proj_weight = torch.randn(dim, dim, dtype=torch.float16, device=device)
    out_proj_bias = torch.zeros(dim, dtype=torch.float16, device=device)

    args = (hidden_states, qkv_weight, qkv_bias,
            q_norm_weight, q_norm_bias, k_norm_weight, k_norm_bias,
            out_proj_weight, out_proj_bias)

    # Reference output
    with torch.no_grad():
        ref_out = run_attention(*args)
    print(f"  Output shape: {ref_out.shape}, dtype: {ref_out.dtype}")

    # --- Tb: Reference eager ---
    print("\n[STEP 1] Reference eager (Tb)")
    Tb = bench(run_attention, args, name="eager")

    # --- Tk: torch.compile ---
    print("\n[STEP 2] Optimized (Tk)")
    candidates = {}

    model = AttentionModel().to(device)

    for mode_name, mode_val in [("default", "default"), ("reduce-overhead", "reduce-overhead")]:
        try:
            cf = torch.compile(model, mode=mode_val, fullgraph=False)
            _ = cf(*args)
            t = bench(cf, args, name=f"torch.compile({mode_name})")
            candidates[t] = (f"torch.compile({mode_name})", cf)
        except Exception as e:
            print(f"  torch.compile({mode_name}) failed: {e}")

    if not candidates:
        print("  ❌ No optimized kernel available!")
        return None

    Tk = min(candidates.keys())
    best_name, best_fn = candidates[Tk]

    print(f"\n  Best Tk = {Tk:.4f} ms ({best_name})")

    # Verify numerical correctness
    print("\n[VERIFY] Numerical correctness")
    with torch.no_grad():
        opt_out = best_fn(*args)
    max_abs = (opt_out - ref_out).abs().max().item()
    rel_err = ((opt_out - ref_out).abs() / (ref_out.abs() + 1e-8)).max().item()
    print(f"  Max abs error: {max_abs:.6e}")
    print(f"  Max rel error: {rel_err:.6e}")
    if max_abs > 0.1:
        print(f"  ⚠️  WARNING: Large absolute error!")
    else:
        print(f"  ✅ Within tolerance")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Tb = {Tb:.4f} ms")
    print(f"  Tk = {Tk:.4f} ms")
    print(f"  Speedup = {Tb/Tk:.4f}x")
    print(f"  Best method: {best_name}")

    return Tb, Tk


if __name__ == "__main__":
    Tb, Tk = main()
    if Tb is not None:
        print(f"\nFINAL: Tb={Tb:.4f} ms, Tk={Tk:.4f} ms")
