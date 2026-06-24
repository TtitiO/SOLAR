"""Shared benchmark harness helpers for studies/l2_capacity.

This module intentionally uses plain eager PyTorch only.  It does not call
``torch.compile`` and does not assume the candidate is a real optimized kernel:
declared reference/honest-failure artifacts are rejected before scoring.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch


DEFAULT_WARMUP = 20
DEFAULT_ITERS = 200


class BenchFailure(RuntimeError):
    """Raised for clear non-scoring benchmark failures."""


def import_module_from_path(path: str | os.PathLike[str], module_name: str) -> Any:
    """Import ``path`` under a caller-provided unique module name."""
    path = str(Path(path).resolve())
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise BenchFailure(f"unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    old = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if old is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = old
        raise
    return module


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [_to_device(v, device) for v in value]
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    return value


def instantiate_model(module: Any, device: torch.device) -> torch.nn.Module:
    if not hasattr(module, "Model") and hasattr(module, "run"):
        return FunctionModule(module.run).to(device)
    if not hasattr(module, "Model"):
        raise BenchFailure(f"{module.__name__} has no Model class")
    init_inputs = _as_list(module.get_init_inputs() if hasattr(module, "get_init_inputs") else [])
    model = module.Model(*init_inputs)
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "to"):
        model.to(device)
    return model


class FunctionModule(torch.nn.Module):
    def __init__(self, fn: Any):
        super().__init__()
        self.fn = fn

    def forward(self, *args: Any) -> Any:
        return self.fn(*args)


def get_baseline_inputs(module: Any, device: torch.device) -> tuple[Any, ...]:
    if not hasattr(module, "get_inputs"):
        raise BenchFailure(f"{module.__name__} has no get_inputs()")
    return tuple(_to_device(v, device) for v in _as_list(module.get_inputs()))


def get_shared_inputs(baseline_module: Any, solution_module: Any, device: torch.device) -> tuple[Any, ...]:
    if hasattr(baseline_module, "get_inputs"):
        return get_baseline_inputs(baseline_module, device)
    if hasattr(solution_module, "get_inputs"):
        return tuple(_to_device(v, device) for v in _as_list(solution_module.get_inputs()))
    raise BenchFailure(f"neither {baseline_module.__name__} nor {solution_module.__name__} has get_inputs()")


def share_state_dict(baseline: Any, candidate: Any) -> str:
    """Copy baseline state into candidate when both expose compatible state_dicts."""
    if not (hasattr(baseline, "state_dict") and hasattr(candidate, "load_state_dict")):
        return "not-applicable"
    base_state = baseline.state_dict()
    if not base_state:
        return "empty"
    try:
        candidate.load_state_dict(base_state, strict=True)
        return "strict"
    except Exception as exc:
        try:
            result = candidate.load_state_dict(base_state, strict=False)
            missing = list(getattr(result, "missing_keys", []))
            unexpected = list(getattr(result, "unexpected_keys", []))
            return f"non-strict missing={missing} unexpected={unexpected}"
        except Exception as exc2:
            raise BenchFailure(f"unable to share/copy state_dict: strict={exc!r}; non_strict={exc2!r}") from exc2


def _tensor_tolerances(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    dtype = a.dtype if torch.is_floating_point(a) else b.dtype
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-2, 1e-2
    if dtype == torch.float32:
        return 1e-4, 1e-4
    if dtype == torch.float64:
        return 1e-7, 1e-7
    return 0.0, 0.0


def compare_outputs(actual: Any, expected: Any, path: str = "output") -> dict[str, Any]:
    """Recursively compare tensors/tuples/lists/dicts with dtype-aware tolerances."""
    if torch.is_tensor(actual) and torch.is_tensor(expected):
        if actual.shape != expected.shape:
            return {"ok": False, "path": path, "reason": f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"}
        if actual.dtype != expected.dtype:
            # Different floating dtypes are allowed if values are within tolerance.
            if not (torch.is_floating_point(actual) and torch.is_floating_point(expected)):
                return {"ok": False, "path": path, "reason": f"dtype {actual.dtype} != {expected.dtype}"}
        atol, rtol = _tensor_tolerances(actual, expected)
        a = actual.detach()
        b = expected.detach()
        if torch.is_floating_point(a) or torch.is_floating_point(b):
            ok = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol, equal_nan=True)
            max_abs = (a.float() - b.float()).abs().max().item() if a.numel() else 0.0
        else:
            ok = torch.equal(a, b)
            max_abs = 0.0
        return {"ok": bool(ok), "path": path, "max_abs": max_abs, "atol": atol, "rtol": rtol}
    if isinstance(actual, (tuple, list)) and isinstance(expected, type(actual)):
        if len(actual) != len(expected):
            return {"ok": False, "path": path, "reason": f"length {len(actual)} != {len(expected)}"}
        worst = {"ok": True, "path": path, "max_abs": 0.0}
        for i, (av, ev) in enumerate(zip(actual, expected)):
            result = compare_outputs(av, ev, f"{path}[{i}]")
            if not result["ok"]:
                return result
            if result.get("max_abs", 0.0) >= worst.get("max_abs", 0.0):
                worst = result
        return worst
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            return {"ok": False, "path": path, "reason": "dict keys differ"}
        worst = {"ok": True, "path": path, "max_abs": 0.0}
        for key in sorted(actual):
            result = compare_outputs(actual[key], expected[key], f"{path}.{key}")
            if not result["ok"]:
                return result
            if result.get("max_abs", 0.0) >= worst.get("max_abs", 0.0):
                worst = result
        return worst
    return {"ok": actual == expected, "path": path, "reason": "non-tensor equality"}


def cuda_event_median(fn: Any, args: Iterable[Any], warmup: int = DEFAULT_WARMUP, iters: int = DEFAULT_ITERS) -> float:
    if not torch.cuda.is_available():
        raise BenchFailure("CUDA is required for CUDA-event timing")
    args = tuple(args)
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args)
        torch.cuda.synchronize()
        times: list[float] = []
        start: Any = torch.cuda.Event(enable_timing=True)
        end: Any = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start.record()
            fn(*args)
            end.record()
            torch.cuda.synchronize()
            times.append(float(start.elapsed_time(end)))
    return float(statistics.median(times))


def _declared_failure(module: Any) -> str | None:
    if bool(getattr(module, "DECLARED_UNBEATABLE", False)):
        return str(getattr(module, "NO_OPTIMIZED_KERNEL", "solution declares no optimized kernel"))
    if getattr(module, "METHOD_LABEL", "") in {"honest-failure-reference", "reference", "baseline"}:
        return str(getattr(module, "NO_OPTIMIZED_KERNEL", "solution is a reference artifact, not an optimized kernel"))
    return None


def run_benchmark(case_dir: str | os.PathLike[str], pid: str, method_label: str | None = None,
                  warmup: int = DEFAULT_WARMUP, iters: int = DEFAULT_ITERS) -> dict[str, Any]:
    case_dir = Path(case_dir).resolve()
    baseline_path = case_dir / "model.py"
    solution_path = case_dir / "solution.py"
    baseline_mod = import_module_from_path(baseline_path, f"l2cap_{pid}_baseline_{os.getpid()}_{time.time_ns()}")
    solution_mod = import_module_from_path(solution_path, f"l2cap_{pid}_solution_{os.getpid()}_{time.time_ns()}")

    declared = _declared_failure(solution_mod)
    label = method_label or str(getattr(solution_mod, "METHOD_LABEL", "solution"))
    baseline_label = str(getattr(baseline_mod, "METHOD_LABEL", "eager"))
    if declared:
        raise BenchFailure(f"{pid}: solution '{label}' declares honest failure/no optimized kernel: {declared}")
    if getattr(solution_mod, "Model", None) is getattr(baseline_mod, "Model", object()):
        raise BenchFailure(f"{pid}: solution Model is identical to baseline Model; refusing to score")

    device = torch.device("cuda")
    baseline = instantiate_model(baseline_mod, device)
    candidate = instantiate_model(solution_mod, device)
    state_status = share_state_dict(baseline, candidate)
    inputs = get_shared_inputs(baseline_mod, solution_mod, device)

    with torch.no_grad():
        expected = baseline(*inputs)
        actual = candidate(*inputs)
    comparison = compare_outputs(actual, expected)
    if not comparison["ok"]:
        raise BenchFailure(f"{pid}: correctness failed at {comparison}")

    tb = cuda_event_median(baseline, inputs, warmup=warmup, iters=iters)
    tk = cuda_event_median(candidate, inputs, warmup=warmup, iters=iters)
    if not (tk < tb):
        raise BenchFailure(f"{pid}: hard gate failed: Tk={tk:.6f} ms is not < Tb={tb:.6f} ms")
    return {
        "pid": pid,
        "baseline_label": baseline_label,
        "method_label": label,
        "baseline_file": str(baseline_path),
        "solution_file": str(solution_path),
        "different_files": baseline_path.resolve() != solution_path.resolve(),
        "Tb_ms": tb,
        "Tk_ms": tk,
        "speedup": tb / tk if tk > 0 else math.inf,
        "state_dict": state_status,
        "correctness": comparison,
        "warmup": warmup,
        "iters": iters,
    }


def main(case_dir: str | os.PathLike[str], pid: str, method_label: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Benchmark {pid} baseline vs solution")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    args = parser.parse_args()
    try:
        result = run_benchmark(case_dir, pid, method_label=method_label, warmup=args.warmup, iters=args.iters)
    except Exception as exc:
        print(json.dumps({"pid": pid, "status": "fail", "error": str(exc)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps({"status": "pass", **result}, indent=2, sort_keys=True))
    return 0
