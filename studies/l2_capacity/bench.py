"""Shared CLI for L2-capacity baseline-vs-solution benchmarks.

Per-problem ``bench.py`` files are kept as tiny compatibility wrappers because
the validation study contract calls for three files in each problem directory.
The benchmark logic lives in ``bench_utils.py`` and this shared CLI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench_utils import DEFAULT_ITERS, DEFAULT_WARMUP, run_benchmark


ROOT = Path(__file__).resolve().parent


def _problem_dirs() -> list[Path]:
    return sorted(path for path in ROOT.iterdir() if (path / "model.py").is_file())


def _run_one(case_dir: Path, warmup: int, iters: int) -> tuple[int, dict]:
    pid = case_dir.name
    try:
        result = run_benchmark(case_dir, pid, warmup=warmup, iters=iters)
    except Exception as exc:
        return 1, {"pid": pid, "status": "fail", "error": str(exc)}
    return 0, {"status": "pass", **result}


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark one or all L2-capacity problems")
    parser.add_argument("problem", nargs="?", help="problem directory name, or omit with --all")
    parser.add_argument("--all", action="store_true", help="run every problem directory under studies/l2_capacity")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--keep-going", action="store_true", help="with --all, continue after failures")
    args = parser.parse_args()

    if args.all:
        dirs = _problem_dirs()
    elif args.problem:
        dirs = [ROOT / args.problem]
    else:
        parser.error("provide a problem name or --all")

    results = []
    exit_code = 0
    for case_dir in dirs:
        if not (case_dir / "model.py").is_file():
            item = {"pid": case_dir.name, "status": "fail", "error": f"missing model.py in {case_dir}"}
            code = 1
        else:
            code, item = _run_one(case_dir, args.warmup, args.iters)
        results.append(item)
        print(json.dumps(item, indent=2, sort_keys=True))
        if code != 0:
            exit_code = code
            if not args.keep_going:
                break

    if args.all:
        summary = {
            "status": "pass" if exit_code == 0 else "fail",
            "passed": sum(1 for item in results if item.get("status") == "pass"),
            "failed": sum(1 for item in results if item.get("status") != "pass"),
            "total": len(results),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
