#!/usr/bin/env python3

# Program: lowering_pass_eval.py
# Description: Run llzk-opt lowering passes over a set of LLZK IR files
#   and write timing results to a CSV file.
#
# Required Programs:
#   - python3: For running this script
#   - llzk-opt: For performing lowering passes
#
# Usage:
#   scripts/lowering_pass_eval.py \
#       --src-dir PATH \
#       [--dest-dir PATH] \
#       [--lvl {1,2,3,4,5,6}] \
#       [--llzk-opt-bin PATH] \
#       [--timeout SECONDS] \
#       [--nthreads N] \
#       [--output-csv PATH]
#
# Example:
#   scripts/lowering_pass_eval.py --src-dir ~/circom-benchmarks/llzk-outputs --dest-dir lowering_eval_out --lvl 6 --timeout 30

import argparse
import csv
import datetime
import multiprocessing
import os
import subprocess
import time
from pathlib import Path
from typing import List, Tuple

Task = Tuple[str, List[str], Path, Path, float]

def get_pass_args(lvl: int) -> List[str]:
    """Return llzk-opt pass arguments for the selected lowering level."""
    pass_by_level = {
        6: ["-llzk-full-r1cs-lowering"],
        5: ["-llzk-full-poly-lowering"],
        4: ["-llzk-full-struct-inlining"],
        3: ["--pass-pipeline=builtin.module(llzk-flatten,llzk-pod-to-scalar,llzk-array-to-scalar)"],
        2: ["--pass-pipeline=builtin.module(llzk-flatten,llzk-pod-to-scalar)"],
        1: ["-llzk-flatten"],
    }
    return pass_by_level[lvl]

def get_llzk_inputs(llzk_files_dir: Path) -> List[Path]:
    """Return sorted .llzk files anywhere under llzk_files_dir."""
    if not llzk_files_dir.exists():
        raise FileNotFoundError(f"{llzk_files_dir} does not exist")

    return sorted(path for path in llzk_files_dir.rglob("*.llzk") if path.is_file())

def _run_task_unpack(packed: Task):
    return run_task(*packed)

def run_task(
    benchmark_name: str,
    args: List[str],
    stdout_path: Path,
    stderr_path: Path,
    timeout: float,
):
    start = time.perf_counter()

    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start

        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")

        if proc.returncode == 0:
            return (
                benchmark_name,
                "success",
                f"{elapsed:.6f}",
                proc.returncode,
                "",
                str(stdout_path),
                str(stderr_path),
            )

        error_message = proc.stderr.strip()[:400]
        return (
            benchmark_name,
            "error",
            f"{elapsed:.6f}",
            proc.returncode,
            error_message,
            str(stdout_path),
            str(stderr_path),
        )

    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start

        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")

        return (
            benchmark_name,
            "timeout",
            f"{elapsed:.6f}",
            "",
            "timeout",
            str(stdout_path),
            str(stderr_path),
        )

def run_llzk_lowering(
    llzk_inputs: List[Path],
    llzk_files_dir: Path,
    llzk_opt_bin: Path,
    dest_dir: Path,
    lvl: int,
    timeout: float,
    nthreads: int,
    output_csv: Path,
):
    results = []
    success_cnt = 0
    error_cnt = 0
    timeout_cnt = 0

    task_args: List[Task] = []
    pass_args = get_pass_args(lvl)

    for llzk_file in llzk_inputs:
        benchmark_name = os.path.relpath(llzk_file, llzk_files_dir)
        rel_parent = Path(benchmark_name).parent
        subdir = dest_dir / rel_parent
        subdir.mkdir(parents=True, exist_ok=True)
        name = llzk_file.stem

        output_path = subdir / f"{name}.lowered.llzk"
        stdout_path = subdir / f"{name}.llzk-opt.stdout.txt"
        stderr_path = subdir / f"{name}.llzk-opt.stderr.txt"

        args = [
            str(llzk_opt_bin),
            *pass_args,
            "-o",
            str(output_path),
            str(llzk_file),
        ]

        task_args.append((benchmark_name, args, stdout_path, stderr_path, timeout))

    if nthreads == 1:
        for benchmark_name, args, stdout_path, stderr_path, _ in task_args:
            print(f"Running {benchmark_name}")
            result = run_task(benchmark_name, args, stdout_path, stderr_path, timeout)
            results.append(result)
            print(f"Exit condition: {result[1]}")
    else:
        total = len(task_args)
        print(f"Launching {total} llzk-opt tasks.")
        next_milestone = 10

        with multiprocessing.Pool(nthreads) as pool:
            for i, result in enumerate(pool.imap_unordered(_run_task_unpack, task_args), start=1):
                results.append(result)

                pct = i * 100 // total
                if pct >= next_milestone:
                    print(f"Progress: {i}/{total} ({pct}%) complete")
                    next_milestone += 10

    results.sort()

    for _, cause, _, _, _, _, _ in results:
        success_cnt += 1 if cause == "success" else 0
        error_cnt += 1 if cause == "error" else 0
        timeout_cnt += 1 if cause == "timeout" else 0

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Benchmark",
                "Result",
                "Time Seconds",
                "Return Code",
                "Error Message",
                "Stdout Path",
                "Stderr Path",
            ]
        )
        writer.writerows(results)

    print(f"success: {success_cnt}, errored: {error_cnt}, timeout: {timeout_cnt}")
    return output_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run llzk-opt lowering passes over a set of LLZK IR files and collect timing results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--src-dir",
        required=True,
        type=Path,
        help="Path to the directory containing LLZK IR files to evaluate.",
    )
    parser.add_argument(
        "--dest-dir",
        default=Path("lowering_eval_out"),
        type=Path,
        help="Path to the output directory for lowered files and logs.",
    )
    parser.add_argument(
        "--lvl",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=6,
        help="Lowering level to run (1=flatten ... 6=full-r1cs-lowering).",
    )
    parser.add_argument(
        "--llzk-opt-bin",
        default=Path("build/bin/llzk-opt"),
        type=Path,
        help="Path to the llzk-opt binary.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30,
        help="Per-file timeout in seconds.",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of jobs to run at once.",
    )
    parser.add_argument(
        "--output-csv",
        default=Path("llzk_lowering_results.csv"),
        type=Path,
        help="CSV file to write.",
    )

    args = parser.parse_args()

    start = time.time()

    print(f"{args.src_dir = }")
    print(f"{args.dest_dir = }")
    print(f"{args.lvl = }")
    print(f"{args.llzk_opt_bin = }")

    llzk_inputs = get_llzk_inputs(args.src_dir)
    print(f"Found {len(llzk_inputs)} .llzk files.")

    if not llzk_inputs:
        raise SystemExit("No .llzk files found.")

    output_path = run_llzk_lowering(
        llzk_inputs=llzk_inputs,
        llzk_files_dir=args.src_dir,
        llzk_opt_bin=args.llzk_opt_bin,
        dest_dir=args.dest_dir,
        lvl=args.lvl,
        timeout=args.timeout,
        nthreads=args.nthreads,
        output_csv=args.output_csv,
    )

    elapsed = datetime.timedelta(seconds=time.time() - start)
    print(f"Wrote results to: {output_path}")
    print(f"Total benchmark execution time: {elapsed}")
