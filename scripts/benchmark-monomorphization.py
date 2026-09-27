#!/usr/bin/env python3
"""Run serial, dump-free frontend and definition-specialization benchmarks.

Transient frontend IR is deleted after each case. Logs and CSV/JSON summaries are
kept in the explicitly selected output directory, which must be outside Git.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import tempfile
import time


def measured(command, log, timeout):
    """Measure one process tree with the platform time utility and a hard timeout."""
    timer = ["/usr/bin/time", "-l" if platform.system() == "Darwin" else "-v"]
    started = time.monotonic()
    with log.open("w") as stream:
        process = subprocess.Popen(timer + command, stdout=stream, stderr=stream,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            status = "ok" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            status = "timeout"
    elapsed = time.monotonic() - started
    text = log.read_text(errors="replace")
    if platform.system() == "Darwin":
        rss = re.search(r"(\d+)\s+maximum resident set size", text)
        peak = int(rss[1]) if rss else None
    else:
        rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
        peak = int(rss[1]) * 1024 if rss else None
    metrics = dict(re.findall(r"(specializations|operations|input_operations|specialized_operations|pass_ms|evaluation_ms|instances|signals|iterations|calls|constraints|emitted_operations|dag_nodes|retained_operations)=(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", text))
    plain = re.sub(r"\x1b\[[0-9;]*m", "", text)
    diagnostic = next((line.strip() for line in plain.splitlines()
                       if "error:" in line or "Failed to" in line or "IR is invalid" in line), "")
    return {"status": status, "wall_seconds": elapsed, "peak_rss_bytes": peak,
            "command": command, "diagnostic": diagnostic, **metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--llzk-opt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("monomorphization-benchmarks.json"))
    parser.add_argument("--filter", default=".*", help="Regex over benchmark names")
    parser.add_argument("--tier", choices=["small", "scale"])
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["monomorphize", "parse", "evaluate"], default="monomorphize")
    parser.add_argument("--build-type", default="unknown", help="Build type of the supplied tools, recorded in summaries")
    parser.add_argument("--plaintext", action="store_true", help="Use temporary textual frontend IR for bytecode compatibility")
    parser.add_argument("--frontend-mode", choices=["templated", "concrete"], default="templated")
    parser.add_argument("--jobs", type=int, choices=[1], default=1)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    output = args.output.resolve()
    if output.is_relative_to(repo):
        parser.error("--output must be outside the repository")
    output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=repo).returncode != 0
    rows = []
    for case in json.loads(args.manifest.read_text()):
        if not re.search(args.filter, case["name"]) or (args.tier and case["tier"] != args.tier):
            continue
        with tempfile.TemporaryDirectory(prefix="llzk-mono-") as temporary:
            frontend_command = [str(args.frontend.resolve()), str((args.corpus / case["source"]).resolve()),
                                "--llzk", args.frontend_mode, "--llzk_strip_debug_info", "-o", temporary]
            if args.plaintext:
                frontend_command.append("--llzk_plaintext")
            row = {"name": case["name"], "tier": case["tier"], "revision": revision,
                   "phase": args.phase, "jobs": args.jobs, "frontend_mode": args.frontend_mode, "build_type": args.build_type,
                   "dirty": dirty, "llzk_opt": str(args.llzk_opt.resolve()),
                   "frontend_tool": str(args.frontend.resolve())}
            row["frontend"] = measured(frontend_command, output / f'{case["name"]}.frontend.log', args.timeout)
            inputs = list(Path(temporary).rglob("*.llzk"))
            if row["frontend"]["status"] == "ok" and len(inputs) == 1:
                command = [str(args.llzk_opt.resolve()), str(inputs[0]), "-o", os.devnull,
                           "--mlir-print-op-on-diagnostic=false"]
                if args.phase in ("monomorphize", "evaluate"):
                    command += ["--llzk-monomorphize=report=true"]
                if args.phase == "evaluate":
                    command += ["--llzk-evaluate-constraints=report=true"]
                row["pass"] = measured(command, output / f'{case["name"]}.pass.log', args.timeout)
            else:
                row["pass"] = {"status": "frontend-failed"}
            rows.append(row)
            print(f'{case["name"]}: {row["pass"]["status"]}', flush=True)
            (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    flat = [{"name": r["name"], "tier": r["tier"], "revision": r["revision"],
             "phase": r["phase"], "frontend_mode": r["frontend_mode"],
             "build_type": r["build_type"], "dirty": r["dirty"], **{k: v for k, v in r["pass"].items() if k != "command"}}
            for r in rows]
    with (output / "summary.csv").open("w", newline="") as stream:
        fields = list(dict.fromkeys(key for row in flat for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    return int(any(row["pass"]["status"] != "ok" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
