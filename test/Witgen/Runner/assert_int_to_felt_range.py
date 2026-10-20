#!/usr/bin/env python3
"""Check ASSERT cast boundaries and runtime failures in both witgen backends."""

import json
from pathlib import Path
import subprocess
import sys


def main():
    """Exercise dynamic indices without allowing constant folding to hide checks."""
    tool, source, directory = sys.argv[1:]
    template = Path(source).read_text()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    inputs = directory / "inputs.json"
    inputs.write_text('{"input": "0"}\n')
    max_index = (1 << 63) - 1
    fields = {
        "babybear": 2013265921,
        "goldilocks": 18446744069414584321,
        "bn254": 21888242871839275222246405745257275088548364400416034343698204186575808495617,
    }
    for field, prime in fields.items():
        values = {0, 1, -1, -(1 << 63), max_index, min(prime - 1, max_index)}
        if prime <= max_index:
            values.update({prime, prime + 1, 1 << 32})
        for value in sorted(values):
            for unused in (False, True):
                text = template.replace("babybear", field).replace(
                    "%offset = arith.constant 0 : index",
                    f"%offset = arith.constant {value} : index",
                )
                if unused:
                    text = text.replace("[@out] = %converted", "[@out] = %zero")
                program = directory / f"{field}-{value}-{unused}.llzk"
                program.write_text(text)
                for backend in ("interpreter", "execution-engine"):
                    result = subprocess.run(
                        [tool, str(program), "--inputs", str(inputs), f"--backend={backend}"],
                        text=True, capture_output=True,
                    )
                    context = f"{field}, index={value}, unused={unused}, {backend}"
                    if 0 <= value < prime:
                        expected = {"out": str(0 if unused else value)}
                        if result.returncode or json.loads(result.stdout) != expected:
                            raise AssertionError(f"{context}: {result.stdout}\n{result.stderr}")
                    else:
                        diagnostic = (
                            "integer does not fit in felt" if backend == "interpreter"
                            else "cast.tofelt input"
                        )
                        if result.returncode == 0 or diagnostic not in result.stdout + result.stderr:
                            raise AssertionError(f"{context}: {result.stdout}\n{result.stderr}")


if __name__ == "__main__":
    main()
