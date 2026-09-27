#!/usr/bin/env python3
"""Check a scalar Poseidon evaluator result against native Circom witness output.

This intentionally accepts only the field-constant/add/sub/mul equation subset.
It solves directed signal equations from concrete main inputs, requires every
signal to be determined and every equation to hold, compares the public output,
and checks that changing each solved signal is detected. This is a deterministic
semantic regression, not a proof of equivalence for arbitrary inputs.
"""
import argparse
import ast
import json
from pathlib import Path
import re
import struct


def witness(path):
    data = Path(path).read_bytes()
    assert data[:4] == b"wtns"
    _, count = struct.unpack_from("<II", data, 4)
    offset = 12
    sections = {}
    for _ in range(count):
        tag, size = struct.unpack_from("<IQ", data, offset)
        offset += 12
        sections[tag] = data[offset:offset + size]
        offset += size
    width = struct.unpack_from("<I", sections[1])[0]
    prime = int.from_bytes(sections[1][4:4 + width], "little")
    return prime, [int.from_bytes(sections[2][i:i + width], "little")
                   for i in range(0, len(sections[2]), width)]


def check(ir, inputs, native):
    text = Path(ir).read_text().split("function.def @__llzk_flat_constrain", 1)[1]
    header, *body = text.splitlines()
    paths = [ast.literal_eval(re.sub(r" : (?:index|i64)", "", p))
             for p in re.findall(r"\bpath = (\[[^\]]*\])", header.split("poly.signal_bindings = ", 1)[1].split("poly.source =", 1)[0])]
    prime, reference = witness(native)
    values = {f"%arg{i}": int(inputs[p[1]]) % prime for i, p in enumerate(paths)
              if len(p) == 2 and p[0] == 1}
    expressions, equations = {}, []
    for line in body:
        line = line.strip()
        if line in ("}", "function.return", ""):
            continue
        constant = re.match(r"(%[\w.]+) = felt.const\s+(-?\d+)", line)
        arithmetic = re.match(r"(%[\w.]+) = felt\.(add|sub|mul) (%[\w.]+), (%[\w.]+)", line)
        equation = re.match(r"constrain.eq (%[\w.]+), (%[\w.]+)", line)
        if constant:
            expressions[constant[1]] = int(constant[2]) % prime
        elif arithmetic:
            expressions[arithmetic[1]] = arithmetic[2], arithmetic[3], arithmetic[4]
        elif equation:
            equations.append((equation[1], equation[2]))
        else:
            raise ValueError(f"unsupported flat operation: {line}")

    def evaluate(name, cache):
        if name in values:
            return values[name]
        if name in cache:
            return cache[name]
        expr = expressions.get(name)
        if expr is None:
            return None
        if isinstance(expr, int):
            return expr
        op, lhs, rhs = expr
        lhs, rhs = evaluate(lhs, cache), evaluate(rhs, cache)
        if lhs is None or rhs is None:
            return None
        result = {"add": lambda: lhs + rhs, "sub": lambda: lhs - rhs,
                  "mul": lambda: lhs * rhs}[op]() % prime
        cache[name] = result
        return result

    while True:
        previous = len(values)
        cache = {}
        for lhs, rhs in equations:
            a, b = evaluate(lhs, cache), evaluate(rhs, cache)
            if a is not None and b is not None:
                assert a == b, f"unsatisfied equation {lhs} == {rhs}"
            elif a is None and b is not None and lhs.startswith("%arg"):
                values[lhs] = b
            elif b is None and a is not None and rhs.startswith("%arg"):
                values[rhs] = a
        if len(values) == previous:
            break
    assert len(values) == len(paths), f"unresolved signals: {len(paths) - len(values)}"
    cache = {}
    assert all(evaluate(a, cache) == evaluate(b, cache) for a, b in equations)
    output = values[f"%arg{paths.index([0, 'out'])}"]
    assert output == reference[1], f"native output mismatch: {output} != {reference[1]}"
    detected = 0
    for name, old in list(values.items()):
        values[name] = (old + 1) % prime
        cache = {}
        assert any(evaluate(a, cache) != evaluate(b, cache) for a, b in equations), name
        values[name] = old
        detected += 1
    return {"constraints": len(equations), "signals": len(paths), "output": str(output),
            "native_output_matches": True, "signal_perturbations_detected": detected}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ir")
    parser.add_argument("input_json")
    parser.add_argument("native_wtns")
    args = parser.parse_args()
    print(json.dumps(check(args.ir, json.loads(Path(args.input_json).read_text())["inputs"], args.native_wtns)))
