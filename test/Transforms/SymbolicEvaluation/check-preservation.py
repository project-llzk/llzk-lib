"""Verify source preservation and the generated function's structural boundary."""
import subprocess
import sys

base = subprocess.check_output([sys.argv[1], sys.argv[2], "--llzk-monomorphize"], text=True)
evaluated = subprocess.check_output(
    [sys.argv[1], sys.argv[2], "--llzk-monomorphize", "--llzk-evaluate-constraints"], text=True)
prefix, generated = evaluated.split("  function.def @__llzk_flat_constrain", 1)
assert base == prefix + "}\n\n", "evaluation changed retained source"
body = generated.splitlines()[1:]
for line in body:
    assert not any(op in line for op in ("scf.", "function.call", "array.", "pod.", "struct.")), line

# Normalize equations by logical path and integer polynomial coefficients, not
# printed SSA order or legacy member-name ordering. This fixture has two children.
import ast
from collections import defaultdict
import re

legacy = subprocess.check_output(
    [sys.argv[1], sys.argv[2], "--llzk-full-struct-inlining"], text=True)


def equations(text, flat):
    if flat:
        text = text.split("function.def @__llzk_flat_constrain", 1)[1]
        header = text.splitlines()[0].split("poly.signal_bindings = ", 1)[1]
        paths = [ast.literal_eval(re.sub(r" : (?:index|i64)", "", p))
                 for p in re.findall(r"\bpath = (\[[^\]]*\])", header)]
        values = {f"%arg{i}": {(repr(p),): 1} for i, p in enumerate(paths)}
    else:
        text = text.split("function.def @constrain", 1)[1]
        values = {"%arg1": {(repr([1]),): 1}}
    result = []
    for line in text.splitlines()[1:]:
        read = re.search(r'(%\w+) = struct.readm .*children_(\d+):!s<@Child>\+out', line)
        multiply = re.search(r'(%\w+) = felt.mul (%\w+), (%\w+)', line)
        equal = re.search(r'constrain.eq (%\w+), (%\w+)', line)
        if read:
            values[read[1]] = {(repr([0, "children", int(read[2]), "out"]),): 1}
        elif multiply:
            terms = defaultdict(int)
            for a, ac in values[multiply[2]].items():
                for b, bc in values[multiply[3]].items():
                    terms[tuple(sorted(a + b))] += ac * bc
            values[multiply[1]] = dict(terms)
        elif equal:
            terms = defaultdict(int, values[equal[1]])
            for term, coefficient in values[equal[2]].items():
                terms[term] -= coefficient
            result.append(tuple(sorted((term, c) for term, c in terms.items() if c)))
    return sorted(result)


assert equations(evaluated, True) == equations(legacy, False), "legacy equations differ"
assert len(equations(evaluated, True)) == 2
