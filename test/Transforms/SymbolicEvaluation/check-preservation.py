"""Verify source preservation and the generated function's structural boundary."""
import subprocess
import sys
from ir_helpers import check_preserved_compute, evaluated_method

base = subprocess.check_output([sys.argv[1], sys.argv[2], "--llzk-monomorphize"], text=True)
evaluated = subprocess.check_output(
    [sys.argv[1], sys.argv[2], "--llzk-monomorphize", "--llzk-evaluate-constraints"], text=True)
generated = check_preserved_compute(base, evaluated)
assert "struct.readm" in generated and "array.read" in generated

# Normalize equations by logical path and integer polynomial coefficients, not
# printed SSA order or legacy member-name ordering. This fixture has two children.
import ast
from collections import defaultdict
import re

legacy = subprocess.check_output(
    [sys.argv[1], sys.argv[2], "--llzk-full-struct-inlining"], text=True)


def equations(text, flat):
    if flat:
        text = evaluated_method(text)
        values = {"%arg1": {(repr([1]),): 1}}
    else:
        text = text.split("function.def @constrain", 1)[1]
        values = {"%arg1": {(repr([1]),): 1}}
    result = []
    for line in text.splitlines()[1:]:
        read = re.search(r'(%\w+) = struct.readm .*children_(\d+):!s<@Child>\+out', line)
        multiply = re.search(r'(%\w+) = felt.mul (%\w+), (%\w+)', line)
        equal = re.search(r'constrain.eq (%\w+), (%\w+)', line)
        binding = re.search(r'(%\w+) = .*poly.signal_binding = .*?\bpath = (\[[^\]]*\])', line)
        if flat and binding:
            path = ast.literal_eval(re.sub(r" : (?:index|i64)", "", binding[2]))
            values[binding[1]] = {(repr(path),): 1}
        elif read:
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
