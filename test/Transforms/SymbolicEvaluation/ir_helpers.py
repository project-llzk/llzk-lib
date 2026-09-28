"""Text helpers for checking evaluated methods without depending on SSA names."""
import re


def evaluated_method(text):
    lines = text.splitlines()
    for start, line in enumerate(lines):
        if 'function.def @constrain' in line and 'poly.evaluated' in line:
            indent = line[:len(line) - len(line.lstrip())]
            end = next(i for i in range(start + 1, len(lines)) if lines[i] == indent + '}')
            return '\n'.join(lines[start:end + 1])
    raise AssertionError('missing evaluated constrain method')


def compute_methods(text):
    return re.findall(r'(?ms)^([ ]*)function.def @compute\b(.*?)^\1}', text)


def check_preserved_compute(before, after):
    assert compute_methods(before) == compute_methods(after), 'evaluation changed rolled compute'
    body = evaluated_method(after)
    assert 'function.call' not in body and 'scf.' not in body, body
    assert '@__llzk_flat_constrain' not in after
    return body


if __name__ == '__main__':
    import sys
    print("module {\n" + evaluated_method(sys.stdin.read()) + "\n}")
