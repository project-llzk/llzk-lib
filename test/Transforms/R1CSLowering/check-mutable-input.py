"""Auxiliary values and input wires use entry values despite local array mutation."""
from pathlib import Path
import json
import struct
import subprocess
import sys

opt, witgen, translate, source, temporary = sys.argv[1:6]
degree = int(sys.argv[6]) if len(sys.argv) > 6 else 2
directory = Path(temporary)
directory.mkdir(parents=True, exist_ok=True)
evaluated = directory / 'evaluated.llzk'
subprocess.check_call([opt, source, '--llzk-monomorphize', '--llzk-evaluate-constraints',
                       '-o', str(evaluated)])


def sections(path, magic):
    data = path.read_bytes()
    assert data[:4] == magic
    result = {}
    offset = 12
    for _ in range(struct.unpack_from('<I', data, 8)[0]):
        kind, size = struct.unpack_from('<IQ', data, offset)
        offset += 12
        result[kind] = data[offset:offset + size]
        offset += size
    assert offset == len(data)
    return result


lowered = directory / 'degree.llzk'
subprocess.check_call([opt, str(evaluated), f'--llzk-poly-lowering-pass=max-degree={degree}',
                       '-o', str(lowered)])
binary = directory / 'circuit.r1cs'
subprocess.check_call([translate, str(lowered), '--llzk-to-r1cs',
                       '--r1cs-prime=2013265921', '-o', str(binary)])
r1cs = sections(binary, b'r1cs')
width = struct.unpack_from('<I', r1cs[1])[0]
prime = int.from_bytes(r1cs[1][4:4 + width], 'little')
wires, _, _, _, _, constraints = struct.unpack_from('<IIIIQI', r1cs[1], 4 + width)
x = 3
inputs = directory / 'input.json'
inputs.write_text(json.dumps([[[x]]] if "multidimensional" in source else [[x]]))
witness_file = directory / 'witness.wtns'
witness = json.loads(subprocess.check_output(
    [witgen, str(lowered), '--inputs', str(inputs), '--output-wtns', str(witness_file)],
    text=True))
expected_output = x if 'helper' in source else (2 * x ** 2 if 'r1cs-aux' in source else x ** 3)
assert int(witness['signals']['out']) == expected_output
wtns = sections(witness_file, b'wtns')
values = [int.from_bytes(wtns[2][i:i + width], 'little')
          for i in range(0, len(wtns[2]), width)]
assert len(values) == wires
expected_values = [1, expected_output, x] + ([] if 'helper' in source else [x ** 2])
assert values == expected_values, values
offset = 0
for _ in range(constraints):
    sides = []
    for _ in range(3):
        terms = struct.unpack_from('<I', r1cs[2], offset)[0]
        offset += 4
        total = 0
        for _ in range(terms):
            wire = struct.unpack_from('<I', r1cs[2], offset)[0]
            coefficient = int.from_bytes(r1cs[2][offset + 4:offset + 4 + width], 'little')
            offset += 4 + width
            total += coefficient * values[wire]
        sides.append(total % prime)
    a, b, c = sides
    assert (a * b - c) % prime == 0, (x, a, b, c)
assert offset == len(r1cs[2])
