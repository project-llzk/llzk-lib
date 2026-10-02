"""Check aggregate copy boundaries and binary R1CS/WTNS agreement for each case."""
from pathlib import Path
import json
import struct
import subprocess
import sys

opt, witgen, translate, source, temporary = sys.argv[1:]
directory = Path(temporary)
directory.mkdir(parents=True, exist_ok=True)


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


raw = directory / 'case.llzk'
raw.write_text(Path(source).read_text())
evaluated = directory / 'case.evaluated.llzk'
subprocess.check_call([opt, str(raw), '--llzk-monomorphize', '--llzk-evaluate-constraints',
                       '-o', str(evaluated)])
binary = directory / 'case.r1cs'
r1cs_ir = directory / 'lowered.r1cs.mlir'
subprocess.check_call([opt, str(evaluated), '--llzk-full-r1cs-lowering',
                       '-o', str(r1cs_ir)])
subprocess.check_call([translate, str(r1cs_ir), '--r1cs-to-binary',
                       '--r1cs-prime=2013265921', '-o', str(binary)])
circuit = sections(binary, b'r1cs')
width = struct.unpack_from('<I', circuit[1])[0]
prime = int.from_bytes(circuit[1][4:4 + width], 'little')
wires, _, _, _, _, constraints = struct.unpack_from('<IIIIQI', circuit[1], 4 + width)
assert constraints > 0
x = 3
inputs = directory / 'input.json'
inputs.write_text(json.dumps([x]))
witness_file = directory / 'witness.wtns'
witness = json.loads(subprocess.check_output(
    [witgen, str(evaluated), '--inputs', str(inputs), '--output-wtns', str(witness_file)],
    text=True))
assert int(witness['signals']['out']) == x, (x, witness)
witness = sections(witness_file, b'wtns')
assert struct.unpack_from('<I', witness[1])[0] == width
values = [int.from_bytes(witness[2][i:i + width], 'little')
          for i in range(0, len(witness[2]), width)]
assert len(values) == wires and values == [1, x, x], values
offset = 0
for _ in range(constraints):
    sides = []
    for _ in range(3):
        terms = struct.unpack_from('<I', circuit[2], offset)[0]
        offset += 4
        total = 0
        for _ in range(terms):
            wire = struct.unpack_from('<I', circuit[2], offset)[0]
            coefficient = int.from_bytes(
                circuit[2][offset + 4:offset + 4 + width], 'little')
            offset += 4 + width
            total += coefficient * values[wire]
        sides.append(total % prime)
    a, b, c = sides
    assert (a * b - c) % prime == 0, (x, a, b, c)
assert offset == len(circuit[2])
