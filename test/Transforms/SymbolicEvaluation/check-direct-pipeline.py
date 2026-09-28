"""Check binary R1CS/WTNS agreement for nested scalar or array storage."""
from pathlib import Path
import json
import struct
import subprocess
import sys

opt, witgen, translate, source, temporary = sys.argv[1:6]
reject_invalid = len(sys.argv) > 6 and sys.argv[6] == "--reject-invalid"
array_case = "direct-array" in source
directory = Path(temporary)
directory.mkdir(parents=True, exist_ok=True)

def run(*args):
    return subprocess.check_output(list(args), text=True)

evaluated = run(opt, source, '--llzk-monomorphize', '--llzk-evaluate-constraints')
eval_file = directory / 'evaluated.llzk'
eval_file.write_text(evaluated)
r1cs_ir = run(opt, str(eval_file), '--llzk-full-direct-r1cs-lowering')
r1cs_file = directory / 'lowered.llzk'
r1cs_file.write_text(r1cs_ir)
binary = directory / 'circuit.r1cs'
run(translate, '--r1cs-to-binary', '--r1cs-prime=2013265921', str(r1cs_file), '-o', str(binary))

def sections(data, magic):
    assert data[:4] == magic
    count = struct.unpack_from('<I', data, 8)[0]
    result = {}
    offset = 12
    for _ in range(count):
        kind, size = struct.unpack_from('<IQ', data, offset)
        offset += 12
        result[kind] = data[offset:offset + size]
        offset += size
    assert offset == len(data)
    return result

r1cs = sections(binary.read_bytes(), b'r1cs')
width = struct.unpack_from('<I', r1cs[1])[0]
prime = int.from_bytes(r1cs[1][4:4 + width], 'little')
wires, pub_outputs, pub_inputs, private_inputs, labels, constraints = struct.unpack_from('<IIIIQI', r1cs[1], 4 + width)
assert (pub_outputs, pub_inputs, private_inputs) == (1, 2 if array_case else 1, 0)
assert (constraints, wires) == ((3, 6) if array_case else (2, 4))

def check_witness(values):
    offset = 0
    valid = True
    for _ in range(constraints):
        combinations = []
        for _ in range(3):
            terms = struct.unpack_from('<I', r1cs[2], offset)[0]
            offset += 4
            total = 0
            for _ in range(terms):
                wire = struct.unpack_from('<I', r1cs[2], offset)[0]
                coefficient = int.from_bytes(r1cs[2][offset + 4:offset + 4 + width], 'little')
                offset += 4 + width
                total += coefficient * values[wire]
            combinations.append(total % prime)
        a, b, c = combinations
        valid &= (a * b - c) % prime == 0
    assert offset == len(r1cs[2])
    return valid

x = 3
inputs = directory / 'input.json'
inputs.write_text(json.dumps([[x, x + 1]] if array_case else [x]))
wtns_file = directory / 'witness.wtns'
witness = json.loads(run(witgen, str(eval_file), '--inputs', str(inputs), '--output-wtns', str(wtns_file)))
assert int(witness['signals']['out']) == (x ** 2 * (x + 1) ** 2 if array_case else x ** 3) % prime
wtns = sections(wtns_file.read_bytes(), b'wtns')
field_width = struct.unpack_from('<I', wtns[1])[0]
count = struct.unpack_from('<I', wtns[1], 4 + field_width)[0]
values = [int.from_bytes(wtns[2][i:i + field_width], 'little') for i in range(0, len(wtns[2]), field_width)]
assert count == wires == len(values) and values[0] == 1
assert values[1] == int(witness['signals']['out']) and values[2] == x
if reject_invalid:
    values[1] = (values[1] + 1) % prime
    assert not check_witness(values), 'corrupted output witness was accepted'
else:
    assert check_witness(values)
