"""Check inferred binary moduli, explicit overrides, and ambiguous/missing fields."""
from pathlib import Path
import struct
import subprocess
import sys

opt, translate, source, temporary = sys.argv[1:]
directory = Path(temporary)
directory.mkdir(parents=True, exist_ok=True)
source_text = Path(source).read_text()


def export(source, name, prime=None, direct=False):
    output = directory / (name + '.r1cs')
    command = [translate, str(source), '--llzk-to-r1cs' if direct else '--r1cs-to-binary',
               '-o', str(output)]
    if prime is not None:
        command.append('--r1cs-prime=' + str(prime))
    result = subprocess.run(command, capture_output=True, text=True)
    return result, output


def read_prime(path):
    data = path.read_bytes()
    assert data[:4] == b'r1cs'
    offset = 12
    for _ in range(struct.unpack_from('<I', data, 8)[0]):
        section, size = struct.unpack_from('<IQ', data, offset)
        offset += 12
        if section == 1:
            width = struct.unpack_from('<I', data, offset)[0]
            return int.from_bytes(data[offset + 4:offset + 4 + width], 'little')
        offset += size
    raise AssertionError('missing R1CS header')


for field, prime in [
    ('babybear', 2013265921),
    ('bn128', 21888242871839275222246405745257275088548364400416034343698204186575808495617),
    ('export_custom', 17),
]:
    text = source_text.replace('babybear', field)
    if field == 'export_custom':
        text = text.replace('llzk.lang,', 'llzk.lang, llzk.fields = #felt.field<"export_custom", 17>,')
    raw = directory / (field + '.llzk')
    raw.write_text(text)
    result, inferred = export(raw, field + '-inferred', direct=True)
    assert result.returncode == 0, result.stderr
    assert read_prime(inferred) == prime
    result, explicit = export(raw, field + '-explicit', prime, direct=True)
    assert result.returncode == 0, result.stderr
    assert inferred.read_bytes() == explicit.read_bytes()
    lowered = directory / (field + '-lowered.mlir')
    subprocess.check_call([opt, str(raw), '--llzk-monomorphize', '--llzk-evaluate-constraints',
                           '--llzk-full-r1cs-lowering', '-o', str(lowered)])
    result, staged = export(lowered, field + '-staged')
    assert result.returncode == 0, result.stderr
    assert staged.read_bytes() == inferred.read_bytes()

# The explicit option remains authoritative, including its validation.
raw = directory / 'babybear.llzk'
result, override = export(raw, 'override', 17, direct=True)
assert result.returncode == 0, result.stderr
assert read_prime(override) == 17
for invalid, diagnostic in [('abc', 'must be a base-10 integer'), ('1', 'must be greater than 1')]:
    result, _ = export(raw, 'invalid', invalid, direct=True)
    assert result.returncode != 0 and diagnostic in result.stderr, result.stderr

# R1CS itself has no field-bearing signal types. Do not guess a modulus when
# there is no LLZK type information, an unspecified felt, or multiple fields.
for name, signature in [
    ('missing', ''),
    ('unspecified', 'function.def @fields(%x: !felt.type) { function.return }'),
    ('mixed', 'function.def @fields(%x: !felt.type<"babybear">, %y: !felt.type<"bn128">) { function.return }'),
]:
    ir = directory / (name + '.mlir')
    ir.write_text('module {\n' + signature + '\nr1cs.circuit @Main inputs (%arg: !r1cs.signal) {}\n}\n')
    result, _ = export(ir, name)
    assert result.returncode != 0, name
    assert "requires a non-empty '--r1cs-prime' option" in result.stderr, result.stderr
    result, explicit = export(ir, name + '-explicit', 17)
    assert result.returncode == 0, result.stderr
    assert read_prime(explicit) == 17
