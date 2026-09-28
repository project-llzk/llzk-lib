"""Check that direct and staged translation emit identical R1CS binaries."""
from pathlib import Path
import subprocess
import sys

opt, translate, source, temporary = sys.argv[1:]
directory = Path(temporary)
directory.mkdir(parents=True, exist_ok=True)
evaluated = directory / 'evaluated.llzk'
lowered = directory / 'lowered.llzk'
reference = directory / 'reference.r1cs'
subprocess.check_call([opt, source, '--llzk-monomorphize', '--llzk-evaluate-constraints', '-o', str(evaluated)])
subprocess.check_call([opt, str(evaluated), '--llzk-full-r1cs-lowering', '-o', str(lowered)])
subprocess.check_call([translate, str(lowered), '--r1cs-to-binary', '--r1cs-prime=2013265921', '-o', str(reference)])
for input_ir in (source, str(evaluated)):
    output = directory / 'direct.r1cs'
    subprocess.check_call([translate, input_ir, '--llzk-to-r1cs', '--r1cs-prime=2013265921', '-o', str(output)])
    assert output.read_bytes() == reference.read_bytes()
