"""Unsupported polynomial operations must diagnose without terminating by signal."""
import subprocess
import sys

opt, translate, source, output = sys.argv[1:]
for command in ([opt, source, '--llzk-monomorphize', '--llzk-evaluate-constraints',
                 '--llzk-full-direct-r1cs-lowering', '-o', output],):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode > 0, (result.returncode, result.stderr)
    assert 'unsupported operation in R1CS normalization' in result.stderr, result.stderr
