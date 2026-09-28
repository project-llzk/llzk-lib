"""Check exact source preservation while dead witness computations are cleaned."""
import subprocess
import sys
from ir_helpers import check_preserved_compute

command = [sys.argv[1], sys.argv[2], "--llzk-monomorphize"]
base = subprocess.check_output(command, text=True)
evaluated = subprocess.check_output(command + ["--llzk-evaluate-constraints"], text=True)
generated = check_preserved_compute(base, evaluated)
assert generated.count("constrain.eq") == 3, generated
assert "function.call" not in generated and "scf." not in generated
