"""Check that mutating an aggregate copy preserves the original value."""
from pathlib import Path
import json
import subprocess
import sys

opt, witgen, translate, source, temporary = sys.argv[1:]
directory = Path(temporary)
directory.mkdir(parents=True, exist_ok=True)
inputs = directory / 'input.json'
inputs.write_text(json.dumps([3]))
witness = json.loads(subprocess.check_output(
    [witgen, source, '--inputs', str(inputs)], text=True))
assert int(witness['out']) == 3, witness
