"""Check inverse reuse, prime isolation, and correct recomputation after eviction."""
from pathlib import Path
import re
import sys
from ir_helpers import evaluated_method

report = Path(sys.argv[1]).read_text()
if len(sys.argv) == 2:
    assert "inverse_cache_hits=3 inverse_cache_misses=2" in report, report
else:
    assert "inverse_cache_hits=1 inverse_cache_misses=8194" in report, report
    generated = evaluated_method(Path(sys.argv[2]).read_text())
    equations = re.findall(r"constrain.eq (%[\w.]+), (%[\w.]+)", generated)
    assert len(equations) == 8195, len(equations)
    assert all(lhs == rhs for lhs, rhs in equations), "x * inverse(x) must fold to one"
