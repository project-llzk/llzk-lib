# RUN: %python %s llzk-opt
# REQUIRES: python-bindings
"""Check that native and Python pass spellings run the same LLZK implementation."""

import subprocess
import sys

# Import MLIR first so its upstream pass is registered before LLZK's alias.
from mlir import ir
from mlir.passmanager import PassManager
from llzk import registration


registry = ir.DialectRegistry()
registration.register_dialects(registry)
registration.register_passes(registry)
registration.register_passes(registry)

# The loop body is unreachable. Upstream dead-value removal leaves a live poison
# as its yielded value; LLZK replaces that poison with nondet. Array writes keep
# the loop's side effects live through canonicalization.
source = """
module attributes {llzk.lang} {
  function.def @keep(%array: !array.type<1 x index>) -> index {
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %result:2 = scf.while (%value = %c5) : (index) -> (index, index) {
      array.write %array[%c0] = %value : <1 x index>, index
      %condition = arith.cmpi slt, %value, %c5 : index
      scf.condition(%condition) %value, %value : index, index
    } do {
    ^bb0(%first: index, %second: index):
      %next = scf.execute_region -> index {
        scf.yield %second : index
      }
      scf.yield %next : index
    }
    function.return %result#0 : index
  }
}
"""

with ir.Context() as context:
    context.append_dialect_registry(registry)
    context.load_all_available_dialects()

    expected = None
    for name in ("llzk-remove-dead-values", "remove-dead-values"):
        pipeline = f"builtin.module({name})"
        manager = PassManager.parse(pipeline)
        assert str(manager) == "builtin.module(llzk-remove-dead-values)"
        module = ir.Module.parse(source)
        manager.run(module.operation)
        assert module.operation.verify()
        result = str(module)
        # Unused poison alone cannot distinguish the upstream pass from LLZK's.
        assert "ub.poison" not in result
        assert result.count("llzk.nondet") == 1
        if expected is None:
            expected = result
        assert result == expected

        for option in (f"--{name}", f"--pass-pipeline={pipeline}"):
            native = subprocess.run(
                [sys.argv[1], option], input=source, text=True,
                capture_output=True, check=False,
            )
            assert native.returncode == 0, native.stderr
            native_module = ir.Module.parse(native.stdout)
            assert native_module.operation.verify()
            assert str(native_module) == expected, option
