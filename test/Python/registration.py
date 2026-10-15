# RUN: %python %s
# REQUIRES: python-bindings
"""Exercise registration through nanobind's MLIR capsule adaptors."""

from mlir import ir
import llzk
from llzk import registration

registry = ir.DialectRegistry()
registration.register_dialects(registry)
registration.register_dialects(registry)

for invalid in (None, object(), 42):
    try:
        registration.register_dialects(invalid)
    except TypeError:
        pass
    else:
        raise AssertionError("register_dialects accepted an invalid registry")

with ir.Context() as context:
    context.append_dialect_registry(registry)
    context.allow_unregistered_dialects = False
    context.load_all_available_dialects()
    assert str(ir.Type.parse('!felt.type<"bn254">')) == '!felt.type<"bn254">'
    assert context.is_registered_operation("felt.const")
    assert context.is_registered_operation("llzk.nondet")

# The registry may be released after appending it to a context.
with ir.Context() as context:
    llzk.add_dialects_to_context(context)
    context.allow_unregistered_dialects = False
    context.load_all_available_dialects()
    assert str(ir.Type.parse("!felt.type")) == "!felt.type"
    assert context.is_registered_operation("struct.def")
