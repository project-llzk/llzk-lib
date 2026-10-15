# RUN: %python %s
# REQUIRES: python-bindings
"""Check generated core dialect wrappers, generic types/attributes, and passes."""

import importlib

from mlir import ir
from mlir.passmanager import PassManager
from mlir.dialects import felt, llzk as core, llzk_dialect
import llzk
from llzk import registration


registry = ir.DialectRegistry()
registration.register_dialects(registry)
registration.register_passes(registry)
registration.register_passes(registry)

with ir.Context() as context, ir.Location.unknown():
    context.append_dialect_registry(registry)
    context.allow_unregistered_dialects = False
    context.load_all_available_dialects()

    # An incorrect -bind-dialect name silently generates no operation classes.
    for dialect in (
        "array", "bool", "cast", "constrain", "felt", "function", "global",
        "include", "llzk", "pod", "poly", "ram", "smt_info", "string",
        "struct", "verif",
    ):
        module = importlib.import_module(f"mlir.dialects.{dialect}")
        operations = {
            value.OPERATION_NAME
            for value in vars(module).values()
            if isinstance(value, type) and issubclass(value, ir.OpView)
            and hasattr(value, "OPERATION_NAME")
        }
        assert operations, f"No generated operations for {dialect}"
        assert all(name.startswith(dialect + ".") for name in operations)
        assert all(context.is_registered_operation(name) for name in operations)

    assert llzk_dialect.NonDetOp is core.NonDetOp
    felt_type = ir.Type.parse('!felt.type<"bn254">')
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        value = core.NonDetOp(felt_type)
    assert value.result.type == felt_type
    assert module.operation.verify()
    reparsed = ir.Module.parse(str(module))
    assert reparsed.operation.verify()
    assert isinstance(reparsed.body.operations[0], core.NonDetOp)

    # Attributes and types need no handwritten Python subclasses to round-trip.
    attribute = ir.Attribute.parse('#felt<const 42 : !felt.type<"bn254">>')
    assert ir.Attribute.parse(str(attribute)) == attribute
    with ir.InsertionPoint(reparsed.body):
        constant = felt.FeltConstantOp(attribute)
        added = felt.AddFeltOp(constant.result, constant.result)
    assert constant.value == attribute
    assert constant.result.type == felt_type
    assert added.result.type == felt_type
    assert reparsed.operation.verify()

    # Exercise a pass exposed by the LLZK registry, beyond upstream canonicalize.
    PassManager.parse("builtin.module(llzk-unused-declaration-elim)").run(
        reparsed.operation
    )
    assert reparsed.operation.verify()

    # Coexist with upstream's eagerly registered remove-dead-values pass.
    poison_module = ir.Module.parse("module { %0 = ub.poison : i32 }")
    PassManager.parse("builtin.module(llzk-remove-dead-values)").run(
        poison_module.operation
    )
    assert "ub.poison" not in str(poison_module)
    assert poison_module.operation.verify()
