"""LLZK dialect registration for the MLIR Python API."""

import mlir.ir
from . import registration


def add_dialects_to_context(context: "mlir.ir.Context") -> None:
    """Register core LLZK dialects in an existing MLIR context."""
    registry = mlir.ir.DialectRegistry()
    registration.register_dialects(registry)
    context.append_dialect_registry(registry)
