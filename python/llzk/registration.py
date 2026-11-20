"""Register LLZK dialects and passes with MLIR."""

from ._mlir_libs._llzkRegistration import register_dialects, register_passes

__all__ = ["register_dialects", "register_passes"]
