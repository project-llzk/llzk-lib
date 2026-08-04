//===-- Wtns.h - snarkjs-compatible witness output -------------*- C++ -*-===//
#pragma once

#include "llzk/Util/Field.h"

#include <mlir/IR/BuiltinOps.h>

#include <llvm/Support/Error.h>
#include <llvm/Support/JSON.h>

namespace llzk::witgen {

/// Write a snarkjs-compatible WTNS v2 file from a full-witness JSON result.
/// Values use the same wire order as LLZK's R1CS lowering and binary exporter.
/// See `doc/doxygen/10_wtns_format.md` for the file schema and ordering contract.
llvm::Error writeWtns(
    mlir::ModuleOp moduleOp, const llvm::json::Value &fullWitness, const Field &field,
    llvm::StringRef outputFilename
);

} // namespace llzk::witgen
