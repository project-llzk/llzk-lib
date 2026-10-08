//===-- LLZKLayout.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>

#include <cstdint>
#include <optional>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace llzk {

/// Logical felt-signal paths in a monomorphized LLZK circuit. The first path
/// segment is "main" for the main instance, or "arg" followed by an i64
/// constrain argument position. Subsequent segments are StringAttr member names or index
/// attributes for array indices. Signal IDs are positions in signalPaths,
/// starting at zero. Struct members and POD fields are visited in declaration
/// order, expanding each field fully before the next. Arrays use numeric index
/// order. Constrain arguments come first in signature order, followed by the
/// main instance's members.
/// Every declared signal receives an ID, including unused signals. Members must
/// have the signal annotation, except main's public members, which are implicit
/// signals. Main's inputs are also implicit signals. These IDs
/// are independent of any backend's wire numbering or auxiliary variables.
struct LLZKLayout {
  /// Structural paths indexed by logical signal ID.
  llvm::SmallVector<mlir::ArrayAttr> signalPaths;
  /// Reverse mapping from structural storage paths to logical signal IDs.
  llvm::DenseMap<mlir::ArrayAttr, uint64_t> signalIds;
  /// Optional display names indexed by constrain argument position. Entry zero
  /// represents self and is always null; unnamed arguments also have null entries.
  llvm::SmallVector<mlir::StringAttr> argumentNames;

  /// Look up a structural path in this layout. Argument names are display
  /// labels; the argument position in the path determines argument identity.
  std::optional<uint64_t> getSignalId(mlir::ArrayAttr path) const;
};

/// Enumerate the signal leaves of llzk.main and its constrain arguments. Struct
/// references and array shapes must be concrete after template monomorphization.
/// Rolled arrays in member storage are resolved through poly.family metadata,
/// including struct fields inside POD elements. The monomorphizer's mapping is
/// trusted to associate each index tuple with the correct specialization arguments;
/// export checks that referenced IDs are unambiguous and origins match. Main's
/// constrain arguments are felt scalars or arrays of felt. Does not evaluate or
/// rewrite function bodies.
mlir::FailureOr<LLZKLayout> buildLLZKLayout(mlir::ModuleOp module);

/// Print the logical signal section of the .llzk-layout format.
void printLLZKLayout(const LLZKLayout &layout, llvm::raw_ostream &output);

} // namespace llzk
