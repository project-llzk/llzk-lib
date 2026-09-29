//===-- R1CSBinary.h - R1CS binary serialization ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>

namespace r1cs {

/// Module reference to the exported circuit and its physical witness-wire map.
constexpr char CIRCUIT_REF_ATTR_NAME[] = "r1cs.main";
constexpr char WIRE_BINDINGS_ATTR_NAME[] = "poly.wire_bindings";
constexpr char LAYOUT_SIGNALS_ATTR_NAME[] = "poly.layout_signals";
constexpr char LAYOUT_SIGNAL_ATTR_NAME[] = "poly.layout_signal";
constexpr char LAYOUT_ARGUMENT_SIGNALS_ATTR_NAME[] = "poly.layout_argument_signals";
constexpr char LAYOUT_ROOT_NAMES_ATTR_NAME[] = "poly.layout_root_names";

/// Serialize one circuit in `moduleOp` to the binary .r1cs format.
/// An empty prime infers the unique field used by the module's LLZK felt types.
/// Supply a decimal modulus explicitly when no unique field can be inferred.
mlir::LogicalResult exportR1CSBinary(
    mlir::ModuleOp moduleOp, llvm::raw_ostream &output, llvm::StringRef prime = {},
    llvm::StringRef circuitName = {}
);

/// Serialize direct-lowering signal paths and their R1CS wire layout.
///
/// Logical signal ids are derived from canonical LLZK access paths; the R1CS
/// section relates those ids to the physical wire ids used by binary export.
mlir::LogicalResult exportLLZKLayoutMap(
    mlir::ModuleOp moduleOp, llvm::raw_ostream &output, llvm::StringRef circuitName = {}
);

} // namespace r1cs
