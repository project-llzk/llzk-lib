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

/// Serialize one circuit in `moduleOp` to the binary .r1cs format.
mlir::LogicalResult exportR1CSBinary(
    mlir::ModuleOp moduleOp, llvm::raw_ostream &output, llvm::StringRef prime,
    llvm::StringRef circuitName = {}
);

} // namespace r1cs
