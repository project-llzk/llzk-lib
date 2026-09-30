//===-- SMTInfoDetail.h - SMT metadata implementation helpers --*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Diagnostics.h>

namespace llzk::smt_info::detail {

/// Creates a keyword attribute without exposing generated attribute storage to
/// operation implementation files.
mlir::Attribute getKeywordAttr(mlir::MLIRContext *context, llvm::StringRef value);

/// Creates a symbol attribute without exposing generated attribute storage to
/// operation implementation files.
mlir::Attribute getSymbolAttr(mlir::MLIRContext *context, llvm::StringRef value);

/// Verifies that `value` is a valid SMT-LIB keyword and reports failures
/// through `emitError`.
mlir::LogicalResult
verifyKeyword(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, llvm::StringRef value);

} // namespace llzk::smt_info::detail
