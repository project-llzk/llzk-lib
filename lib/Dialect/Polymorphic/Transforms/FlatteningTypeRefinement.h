//===-- FlatteningTypeRefinement.h ------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Private type-refinement contracts used by scalarization and propagation.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"

#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>

namespace llzk::polymorphic::detail::flattening {

class ConversionTracker;

/// Return a representable common refinement, accounting for this pass's instantiated types.
mlir::FailureOr<mlir::Type>
getCommonRefinedType(mlir::Type lhs, mlir::Type rhs, const ConversionTracker &tracker);

/// Return whether a scalar source can supply the requested common refinement.
bool canUseScalarizedValueAsType(
    mlir::Type sourceType, mlir::Type targetType, const ConversionTracker &tracker,
    const char *patName
);

/// Check the independently declared type requirements of an array allocation's consumers.
bool canRefineCreateArrayUsersToType(
    array::CreateArrayOp createOp, mlir::Type refinedType, mlir::SymbolTableCollection &tables
);

/// Check that a member-read refinement preserves each supported consumer's type contract.
bool canRefineMemberReadUsersToType(
    component::MemberReadOp readOp, mlir::Type refinedType, mlir::SymbolTableCollection &tables,
    const ConversionTracker &tracker
);

/// Validate external witness observations and return the constraint calls to retarget.
/// `isInternalUse` identifies operands the caller rewrites itself. Failures are silent unless
/// `diagnosticOp` is provided, allowing speculative propagation and diagnosed scalarization.
mlir::FailureOr<llvm::SmallVector<function::CallOp>> collectSpecializedWitnessConstraints(
    mlir::Value source, mlir::Type type, mlir::SymbolTableCollection &tables,
    llvm::function_ref<bool(mlir::OpOperand &)> isInternalUse,
    mlir::Operation *diagnosticOp = nullptr
);

/// Create a refined witness after its source and retarget already-validated constraint calls.
mlir::Value materializeSpecializedNondet(
    mlir::PatternRewriter &rewriter, NonDetOp sourceNondet, mlir::Location loc, mlir::Type type,
    llvm::ArrayRef<function::CallOp> externalConstraints = {}
);

} // namespace llzk::polymorphic::detail::flattening
