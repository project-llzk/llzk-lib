//===-- TransformationPasses.h ----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Pass/PassBase.h"
#include "llzk/Util/Walk.h"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/ValueRange.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

namespace llzk::pod {

namespace detail {

/// Return the nearest enclosing SCF loop that carries writes to the same external POD record.
///
/// This is used by POD-to-scalar lowering to decide when synthetic split-array backing for a fresh
/// array-of-POD field read must be hoisted out of the loop body so writes remain visible to later
/// iterations.
inline mlir::Operation *findNearestLoopCarriedPodAccess(ReadPodOp readOp) {
  auto isValueDefinedInside = [](mlir::Operation *ancestor, mlir::Value value) {
    if (mlir::Operation *defOp = value.getDefiningOp()) {
      return ancestor->isAncestor(defOp);
    }

    auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value);
    mlir::Operation *parentOp = blockArg.getOwner()->getParentOp();
    return parentOp && ancestor->isAncestor(parentOp);
  };

  for (mlir::Operation *parent = readOp->getParentOp(); parent; parent = parent->getParentOp()) {
    if (!mlir::isa<mlir::scf::ForOp, mlir::scf::WhileOp>(parent) ||
        isValueDefinedInside(parent, readOp.getPodRef())) {
      continue;
    }

    if (walkContainsMatch<WritePodOp>(*parent, [&readOp](WritePodOp writeOp) {
      return writeOp.getPodRef() == readOp.getPodRef() &&
             writeOp.getRecordNameAttr() == readOp.getRecordNameAttr();
    })) {
      return parent;
    }
  }
  return nullptr;
}

/// Append `values` only when every entry exactly matches the corresponding expected type.
///
/// On failure, `output` is left unchanged. POD-to-scalar uses this to avoid leaking partially
/// collected split-array state into later fallback paths.
inline bool appendValuesWithExactTypes(
    mlir::ValueRange values, mlir::TypeRange expectedTypes,
    llvm::SmallVectorImpl<mlir::Value> &output
) {
  if (values.size() != expectedTypes.size()) {
    return false;
  }

  llvm::SmallVector<mlir::Value> stagedValues;
  stagedValues.reserve(values.size());
  for (auto [value, expectedType] : llvm::zip_equal(values, expectedTypes)) {
    if (value.getType() != expectedType) {
      return false;
    }
    stagedValues.push_back(value);
  }

  llvm::append_range(output, stagedValues);
  return true;
}

} // namespace detail

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "llzk/Dialect/POD/Transforms/TransformationPasses.h.inc"

} // namespace llzk::pod
