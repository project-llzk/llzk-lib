//===-- LLZKConversionUtils.h -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Shared utilities for dialect converting transformations.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Function/IR/Ops.h"

#include <mlir/IR/PatternMatch.h>

namespace llzk {

/// General helper for converting a `FuncDefOp` by changing its input and/or result types and the
/// associated attributes for those types.
class FunctionTypeConverter {

protected:
  virtual llvm::SmallVector<mlir::Type> convertInputs(mlir::ArrayRef<mlir::Type> origTypes) = 0;
  virtual llvm::SmallVector<mlir::Type> convertResults(mlir::ArrayRef<mlir::Type> origTypes) = 0;

  virtual mlir::ArrayAttr
  convertInputAttrs(mlir::ArrayAttr origAttrs, llvm::SmallVector<mlir::Type> newTypes) = 0;
  virtual mlir::ArrayAttr
  convertResultAttrs(mlir::ArrayAttr origAttrs, llvm::SmallVector<mlir::Type> newTypes) = 0;

  virtual void processBlockArgs(mlir::Block &entryBlock, mlir::RewriterBase &rewriter) = 0;

public:
  virtual ~FunctionTypeConverter() = default;

  void convert(function::FuncDefOp op, mlir::RewriterBase &rewriter) {
    // Update in/out types of the function
    mlir::FunctionType oldTy = op.getFunctionType();
    llvm::SmallVector<mlir::Type> newInputs = convertInputs(oldTy.getInputs());
    llvm::SmallVector<mlir::Type> newResults = convertResults(oldTy.getResults());
    mlir::FunctionType newTy = mlir::FunctionType::get(
        oldTy.getContext(), mlir::TypeRange(newInputs), mlir::TypeRange(newResults)
    );
    if (newTy == oldTy) {
      return; // nothing to change
    }

    // Pre-condition: arg/result count equals corresponding attribute count
    assert(!op.getResAttrsAttr() || op.getResAttrsAttr().size() == op.getNumResults());
    assert(!op.getArgAttrsAttr() || op.getArgAttrsAttr().size() == op.getNumArguments());
    rewriter.modifyOpInPlace(op, [&]() {
      op.setFunctionType(newTy);

      // If any input or result types were added, ensure the attributes are updated too.
      if (mlir::ArrayAttr newArgAttrs = convertInputAttrs(op.getArgAttrsAttr(), newInputs)) {
        op.setArgAttrsAttr(newArgAttrs);
      }
      if (mlir::ArrayAttr newResAttrs = convertResultAttrs(op.getResAttrsAttr(), newResults)) {
        op.setResAttrsAttr(newResAttrs);
      }
    });
    // Post-condition: arg/result count equals corresponding attribute count
    assert(!op.getResAttrsAttr() || op.getResAttrsAttr().size() == op.getNumResults());
    assert(!op.getArgAttrsAttr() || op.getArgAttrsAttr().size() == op.getNumArguments());

    // If the function has a body, ensure the entry block arguments match the function inputs.
    if (mlir::Region *body = op.getCallableRegion()) {
      mlir::Block &entryBlock = body->front();
      if (!std::cmp_equal(entryBlock.getNumArguments(), newInputs.size())) {
        processBlockArgs(entryBlock, rewriter);
        // Post-condition: block args must match function inputs
        assert(std::cmp_equal(entryBlock.getNumArguments(), newInputs.size()));
      }
    }
  }
};

} // namespace llzk
