//===-- Ops.cpp - LLZK operation implementations ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/IR/Ops.h"

#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/PatternMatch.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/LLZK/IR/Ops.cpp.inc"

using namespace mlir;

namespace llzk {

namespace {

struct RemoveUnusedNonDetPattern : public OpRewritePattern<NonDetOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NonDetOp op, PatternRewriter &rewriter) const override {
    if (!op.getResult().use_empty()) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===------------------------------------------------------------------===//
// NonDetOp
//===------------------------------------------------------------------===//

void NonDetOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "nondet");
}

void NonDetOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<RemoveUnusedNonDetPattern>(context);
}

//===------------------------------------------------------------------===//
// AuxOp
//===------------------------------------------------------------------===//

void AuxOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) { setNameFn(getResult(), "aux"); }

LogicalResult AuxOp::verify() {
  if (!llvm::isa<felt::FeltType>(getResult().getType())) {
    return emitOpError("result #0 must be finite field element");
  }
  return success();
}

} // namespace llzk
