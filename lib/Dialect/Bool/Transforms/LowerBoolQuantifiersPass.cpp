//===-- LowerBoolQuantifiersPass.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-lower-bool-quantifiers` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Bool/IR/Utils.h"
#include "llzk/Dialect/Bool/Transforms/TransformationPasses.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

// Include the generated base pass class definitions.
namespace llzk::boolean {
#define GEN_PASS_DEF_LOWERBOOLQUANTIFIERSPASS
#include "llzk/Dialect/Bool/Transforms/TransformationPasses.h.inc"
} // namespace llzk::boolean

using namespace mlir;
using namespace llzk;
using namespace llzk::array;
using namespace llzk::boolean;

namespace {

/// Build the value passed to the cloned quantifier body for the current loop index.
static Value buildQuantifierIterValue(
    Location loc, Value sort, ArrayType sortType, Value index, PatternRewriter &rewriter
) {
  if (sortType.getDimensionSizes().size() == 1) {
    return rewriter.create<ReadArrayOp>(loc, sort, ValueRange {index});
  }

  Type iterType = getQuantifierOpDomainIterType(sortType);
  return rewriter.create<ExtractArrayOp>(loc, iterType, sort, ValueRange {index});
}

/// Lower a bool quantifier to an `scf.for` loop over the first dimension of its array sort.
template <typename QuantifierOp, typename CombineOp>
static LogicalResult
lowerQuantifier(QuantifierOp op, PatternRewriter &rewriter, bool initialValue) {
  PatternRewriter::InsertionGuard guard(rewriter);
  Location loc = op.getLoc();
  auto sortType = cast<ArrayType>(op.getSort().getType());

  Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value upperBound = rewriter.create<ArrayLengthOp>(loc, op.getSort(), lowerBound);
  Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value init = rewriter.create<arith::ConstantIntOp>(loc, initialValue, rewriter.getI1Type());

  auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step, ValueRange {init});
  loop->setDiscardableAttrs(op->getDiscardableAttrDictionary());

  Block &loopBody = *loop.getBody();
  if (!loopBody.empty()) {
    rewriter.eraseOp(&loopBody.back());
  }

  rewriter.setInsertionPointToStart(&loopBody);
  Value iterValue =
      buildQuantifierIterValue(loc, op.getSort(), sortType, loop.getInductionVar(), rewriter);

  IRMapping mapping;
  mapping.map(op.getBody()->getArgument(0), iterValue);
  for (Operation &nestedOp : op.getBody()->without_terminator()) {
    rewriter.clone(nestedOp, mapping);
  }

  auto yieldOp = cast<YieldOp>(op.getBody()->getTerminator());
  Value predicate = mapping.lookupOrDefault(yieldOp.getValue());
  Value combined = rewriter.create<CombineOp>(loc, loop.getRegionIterArg(0), predicate);
  rewriter.create<scf::YieldOp>(loc, combined);

  rewriter.replaceOp(op, loop.getResults());
  return success();
}

class LowerForAllOp : public OpRewritePattern<ForAllOp> {
public:
  using OpRewritePattern<ForAllOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForAllOp op, PatternRewriter &rewriter) const override {
    return lowerQuantifier<ForAllOp, AndBoolOp>(op, rewriter, /*initialValue=*/true);
  }
};

class LowerExistsOp : public OpRewritePattern<ExistsOp> {
public:
  using OpRewritePattern<ExistsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExistsOp op, PatternRewriter &rewriter) const override {
    return lowerQuantifier<ExistsOp, OrBoolOp>(op, rewriter, /*initialValue=*/false);
  }
};

class PassImpl : public llzk::boolean::impl::LowerBoolQuantifiersPassBase<PassImpl> {
  using Base = LowerBoolQuantifiersPassBase<PassImpl>;

public:
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerForAllOp, LowerExistsOp>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
