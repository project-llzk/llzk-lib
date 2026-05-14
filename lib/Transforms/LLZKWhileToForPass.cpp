//===-- LLZKWhileToForPass.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-while-to-for` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <llvm/Support/Debug.h>

namespace llzk {
#define GEN_PASS_DECL_WHILETOFORPASS
#define GEN_PASS_DEF_WHILETOFORPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

#define DEBUG_TYPE "while-to-for"

using namespace mlir;

namespace llzk {

using namespace boolean;
using namespace felt;
using namespace scf;

struct ForOpInfo {
  // SSA values holding the loop bounds
  std::optional<Value> lb, ub, step;

  // Block argument index of the induction variable in the *before* block
  std::optional<size_t> ivarIndexBefore;
  // Block argument index of the induction variable in the *after* block
  std::optional<size_t> ivarIndexAfter;

  void reset() {
    lb.reset();
    ub.reset();
    step.reset();
  }

  bool success() const {
    return lb.has_value() && ub.has_value() && step.has_value() && ivarIndexBefore.has_value() &&
           ivarIndexAfter.has_value();
  }
};

static inline ForOpInfo parseInfo(WhileOp op) {
  ForOpInfo info;

  auto isRuntimeConstant = [&op](Value val) -> bool {
    // The value can't come from a block argument owned by the while loop
    if (auto blockArg = dyn_cast<BlockArgument>(val)) {
      return blockArg.getParentBlock()->getParentOp() != op;
    }
    return true;
  };

  (void)isRuntimeConstant;

  auto condition = op.getConditionOp().getCondition();
  Value ivarBefore;
  if (auto cmp = condition.getDefiningOp<CmpOp>();
      cmp && cmp.getPredicate() == FeltCmpPredicate::LT) {
    // We found the ivar and the ub
    ivarBefore = cmp.getLhs();
    info.ub = cmp.getRhs();
  } else {
    return info;
  }

  auto getBlockArgIndex = [](Value v, ValueRange argList, std::optional<size_t> &index) {
    for (auto [i, arg] : llvm::enumerate(argList)) {
      if (arg == v) {
        index.emplace(i);
        break;
      }
    }
  };

  // Find which # block arg the ivar is in the before and after blocks
  getBlockArgIndex(ivarBefore, op.getBeforeArguments(), info.ivarIndexBefore);
  getBlockArgIndex(ivarBefore, op.getConditionOp().getArgs(), info.ivarIndexAfter);

  if (!info.ivarIndexBefore.has_value() || !info.ivarIndexAfter.has_value()) {
    return info;
  }

  // If the yielded final value of the induction variable has any uses, we can't cleanly transform
  // this to an scf.for (which doesn't explicitly yield its induction var) without doing some extra
  // computation. Lets just conservatively bail out in that case
  auto yieldedIVar = op->getResults().drop_front(*info.ivarIndexAfter).front();
  if (!yieldedIVar.use_empty()) {
    return ForOpInfo {};
  }

  Value ivarAfter = op.getAfterArguments().drop_front(*info.ivarIndexAfter).front();

  // We need an induction variable anyway, but if the loop has {llzk.loopbounds} we can skip trying
  // to parse the rest of the bounds and just materialize constants
  if (op->hasAttr(LoopBoundsAttr::name)) {
    auto bounds = op->getAttrOfType<LoopBoundsAttr>(LoopBoundsAttr::name);

    IRRewriter rewriter {op->getContext()};
    rewriter.setInsertionPoint(op);

    // Make these constant felts for now; the actual for op builder will later clean it up
    info.lb = rewriter
                  .create<felt::FeltConstantOp>(
                      op->getLoc(), FeltConstAttr::get(op->getContext(), bounds.getLower())
                  )
                  .getResult();
    info.ub = rewriter
                  .create<felt::FeltConstantOp>(
                      op->getLoc(), FeltConstAttr::get(op->getContext(), bounds.getUpper())
                  )
                  .getResult();
    info.step = rewriter
                    .create<felt::FeltConstantOp>(
                        op->getLoc(), FeltConstAttr::get(op->getContext(), bounds.getStep())
                    )
                    .getResult();
    return info;
  }

  // Now, look for the lb as the corresponding init arg
  info.lb = *op.getInits().drop_front(*info.ivarIndexBefore).begin();

  // Finally, look for the step
  auto nextIvar = *op.getYieldedValues().drop_front(*info.ivarIndexBefore).begin();
  if (auto incOp = nextIvar.getDefiningOp<AddFeltOp>()) {
    if (incOp.getRhs() == ivarAfter) {
      info.step = incOp.getLhs();
    } else if (incOp.getLhs() == ivarAfter) {
      info.step = incOp.getRhs();
    }
  }

  // Make sure the bounds aren't loop-carried, making this not a legal for loop
  // We don't actually need to check the LB because its always passed in via an init block arg
  if (!isRuntimeConstant(*info.ub)) {
    info.ub.reset();
  }
  if (!isRuntimeConstant(*info.step)) {
    info.step.reset();
  }
  return info;
}

static inline llvm::FailureOr<mlir::scf::ForOp>
transformWhileToFor(mlir::scf::WhileOp op, ForOpInfo info, mlir::RewriterBase &rewriter) {
  llzk::ensure(info.success(), "attempting to convert non-constant while loop");

  rewriter.setInsertionPointAfter(op);
  mlir::IRMapping mapping;

  auto copyValue = [op, &rewriter, &mapping](mlir::Value val) -> mlir::Value {
    if (auto *definingOp = val.getDefiningOp();
        definingOp && definingOp->getParentOfType<mlir::scf::WhileOp>() == op) {
      return rewriter.clone(*definingOp, mapping)->getResult(0);
    }
    return val;
  };

  // scf.for bounds have to be `index`, so we might have to cast them here felt -> index before
  // building the op, and cast them back index -> felt inside the body
  auto toIndex = [&rewriter](mlir::Value val) -> mlir::Value {
    if (!isa<FeltType>(val.getType())) {
      return val;
    }
    return rewriter.create<cast::FeltToIndexOp>(val.getLoc(), val).getResult();
  };

  // Emit a prelude setting up the loop bounds
  auto lb = copyValue(*info.lb);
  auto ub = copyValue(*info.ub);
  auto step = copyValue(*info.step);

  // Store the original type of the scf.while's induction var so we can cast back if necessary
  ensure(
      lb.getType() == ub.getType() && lb.getType() == step.getType(),
      "cannot have differing types for loop bounds"
  );
  Type ivarType = lb.getType();
  if (isa<FeltType>(ivarType)) {
    lb = toIndex(lb);
    ub = toIndex(ub);
    step = toIndex(step);
  }

  llvm::SmallVector<mlir::Value> inits;
  for (auto [i, init] : llvm::enumerate(op.getInits())) {
    if (i == info.ivarIndexBefore) {
      continue;
    }
    inits.push_back(init);
  }

  // Build the skeleton of the for loop
  auto forOp = rewriter.create<mlir::scf::ForOp>(op->getLoc(), lb, ub, step, inits);
  rewriter.setInsertionPointToStart(forOp.getBody());

  auto inductionVar = forOp.getInductionVar();
  if (isa<FeltType>(ivarType)) {
    // If the induction var was a felt, we need to cast it back to felt in the scf.for body
    // Note that this means the body of the scf.for might cast it back to index again anyway, but
    // --canonicalize should fix that
    inductionVar =
        rewriter.create<cast::IntToFeltOp>(forOp.getLoc(), ivarType, inductionVar).getResult();
  }

  // // mapping.map(*info.ivar, forOp.getInductionVar()); // I don't think I need this anymore

  // Map uses of the old scf.while "induction var" with the new scf.for induction variable
  auto *whileBody = op.getAfterBody();
  for (size_t i = 0; i < whileBody->getNumArguments(); i++) {
    if (i == info.ivarIndexAfter) {
      mapping.map(whileBody->getArgument(i), inductionVar);
      continue;
    }
    // map uses of the old "iter args" to the scf.for's actual iter_args
    mapping.map(
        whileBody->getArgument(i), forOp.getRegionIterArg(i > info.ivarIndexAfter ? i - 1 : i)
    );
  }

  for (auto &bodyOp : *op.getAfterBody()) {
    // scf.yield is special here because we don't want to yield the induction var to the next
    // iteration...
    if (auto yieldOp = mlir::dyn_cast<mlir::scf::YieldOp>(&bodyOp)) {
      llvm::SmallVector<mlir::Value> valuesToYield;
      for (auto [i, val] : llvm::enumerate(yieldOp.getResults())) {
        if (i == info.ivarIndexAfter) {
          continue;
        }
        // ...but the other yielded values should point to the correct iter_arg
        valuesToYield.push_back(mapping.lookupOrDefault(val));
      }
      if (!valuesToYield.empty()) {
        rewriter.create<mlir::scf::YieldOp>(yieldOp.getLoc(), valuesToYield);
      }
      continue;
    }
    rewriter.clone(bodyOp, mapping);
  }

  // scf.for doesn't explicitly yield its induction var from the final iteration, so we need to
  // reconstruct it
  llvm::SmallVector<mlir::Value> replacedValues;
  for (auto [i, result] : llvm::enumerate(op.getResults())) {
    if (i == info.ivarIndexBefore) {
      // Note that the final value of the induction variable might not actually be the upper bound
      // (e.g. if `step` doesn't divide `(ub - lb)`), but we've already guaranteed earlier that this
      // value isn't being used so it doesn't matter what gets yielded here (the canonicalizer can
      // clean it up). But we still have to yield something to preserve the shape of the op.
      replacedValues.push_back(*info.ub);
      continue;
    }

    replacedValues.push_back(forOp.getResult(i > info.ivarIndexBefore ? i - 1 : i));
  }

  rewriter.replaceOp(op, replacedValues);
  return forOp;
}

class WhileToForPass : public impl::WhileToForPassBase<WhileToForPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cast::CastDialect>();
  }
  void runOnOperation() override {
    IRRewriter rewriter {getOperation().getContext()};
    auto result = getOperation()->walk([&rewriter](scf::WhileOp op) {
      ForOpInfo info = parseInfo(op);
      if (!info.success()) {
        // Ignore loops we can't prove have constant bounds
        return WalkResult::advance();
      }
      if (failed(transformWhileToFor(op, info, rewriter))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createWhileToForPass() { return std::make_unique<WhileToForPass>(); }
} // namespace llzk
