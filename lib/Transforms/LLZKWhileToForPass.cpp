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
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
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
  std::optional<Value> lb, ub, step, ivar;
  std::optional<size_t> ivar_index;
  bool success() const {
    return lb.has_value() && ub.has_value() && step.has_value() && ivar.has_value() &&
           ivar_index != static_cast<size_t>(-1);
  }
};

static inline ForOpInfo parseInfo(WhileOp op) {
  ForOpInfo info;

  auto condition = op.getConditionOp().getCondition();
  if (auto cmp = condition.getDefiningOp<CmpOp>(); cmp.getPredicate() == FeltCmpPredicate::LT) {
    // We found the ivar and the ub
    info.ivar = cmp.getLhs();
    info.ub = cmp.getRhs();
  } else {
    return info;
  }

  // Find which # block arg the ivar is
  for (auto [i, arg] : llvm::enumerate(op.getConditionOp().getArgs())) {
    if (arg == *info.ivar) {
      info.ivar_index.emplace(i);
      break;
    }
  }
  if (!info.ivar_index.has_value()) {
    return info;
  }

  // Now, look for the lb as the corresponding init arg
  info.lb = *op.getInits().drop_front(*info.ivar_index).begin();

  // Finally, look for the step
  auto nextIvar = *op.getYieldedValues().drop_front(*info.ivar_index).begin();
  if (auto incOp = nextIvar.getDefiningOp<AddFeltOp>()) {
    if (incOp.getLhs().getDefiningOp<FeltConstantOp>()) {
      info.step = incOp.getLhs();
    } else if (incOp.getRhs().getDefiningOp<FeltConstantOp>()) {
      info.step = incOp.getRhs();
    }
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
    if (i == info.ivar_index) {
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
    if (i == info.ivar_index) {
      mapping.map(whileBody->getArgument(i), inductionVar);
      continue;
    }
    // map uses of the old "iter args" to the scf.for's actual iter_args
    mapping.map(whileBody->getArgument(i), forOp.getRegionIterArg(i > info.ivar_index ? i - 1 : i));
  }

  for (auto &bodyOp : *op.getAfterBody()) {
    // scf.yield is special here because we don't want to yield the induction var to the next
    // iteration...
    if (auto yieldOp = mlir::dyn_cast<mlir::scf::YieldOp>(&bodyOp)) {
      llvm::SmallVector<mlir::Value> valuesToYield;
      for (auto [i, val] : llvm::enumerate(yieldOp.getResults())) {
        if (i == info.ivar_index) {
          continue;
        }
        // ...but the other yielded values should point to the correct iter_arg
        valuesToYield.push_back(mapping.lookupOrDefault(val));
      }
      rewriter.create<mlir::scf::YieldOp>(yieldOp.getLoc(), valuesToYield);
      continue;
    }
    rewriter.clone(bodyOp, mapping);
  }

  // scf.for doesn't explicitly yield its induction var from the final iteration, so we need to
  // reconstruct it
  llvm::SmallVector<mlir::Value> replacedValues;
  for (auto [i, result] : llvm::enumerate(op.getResults())) {
    if (i == info.ivar_index) {
      // TODO: this might not actually be correct if, e.g., `step` doesn't divide `(ub - lb)`
      // So it would be better to explicitly compute what the final value will be when the loop
      // terminates and yield that instead
      replacedValues.push_back(*info.ub);
      continue;
    }

    replacedValues.push_back(forOp.getResult(i > info.ivar_index ? i - 1 : i));
  }

  rewriter.replaceOp(op, replacedValues);
  return forOp;
}

class WhileToForPass : public impl::WhileToForPassBase<WhileToForPass> {
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
