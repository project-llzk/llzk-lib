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
/// This file identifies scf.while loops that can be converted to scf.for loops and performs the
/// conversion. A while loop can be converted if:
/// * The scf.condition directly forwards `before` block arguments to the `after` block
/// * The `before` block has an argument %arg (the "induction variable") such that:
///   * The scf.condition condition has the form `bool.cmp lt(%arg, %upper_bound)`
///   * The value yielded from the after block has the form `felt.add %arg, %step`
///     or is a constant that makes a constant-bound loop execute exactly once
/// * `%upper_bound` and `%step` do not depend on any loop-carried variables
/// * The final yielded value of the induction variable does not have uses outside the loop
///
/// This pass begins by identifying the induction variable, lower and upper bounds, and step
/// (`ForOp::parseInfo`), then materializes a for loop with these parameters and copies the `before`
/// and `after` blocks into the body, drops the yielded induction variable, and remaps uses of the
/// other yielded values (`transformWhileToFor`).
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Cast/Util/OverflowSemantics.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Transforms/ConversionUtils.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/Field.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/Support/Debug.h>
#include <llvm/Support/Error.h>

#include <algorithm>
#include <cassert>
#include <limits>

namespace llzk {
#define GEN_PASS_DEF_WHILETOFORPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

#define DEBUG_TYPE "while-to-for"

using namespace mlir;
using namespace scf;
using namespace llzk::boolean;
using namespace llzk::felt;

namespace {

struct ForOpInfo {
  // SSA values holding the loop bounds
  std::optional<Value> lb, ub, step;

  // Block argument index of the induction variable in the *before* block
  std::optional<size_t> ivarIndexBefore;
  // Block argument index of the induction variable in the *after* block
  std::optional<size_t> ivarIndexAfter;

  bool success() const {
    return lb.has_value() && ub.has_value() && step.has_value() && ivarIndexBefore.has_value() &&
           ivarIndexAfter.has_value();
  }
};

static inline ForOpInfo parseInfo(WhileOp op) {
  auto reportFailureReason = [&op](Twine reason) {
    return op.emitWarning("'llzk-while-to-for' failed to transform loop: ") << reason;
  };

  ForOpInfo info;

  if (!std::equal(
          op.getBeforeArguments().begin(), op.getBeforeArguments().end(),
          op.getConditionOp().getArgs().begin()
      )) {
    reportFailureReason("block arguments not passed through from preamble to body");
    return info;
  }

  auto condition = op.getConditionOp().getCondition();
  Value ivarBefore;
  if (auto cmp = condition.getDefiningOp<CmpOp>();
      cmp && cmp.getPredicate() == FeltCmpPredicate::LT) {
    // We found the ivar and the ub
    ivarBefore = cmp.getLhs();
    info.ub = cmp.getRhs();
  } else {
    reportFailureReason("could not identify an upper bound");
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
    reportFailureReason("could not identify an induction variable");
    return info;
  }

  // If the yielded final value of the induction variable has any uses, we can't cleanly transform
  // this to an scf.for (which doesn't explicitly yield its induction var) without doing some extra
  // computation. Lets just conservatively bail out in that case
  auto yieldedIVar = op->getResults().drop_front(*info.ivarIndexAfter).front();
  if (!yieldedIVar.use_empty()) {
    auto report = reportFailureReason("final ivar value unsafe to drop");
    for (auto *use : yieldedIVar.getUsers()) {
      report.attachNote(use->getLoc()) << "used here";
    }
    return info;
  }

  OpBuilder builder(op);
  // Materialize a felt bound before the loop in the supplied induction variable type.
  auto createBound = [&builder, &op](const llvm::DynamicAPInt &value, FeltType type) -> Value {
    auto attr = FeltConstAttr::get(builder.getContext(), value, type);
    return FeltConstantOp::create(builder, op.getLoc(), attr).getResult();
  };

  // We need an induction variable anyway, but if the loop has {llzk.loopbounds} we can skip trying
  // to parse the rest of the bounds and just materialize constants
  if (op->hasAttr(llzk::LoopBoundsAttr::name)) {
    auto bounds = op->getAttrOfType<llzk::LoopBoundsAttr>(llzk::LoopBoundsAttr::name);
    // Signed annotations must not become field representatives before conversion to index.
    // This conversion only supports nonnegative bounds and a positive scf.for step.
    if (bounds.getLower() < 0 || bounds.getUpper() < 0 || bounds.getStep() <= 0) {
      reportFailureReason("loopbounds must have nonnegative bounds and a positive step");
      return info;
    }
    // Use the same nonnegative index range as cast.toindex assert.
    auto fitsIndex = [](const llvm::DynamicAPInt &value) {
      return llzk::cast::applyFeltToIndexOverflow(
                 value, IndexType::kInternalStorageBitWidth, llzk::cast::OverflowSemantics::ASSERT
      )
          .has_value();
    };
    if (!fitsIndex(bounds.getLower()) || !fitsIndex(bounds.getUpper()) ||
        !fitsIndex(bounds.getStep())) {
      reportFailureReason("loopbounds values cannot be represented as nonnegative indices");
      return info;
    }
    // scf.for also requires the value after the final increment to fit, even though it is
    // outside the iteration range. Compute it with mathematical integers to avoid overflow.
    if (bounds.getLower() < bounds.getUpper()) {
      auto distance = bounds.getUpper() - bounds.getLower();
      auto tripCount = (distance + bounds.getStep() - 1) / bounds.getStep();
      auto end = bounds.getLower() + tripCount * bounds.getStep();
      if (!fitsIndex(end)) {
        reportFailureReason("loopbounds final induction value cannot be represented as an index");
        return info;
      }
    }

    // Annotations are mathematical integers, independent of the induction variable's field.
    auto createIndexBound = [&builder, &op](const llvm::DynamicAPInt &value) -> Value {
      return arith::ConstantIndexOp::create(
          builder, op.getLoc(), llvm::cantFail(llzk::checkedToInt64(value))
      );
    };
    info.lb = createIndexBound(bounds.getLower());
    info.ub = createIndexBound(bounds.getUpper());
    info.step = createIndexBound(bounds.getStep());
    return info;
  }

  Value ivarAfter = op.getAfterArguments().drop_front(*info.ivarIndexAfter).front();

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

  // Constant propagation can replace the final increment of a one-iteration loop
  // with a constant. Only accept it when the initial condition is true and the
  // yielded value makes the next condition false; otherwise it may loop forever.
  if (!info.step.has_value()) {
    if (auto type = dyn_cast<FeltType>(ivarAfter.getType()); type && type.hasField()) {
      auto lower = info.lb->getDefiningOp<FeltConstantOp>();
      auto upper = info.ub->getDefiningOp<FeltConstantOp>();
      auto next = nextIvar.getDefiningOp<FeltConstantOp>();
      if (lower && upper && next) {
        // In valid IR, all 3 constants must have the same type as the induction variable
        // and we checked above that it has a field so they can all be reduced via the field.
        assert(lower.getType() == type && "lower bound type must match induction variable");
        assert(upper.getType() == type && "upper bound type must match induction variable");
        assert(next.getType() == type && "yielded constant type must match induction variable");
        const llzk::Field &field = type.getField();
        auto lo = field.reduce(lower.getValue().getRawValue());
        auto hi = field.reduce(upper.getValue().getRawValue());
        auto end = field.reduce(next.getValue().getRawValue());
        if (lo < hi && end >= hi) {
          // The for loop casts its bounds and step to index. Since 0 <= lo < hi,
          // bounding hi also bounds lo and hi - lo.
          if (hi > std::numeric_limits<int64_t>::max()) {
            reportFailureReason("upper bound cannot be represented as a nonnegative index");
            return info;
          }
          // A while loop also executes the condition region on its final check.
          // Dropping that execution is only safe for pure preamble operations.
          bool purePreamble =
              llvm::all_of(op.getBefore().front().without_terminator(), [](Operation &preambleOp) {
            return mlir::isPure(&preambleOp);
          });
          if (!purePreamble) {
            reportFailureReason("condition region contains operations that are not pure");
            return info;
          }
          // Use canonical representatives, including when source literals are negative.
          info.lb = createBound(lo, type);
          info.ub = createBound(hi, type);
          info.step = createBound(hi - lo, type);
        }
      }
    }
  }

  if (!info.step.has_value()) {
    reportFailureReason("could not identify step");
    return info;
  }

  // Make sure the bounds aren't loop-carried, making this not a legal for loop
  // We don't actually need to check the LB because its always passed in via an init block arg
  std::function<bool(Value, InFlightDiagnostic &)> isRuntimeConstant;
  isRuntimeConstant = [&op, &isRuntimeConstant](Value val, InFlightDiagnostic &reporter) -> bool {
    // The value can't come from a block argument owned by the while loop
    if (auto blockArg = dyn_cast<BlockArgument>(val)) {
      reporter.attachNote(blockArg.getLoc()) << "depends on loop-carried value";
      return blockArg.getParentBlock()->getParentOp() != op;
    }

    // The value also can't depend on any value that comes from a block argument owned by the while
    // loop
    return !llvm::any_of(val.getDefiningOp()->getOperands(), [&](Value operand) {
      return !isRuntimeConstant(operand, reporter);
    });
  };

  auto ubReport = reportFailureReason("upper bound may not be constant");
  if (!isRuntimeConstant(*info.ub, ubReport)) {
    return ForOpInfo {};
  }
  ubReport.abandon();

  auto stepReport = reportFailureReason("step may not be constant");
  if (!isRuntimeConstant(*info.step, stepReport)) {
    return ForOpInfo {};
  }
  stepReport.abandon();

  return info;
}

static inline FailureOr<scf::ForOp>
transformWhileToFor(scf::WhileOp op, ForOpInfo info, RewriterBase &rewriter) {
  llzk::ensure(info.success(), "attempting to convert non-constant while loop");

  rewriter.setInsertionPointAfter(op);
  IRMapping mapping;

  auto copyIfNeeded = [op, &rewriter, &mapping](Value val) -> Value {
    if (auto *definingOp = val.getDefiningOp();
        definingOp && definingOp->getParentOfType<scf::WhileOp>() == op) {
      return rewriter.clone(*definingOp, mapping)->getResult(0);
    }
    return val;
  };

  // scf.for bounds have to be `index`, so we might have to cast them here felt -> index before
  // building the op, and cast them back index -> felt inside the body
  auto toIndex = [&rewriter](Value val) -> Value {
    if (!isa<FeltType>(val.getType())) {
      return val;
    }
    return llzk::cast::FeltToIndexOp::create(rewriter, val.getLoc(), val).getResult();
  };

  // Emit a prelude setting up the loop bounds
  auto lb = copyIfNeeded(*info.lb);
  auto ub = copyIfNeeded(*info.ub);
  auto step = copyIfNeeded(*info.step);

  // Store the original type of the scf.while's induction var so we can cast back if necessary
  llzk::ensure(
      lb.getType() == ub.getType() && lb.getType() == step.getType(),
      "cannot have differing types for loop bounds"
  );
  Type ivarType = op.getBeforeArguments()[*info.ivarIndexBefore].getType();
  lb = toIndex(lb);
  ub = toIndex(ub);
  step = toIndex(step);

  SmallVector<Value> inits;
  for (auto [i, init] : llvm::enumerate(op.getInits())) {
    if (i == info.ivarIndexBefore) {
      continue;
    }
    inits.push_back(init);
  }

  // Build the skeleton of the for loop
  auto forOp = llzk::preserveDiscardableAttrs(
      op, scf::ForOp::create(rewriter, op->getLoc(), lb, ub, step, inits)
  );
  rewriter.setInsertionPointToStart(forOp.getBody());

  auto inductionVar = forOp.getInductionVar();
  if (isa<FeltType>(ivarType)) {
    // Annotated integer IVs map to their field representatives in the body, including when
    // they reach or exceed the modulus. Inferred felt bounds retain their checked cast.
    auto overflow = op->hasAttr(llzk::LoopBoundsAttr::name) ? llzk::cast::OverflowSemantics::WRAP
                                                            : llzk::cast::OverflowSemantics::ASSERT;
    inductionVar =
        llzk::cast::IntToFeltOp::create(rewriter, forOp.getLoc(), ivarType, overflow, inductionVar)
            .getResult();
  }

  // Start by mapping the `before` block to the loop body
  // Each block arg of the before block should get mapped to the corresponding iter_arg, with the
  // exception of the induction var which should get mapped to the induction var
  auto *whilePreamble = op.getBeforeBody();
  for (size_t i = 0; i < whilePreamble->getNumArguments(); i++) {
    if (i == info.ivarIndexBefore) {
      mapping.map(whilePreamble->getArgument(i), inductionVar);
      continue;
    }
    mapping.map(
        whilePreamble->getArgument(i), forOp.getRegionIterArg(i > info.ivarIndexBefore ? i - 1 : i)
    );
  }

  // Emit the preamble into the for loop body
  for (auto &preambleOp : *whilePreamble) {
    // Don't emit the scf.condition; rather, use it to update the mapping in preparation for the
    // loop body
    if (auto condOp = dyn_cast<scf::ConditionOp>(&preambleOp)) {
      for (auto [value, blockArg] : llvm::zip(condOp.getArgs(), op.getAfterArguments())) {
        // TODO: maybe this isn't transitive?
        mapping.map(blockArg, mapping.lookupOrDefault(value));
      }
      continue;
    }
    rewriter.clone(preambleOp, mapping);
  }

  auto *whileBody = op.getAfterBody();

  for (auto &bodyOp : *whileBody) {
    // scf.yield is special here because we don't want to yield the induction var to the next
    // iteration...
    if (auto yieldOp = dyn_cast<scf::YieldOp>(&bodyOp)) {
      SmallVector<Value> valuesToYield;
      for (auto [i, val] : llvm::enumerate(yieldOp.getResults())) {
        if (i == info.ivarIndexAfter) {
          continue;
        }
        // ...but the other yielded values should point to the correct iter_arg
        valuesToYield.push_back(mapping.lookupOrDefault(val));
      }
      if (!valuesToYield.empty()) {
        scf::YieldOp::create(rewriter, yieldOp.getLoc(), valuesToYield);
      }
      continue;
    }
    rewriter.clone(bodyOp, mapping);
  }

  // Match the original result list, using a type-correct placeholder for the unused final IV.
  SmallVector<Value> replacedValues;
  for (auto [i, result] : llvm::enumerate(op.getResults())) {
    if (i == info.ivarIndexBefore) {
      // parseInfo guarantees this result is unused. The original initializer has its type,
      // even when the annotated loop bounds were materialized as index constants.
      replacedValues.push_back(op.getInits()[*info.ivarIndexBefore]);
      continue;
    }

    replacedValues.push_back(forOp.getResult(i > info.ivarIndexBefore ? i - 1 : i));
  }

  rewriter.replaceOp(op, replacedValues);
  return forOp;
}

class PassImpl : public llzk::impl::WhileToForPassBase<PassImpl> {
  using Base = WhileToForPassBase<PassImpl>;
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, llzk::cast::CastDialect>();
  }
  void runOnOperation() override {
    Operation *op = getOperation();
    IRRewriter rewriter {op->getContext()};
    auto result = op->walk([&rewriter](scf::WhileOp loop) -> WalkResult {
      ForOpInfo info = parseInfo(loop);
      if (!info.success()) {
        // Ignore loops we can't prove have constant bounds
        return WalkResult::advance();
      }
      return LogicalResult(transformWhileToFor(loop, info, rewriter));
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace
