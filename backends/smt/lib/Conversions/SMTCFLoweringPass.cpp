//===-- SMTCFLoweringPass.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "smt/Conversions/ConversionPasses.h"

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/Include/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/SMT/IR/SMTOps.h"
#include "llzk/Dialect/String/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Dialect.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>

#include <utility>

namespace llzk {
namespace smt {
#define GEN_PASS_DEF_SMTCFLOWERINGPASS
#include "smt/Conversions/ConversionPasses.h.inc"
} // namespace smt

using namespace mlir;

class SMTCFLoweringPass : public smt::impl::SMTCFLoweringPassBase<SMTCFLoweringPass> {
  // The condition of the ifOp might be an `unrealized_conversion_cast` from some !smt.bool
  // to an i1, so we need to see through that
  Value getCondition(scf::IfOp ifOp) {
    Value condition = ifOp.getCondition();
    while (auto op = dyn_cast<mlir::UnrealizedConversionCastOp>(condition.getDefiningOp())) {
      condition = op.getOperand(0);
    }
    return condition;
  }
  IRMapping flatten(scf::IfOp ifOp, RewriterBase &rewriter) {
    IRMapping mapping;

    rewriter.setInsertionPoint(ifOp);
    for (auto &op : ifOp.getThenRegion().front().without_terminator()) {
      rewriter.clone(op, mapping);
    }
    if (!ifOp.getElseRegion().empty()) {
      for (auto &op : ifOp.getElseRegion().front().without_terminator()) {
        rewriter.clone(op, mapping);
      }
    }
    return mapping;
  }

public:
  LogicalResult processContainedAsserts(scf::IfOp ifOp, RewriterBase &rewriter) {
    Value condition = getCondition(ifOp);

    SmallVector<smt::AssertOp> thenAssertions, elseAssertions;
    ifOp.getThenRegion().walk([&](smt::AssertOp op) { thenAssertions.push_back(op); });
    ifOp.getElseRegion().walk([&](smt::AssertOp op) { elseAssertions.push_back(op); });

    if (thenAssertions.empty() && elseAssertions.empty()) {
      // No assertions, nothing to do!
      return success();
    }

    for (auto assertion : thenAssertions) {
      rewriter.setInsertionPoint(assertion);
      auto implies =
          rewriter.create<smt::ImpliesOp>(assertion.getLoc(), condition, assertion.getInput());
      assertion.getInputMutable().assign(implies.getResult());
    }

    if (elseAssertions.empty()) {
      // Don't bother materializing an SSA value for "~b" if there's no use for it
      return success();
    }

    rewriter.setInsertionPoint(ifOp);
    Value notCondition;
    if (auto notOp = dyn_cast<smt::NotOp>(condition.getDefiningOp())) {
      // Don't generate (not (not x))
      notCondition = notOp.getInput();
    } else {
      notCondition = rewriter.create<smt::NotOp>(ifOp.getLoc(), condition).getResult();
    }

    for (auto assertion : elseAssertions) {
      rewriter.setInsertionPoint(assertion);
      auto implies =
          rewriter.create<smt::ImpliesOp>(assertion.getLoc(), notCondition, assertion.getInput());
      assertion.getInputMutable().assign(implies.getResult());
    }

    return success();
  }

  LogicalResult processYieldedResults(scf::IfOp ifOp, RewriterBase &rewriter) {

    if (ifOp->getNumResults() == 0) {
      flatten(ifOp, rewriter);
      ifOp.erase();
      return success();
    }

    auto *thenBlock = &ifOp.getThenRegion().front();
    auto *elseBlock = &ifOp.getElseRegion().front();

    rewriter.setInsertionPoint(ifOp);

    SmallVector<std::pair<Value, Value>> yieldedValues;
    for (auto [v1, v2] : llvm::zip(
             thenBlock->getTerminator()->getOperands(), elseBlock->getTerminator()->getOperands()
         )) {
      yieldedValues.push_back({v1, v2});
    }

    auto mapping = flatten(ifOp, rewriter);

    SmallVector<Value> muxedValues;
    for (auto [v1, v2] : yieldedValues) {
      auto iteOp = rewriter.create<smt::IteOp>(
          ifOp.getLoc(), getCondition(ifOp), mapping.lookupOrDefault(v1),
          mapping.lookupOrDefault(v2)
      );
      muxedValues.push_back(iteOp.getResult());
    }

    rewriter.replaceAllOpUsesWith(ifOp, muxedValues);
    ifOp.erase();

    return success();
  }
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter {&getContext()};
    mod.walk([this, &rewriter](scf::IfOp ifOp) {
      if (failed(processContainedAsserts(ifOp, rewriter))) {
        signalPassFailure();
      }
      if (failed(processYieldedResults(ifOp, rewriter))) {
        signalPassFailure();
      }
    });
  }
};

namespace smt {

std::unique_ptr<mlir::Pass> createSMTCFLoweringPass() {
  return std::make_unique<SMTCFLoweringPass>();
}

} // namespace smt
} // namespace llzk
