//===-- SMTCFLoweringPass.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/Include/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/String/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Dialect.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>

#include "smt/Conversions/ConversionPasses.h"
#include "smt/Dialect/IR/SMTOps.h"

namespace llzk {
namespace smt {
#define GEN_PASS_DEF_SMTCFLOWERINGPASS
#include "smt/Conversions/ConversionPasses.h.inc"
} // namespace smt

using namespace mlir;

namespace {

class SMTCFLoweringPass : public smt::impl::SMTCFLoweringPassBase<SMTCFLoweringPass> {
public:
  LogicalResult processContainedAsserts(scf::IfOp ifOp, RewriterBase &rewriter) {
    // The condition of the ifOp might be an `unrealized_conversion_cast` from some !smt.bool
    // to an i1, so we need to see through that
    Value condition = ifOp.getCondition();
    while (auto op = dyn_cast<mlir::UnrealizedConversionCastOp>(condition.getDefiningOp())) {
      condition = op.getOperand(0);
    }

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
    auto notCondition = rewriter.create<smt::NotOp>(ifOp.getLoc(), condition).getResult();

    for (auto assertion : elseAssertions) {
      rewriter.setInsertionPoint(assertion);
      auto implies =
          rewriter.create<smt::ImpliesOp>(assertion.getLoc(), notCondition, assertion.getInput());
      assertion.getInputMutable().assign(implies.getResult());
    }

    return success();
  }
  LogicalResult processYieldedResults(scf::IfOp, RewriterBase &) { return failure(); }
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter {&getContext()};
    mod.walk([this, &rewriter](scf::IfOp ifOp) {
      if (failed(processContainedAsserts(ifOp, rewriter))) {
        signalPassFailure();
      }
    });
  }
};

} // namespace

namespace smt {

std::unique_ptr<mlir::Pass> createSMTCFLoweringPass() {
  return std::make_unique<SMTCFLoweringPass>();
}

} // namespace smt
} // namespace llzk
