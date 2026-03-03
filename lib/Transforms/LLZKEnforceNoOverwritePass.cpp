//===-- LLZKEnforceNoOverwritePass.cpp --------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-enforce-no-overwrite` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/AnalysisUtil.h"
#include "llzk/Analysis/MemberOverwriteAnalysis.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <mlir/Analysis/DataFlowFramework.h>

#include <llvm/Support/Debug.h>

#include <memory>

namespace llzk {
#define GEN_PASS_DECL_ENFORCENOMEMBEROVERWRITEPASS
#define GEN_PASS_DEF_ENFORCENOMEMBEROVERWRITEPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

#define DEBUG_TYPE "llzk-enforce-no-overwrites-pass"

using std::make_unique;

using namespace mlir;

namespace llzk {

using namespace function;
using namespace component;

class EnforceNoMemberOverwritePass
    : public llzk::impl::EnforceNoMemberOverwritePassBase<EnforceNoMemberOverwritePass> {

  bool containsOverwrite(FuncDefOp funcDef) {
    DataFlowSolver solver;
    dataflow::loadRequiredAnalyses(solver);
    solver.load<MemberOverwriteAnalysis>();
    if (failed(solver.initializeAndRun(funcDef))) {
      signalPassFailure();
    }

    auto &funcBody = funcDef.getBody();
    if (funcBody.empty()) {
      // No overwrites if theres nothing
      return false;
    }

    auto *returnOp = funcBody.back().getTerminator();
    const auto *lattice =
        solver.lookupState<MemberOverwriteLattice>(solver.getProgramPointAfter(returnOp));
    if (!lattice) {
      signalPassFailure();
    }

    lattice->emitOverwriteErrors();
    return lattice->hasOverwrites();
  }

  void runOnOperation() override {
    getOperation()->walk([this](FuncDefOp funcDef) {
      if (containsOverwrite(funcDef)) {
        // TODO: we could instead do something to repair the overwrite
        signalPassFailure();
      }
    });
  }
};

std::unique_ptr<mlir::Pass> createNoOverwritesPass() {
  return make_unique<EnforceNoMemberOverwritePass>();
}
} // namespace llzk
