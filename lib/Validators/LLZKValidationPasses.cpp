//===-- LLZKValidationPasses.cpp - LLZK validation passes -------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation for the `-llzk-validate-member-writes`
/// pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/AnalysisUtil.h"
#include "llzk/Analysis/MemberOverwriteAnalysis.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Validators/LLZKValidationPasses.h"

#include <mlir/IR/BuiltinOps.h>

// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_MEMBERWRITEVALIDATORPASS
#include "llzk/Validators/LLZKValidationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;
using namespace llzk::component;
using namespace llzk::function;

namespace {
class MemberWriteValidatorPass
    : public llzk::impl::MemberWriteValidatorPassBase<MemberWriteValidatorPass> {
  void runOnOperation() override {
    StructDefOp structDef = getOperation();
    FuncDefOp computeOrProductFunc = structDef.getComputeFuncOp();
    if (!computeOrProductFunc) {
      computeOrProductFunc = structDef.getProductFuncOp();
    }

    DataFlowSolver solver {DataFlowConfig {}.setInterprocedural(false)};
    llzk::dataflow::loadRequiredAnalyses(solver);
    solver.load<MemberOverwriteAnalysis>();
    if (failed(solver.initializeAndRun(computeOrProductFunc))) {
      signalPassFailure();
    }

    auto &funcBody = computeOrProductFunc.getBody();
    if (funcBody.empty()) {
      // No overwrites if theres nothing
      return;
    }

    auto *returnOp = funcBody.back().getTerminator();
    const auto *lattice =
        solver.lookupState<MemberOverwriteLattice>(solver.getProgramPointAfter(returnOp));
    if (!lattice) {
      signalPassFailure();
    }

    for (auto member : structDef.getMemberDefs()) {
      lattice->ensureWritten(member);
    }

    lattice->emitOverwriteErrors();

    markAllAnalysesPreserved();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> llzk::createMemberWriteValidatorPass() {
  return std::make_unique<MemberWriteValidatorPass>();
};
