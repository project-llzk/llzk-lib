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

    auto result = analyzeStruct(structDef);
    if (failed(result)) {
      signalPassFailure();
    }
    const auto &[overwrites, written] = *result;

    for (auto member : structDef.getMemberDefs()) {
      if (!written.contains(member.getSymName())) {
        member->emitWarning() << "member may not be written to";
      }
    }

    for (auto [first, over] : overwrites) {
      auto diag = over->emitWarning()
                  << "may overwrite struct member '@" << over.getMemberName() << '\'';
      diag.attachNote(first.getLoc()) << "previously written to here";
      diag.report();
    }

    markAllAnalysesPreserved();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> llzk::createMemberWriteValidatorPass() {
  return std::make_unique<MemberWriteValidatorPass>();
};
