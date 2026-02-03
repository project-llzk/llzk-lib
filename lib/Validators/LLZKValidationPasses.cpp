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
    FuncDefOp computeFunc = structDef.getComputeFuncOp();

    // Initialize map with all member names mapped to nullptr (i.e., no write found).
    llvm::StringMap<MemberWriteOp> memberNameToWriteOp;
    for (MemberDefOp x : structDef.getMemberDefs()) {
      memberNameToWriteOp[x.getSymName()] = nullptr;
    }
    // Search the function body for writes, store them in the map and emit warning if multiple
    // writes to the same member are found.
    for (Block &block : computeFunc.getBody()) {
      for (Operation &op : block) {
        if (MemberWriteOp write = dyn_cast<MemberWriteOp>(op)) {
          // `MemberWriteOp::verifySymbolUses()` ensures MemberWriteOp only target the containing
          // "self" struct. That means the target of the MemberWriteOp must be in
          // `memberNameToWriteOp` so using 'at()' will not abort.
          assert(structDef.getType() == write.getComponent().getType());
          StringRef writeToMemberName = write.getMemberName();
          if (MemberWriteOp earlierWrite = memberNameToWriteOp.at(writeToMemberName)) {
            auto diag = write.emitWarning().append(
                "found multiple writes to '", MemberDefOp::getOperationName(), "' named \"@",
                writeToMemberName, '"'
            );
            diag.attachNote(earlierWrite.getLoc()).append("earlier write here");
            diag.report();
          }
          memberNameToWriteOp[writeToMemberName] = write;
        }
      }
    }
    // Finally, report a warning if any member was not written at all.
    for (auto &[a, b] : memberNameToWriteOp) {
      if (!b) {
        computeFunc.emitWarning()
            .append(
                '\'', FuncDefOp::getOperationName(), "' op \"@", FUNC_NAME_COMPUTE,
                "\" missing write to '", MemberDefOp::getOperationName(), "' named \"@", a, '"'
            )
            .report();
      }
    }

    markAllAnalysesPreserved();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> llzk::createMemberWriteValidatorPass() {
  return std::make_unique<MemberWriteValidatorPass>();
};
