//===-- LLZKInlineTopLevelFunctionsPass.cpp ---------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-inline-top-level-functions` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/SymbolTableLLZK.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Transforms/InliningUtils.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

namespace llzk {
#define GEN_PASS_DEF_INLINETOPLEVELFUNCTIONSPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

#define DEBUG_TYPE "llzk-inline-top-level-functions"

using namespace mlir;
using namespace llzk;
using namespace llzk::function;

namespace {

/// A "root" function is a `function.def` whose immediate parent is a
/// `ModuleOp`.
static bool isRootFunction(FuncDefOp func) {
  return llvm::isa_and_nonnull<ModuleOp>(func->getParentOp());
}

/// A `function.call` paired with its callee.
struct TopLevelCall {
  CallOp call;
  FuncDefOp callee;
};

/// Collect every `function.call` in `mod` whose callee is a non-external
/// top-level helper, paired with the resolved callee.
static SmallVector<TopLevelCall> collectTopLevelCalls(ModuleOp mod, SymbolTableCollection &tables) {
  SmallVector<TopLevelCall> topLevelCalls;
  mod.walk([&](CallOp call) {
    auto tgtRes = call.getCalleeTarget(tables);
    if (failed(tgtRes)) {
      return;
    }
    FuncDefOp callee = tgtRes->get();
    if (!isRootFunction(callee) || callee.isExternal()) {
      return;
    }
    topLevelCalls.push_back({call, callee});
  });
  return topLevelCalls;
}

/// Collect every non-external top-level helper in `mod` whose symbol has no
/// remaining uses.
static SmallVector<FuncDefOp> collectUnusedHelpers(ModuleOp mod) {
  SmallVector<FuncDefOp> unusedFunctions;
  for (FuncDefOp func : mod.getOps<FuncDefOp>()) {
    if (!isRootFunction(func) || func.isExternal()) {
      continue;
    }
    if (symbolKnownUseEmpty(func.getOperation(), mod.getOperation())) {
      unusedFunctions.push_back(func);
    }
  }
  return unusedFunctions;
}

class PassImpl : public llzk::impl::InlineTopLevelFunctionsPassBase<PassImpl> {
  using Base = InlineTopLevelFunctionsPassBase<PassImpl>;
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SymbolTableCollection tables;
    InlinerInterface inliner(&getContext());

    if (failed(inlineCalls(mod, tables, inliner))) {
      signalPassFailure();
      return;
    }
    removeUnusedFunctions(mod);
  }

  /// Collects the current top-level helper call sites, then inlines them.
  /// Iterates until all module level calls are inlined.
  LogicalResult
  inlineCalls(ModuleOp mod, SymbolTableCollection &tables, InlinerInterface &inliner) {
    SmallVector<TopLevelCall> callsToInline = collectTopLevelCalls(mod, tables);
    while (!callsToInline.empty()) {
      LLVM_DEBUG({
        llvm::dbgs() << "[llzk-inline-top-level-functions] round found " << callsToInline.size()
                     << " top-level call site(s) to inline\n";
      });

      for (auto [call, callee] : callsToInline) {
        if (failed(inlineCall(inliner, call, callee, callee.getCallableRegion(), true))) {
          return call.emitError("failed to inline top-level function.call");
        }
        call.erase();
      }

      callsToInline = collectTopLevelCalls(mod, tables);
    }
    return success();
  }

  /// Erase top-level helpers that are no longer referenced anywhere in the
  /// module.
  void removeUnusedFunctions(ModuleOp mod) {
    SmallVector<FuncDefOp> toErase = collectUnusedHelpers(mod);
    while (!toErase.empty()) {
      for (FuncDefOp func : toErase) {
        func.erase();
      }
      toErase = collectUnusedHelpers(mod);
    }
  }
};
} // namespace
