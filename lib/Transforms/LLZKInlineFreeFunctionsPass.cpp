//===-- LLZKInlineFreeFunctionsPass.cpp -------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-inline-free-functions` pass.
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
#define GEN_PASS_DEF_INLINEFREEFUNCTIONSPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

#define DEBUG_TYPE "llzk-inline-free-functions"

using namespace mlir;
using namespace llzk;
using namespace llzk::function;

namespace {

/// A free function is a `function.def` whose immediate parent is a
/// `ModuleOp`.
static bool isFreeFunction(FuncDefOp func) {
  return llvm::isa_and_nonnull<ModuleOp>(func->getParentOp());
}

/// A `function.call` paired with its callee.
struct FreeFunctionCall {
  CallOp call;
  FuncDefOp callee;
};

/// Collect every `function.call` in `mod` whose callee is a non-external
/// free function, paired with the resolved callee.
static SmallVector<FreeFunctionCall>
collectFreeFunctionCalls(ModuleOp mod, SymbolTableCollection &tables) {
  SmallVector<FreeFunctionCall> calls;
  mod.walk([&](CallOp call) {
    auto tgtRes = call.getCalleeTarget(tables);
    if (failed(tgtRes)) {
      return;
    }
    FuncDefOp callee = tgtRes->get();
    if (!isFreeFunction(callee) || callee.isExternal()) {
      return;
    }
    calls.push_back({call, callee});
  });
  return calls;
}

/// Collect every non-external free function in `mod` whose symbol has no
/// remaining uses.
static SmallVector<FuncDefOp> collectUnusedHelpers(ModuleOp mod) {
  SmallVector<FuncDefOp> unusedFunctions;
  for (FuncDefOp func : mod.getOps<FuncDefOp>()) {
    if (!isFreeFunction(func) || func.isExternal()) {
      continue;
    }
    if (symbolKnownUseEmpty(func.getOperation(), mod.getOperation())) {
      unusedFunctions.push_back(func);
    }
  }
  return unusedFunctions;
}

class PassImpl : public llzk::impl::InlineFreeFunctionsPassBase<PassImpl> {
  using Base = InlineFreeFunctionsPassBase<PassImpl>;
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

  /// Collects the current free-function call sites, then inlines them.
  /// Iterates until all such calls are inlined.
  LogicalResult
  inlineCalls(ModuleOp mod, SymbolTableCollection &tables, InlinerInterface &inliner) {
    SmallVector<FreeFunctionCall> callsToInline = collectFreeFunctionCalls(mod, tables);
    while (!callsToInline.empty()) {
      LLVM_DEBUG({
        llvm::dbgs() << "[" DEBUG_TYPE "] round found " << callsToInline.size()
                     << " free-function call site(s) to inline\n";
      });

      for (auto [call, callee] : callsToInline) {
        if (failed(inlineCall(inliner, call, callee, callee.getCallableRegion(), true))) {
          return call.emitError("failed to inline free function call");
        }
        call.erase();
      }

      callsToInline = collectFreeFunctionCalls(mod, tables);
    }
    return success();
  }

  /// Erase free functions that are no longer referenced anywhere in the
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
