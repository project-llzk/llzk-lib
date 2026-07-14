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

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
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

/// Resolve `call`'s callee if it is a non-external free function; returns
/// null otherwise.
static FuncDefOp resolveFreeCallee(CallOp call, SymbolTableCollection &tables) {
  auto tgtRes = call.getCalleeTarget(tables);
  if (failed(tgtRes)) {
    return nullptr;
  }
  FuncDefOp callee = tgtRes->get();
  if (!isFreeFunction(callee) || callee.isExternal()) {
    return nullptr;
  }
  return callee;
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
    if (FuncDefOp callee = resolveFreeCallee(call, tables)) {
      calls.push_back({call, callee});
    }
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

/// Inlining re-materializes the callee body, including any calls it
/// contains, so a call cycle among free functions never converges. The
/// functions considered must match what the inliner acts on (every
/// non-external free function, including in nested modules), or a cycle
/// could slip past the check and hang the pass.
static LogicalResult checkNoRecursion(ModuleOp mod, SymbolTableCollection &tables) {
  SmallVector<FuncDefOp> funcs;
  mod.walk([&](FuncDefOp func) {
    if (isFreeFunction(func) && !func.isExternal()) {
      funcs.push_back(func);
    }
  });
  llvm::DenseSet<Operation *> remaining(funcs.begin(), funcs.end());
  llvm::DenseMap<Operation *, SmallVector<Operation *>> callees;
  for (FuncDefOp func : funcs) {
    auto &edges = callees[func];
    func.walk([&](CallOp call) {
      if (FuncDefOp callee = resolveFreeCallee(call, tables)) {
        edges.push_back(callee);
      }
    });
  }

  // Release functions whose bodies call no unreleased free function; what
  // remains is on or upstream of a cycle.
  bool changed = true;
  while (changed) {
    changed = false;
    for (FuncDefOp func : funcs) {
      if (!remaining.contains(func)) {
        continue;
      }
      bool callsRemaining = llvm::any_of(callees[func], [&](Operation *callee) {
        return remaining.contains(callee);
      });
      if (!callsRemaining) {
        remaining.erase(func);
        changed = true;
      }
    }
  }
  if (remaining.empty()) {
    return success();
  }

  // A leftover function may merely call into a cycle; follow unreleased
  // edges (every leftover has one) until a repeat to blame an actual cycle
  // member.
  auto isRemaining = [&](Operation *op) { return remaining.contains(op); };
  Operation *cur = *llvm::find_if(funcs, isRemaining);
  llvm::DenseSet<Operation *> visited;
  while (visited.insert(cur).second) {
    cur = *llvm::find_if(callees[cur], isRemaining);
  }
  return cur->emitError("cannot inline recursive free function");
}

class PassImpl : public llzk::impl::InlineFreeFunctionsPassBase<PassImpl> {
  using Base = InlineFreeFunctionsPassBase<PassImpl>;
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SymbolTableCollection tables;
    InlinerInterface inliner(&getContext());

    if (failed(checkNoRecursion(mod, tables))) {
      signalPassFailure();
      return;
    }
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
