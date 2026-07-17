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

#include "llzk/Analysis/CallGraphAnalyses.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/SymbolTableLLZK.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Transforms/InliningUtils.h>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SCCIterator.h>
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

/// Free functions that participate in a call cycle: inlining one would
/// re-materialize its calls forever, so their call sites are skipped and
/// their definitions left untouched. A free function whose cycle passes
/// through struct functions is skipped too — over-conservative but safe.
static llvm::DenseSet<Operation *> collectRecursiveFunctions(const llzk::CallGraph &cg) {
  llvm::DenseSet<Operation *> recursive;
  for (auto scc = llvm::scc_begin(&cg); !scc.isAtEnd(); ++scc) {
    if (scc->size() == 1 && !scc.hasCycle()) {
      continue;
    }
    for (const llzk::CallGraphNode *node : *scc) {
      if (!node->isExternal()) {
        recursive.insert(node->getCalledFunction().getOperation());
      }
    }
  }
  return recursive;
}

/// Resolve `call`'s callee if it is a non-external, non-skipped free
/// function; returns null otherwise.
static FuncDefOp resolveFreeCallee(
    CallOp call, SymbolTableCollection &tables, const llvm::DenseSet<Operation *> &skippedCallees
) {
  auto tgtRes = call.getCalleeTarget(tables);
  if (failed(tgtRes)) {
    return nullptr;
  }
  FuncDefOp callee = tgtRes->get();
  if (!isFreeFunction(callee) || callee.isExternal() || skippedCallees.contains(callee)) {
    return nullptr;
  }
  return callee;
}

/// A `function.call` paired with its callee.
struct FreeFunctionCall {
  CallOp call;
  FuncDefOp callee;
};

/// Collect every `function.call` in `mod` whose callee is a non-external,
/// non-skipped free function, paired with the resolved callee.
static SmallVector<FreeFunctionCall> collectFreeFunctionCalls(
    ModuleOp mod, SymbolTableCollection &tables, const llvm::DenseSet<Operation *> &skippedCallees
) {
  SmallVector<FreeFunctionCall> calls;
  mod.walk([&](CallOp call) {
    if (FuncDefOp callee = resolveFreeCallee(call, tables, skippedCallees)) {
      calls.push_back({call, callee});
    }
  });
  return calls;
}

/// Collect free functions targeted by calls anywhere in `mod`, including
/// across nested symbol tables.
static llvm::DenseSet<Operation *> collectCalledFreeFunctions(ModuleOp mod) {
  SymbolTableCollection tables;
  llvm::DenseSet<Operation *> calledFunctions;
  mod.walk([&](CallOp call) {
    auto tgtRes = call.getCalleeTarget(tables);
    if (succeeded(tgtRes) && isFreeFunction(tgtRes->get())) {
      calledFunctions.insert(tgtRes->get().getOperation());
    }
  });
  return calledFunctions;
}

/// Collect every non-external free function in `mod` whose symbol has no
/// remaining uses. Call targets are collected separately because
/// `symbolKnownUseEmpty` does not traverse nested symbol tables.
static SmallVector<FuncDefOp> collectUnusedHelpers(ModuleOp mod) {
  llvm::DenseSet<Operation *> calledFunctions = collectCalledFreeFunctions(mod);
  SmallVector<FuncDefOp> unusedFunctions;
  for (FuncDefOp func : mod.getOps<FuncDefOp>()) {
    if (!isFreeFunction(func) || func.isExternal()) {
      continue;
    }
    if (!calledFunctions.contains(func) &&
        symbolKnownUseEmpty(func.getOperation(), mod.getOperation())) {
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
    // Seeded with recursive functions; grows with callees that cannot be
    // inlined.
    llvm::DenseSet<Operation *> skippedCallees =
        collectRecursiveFunctions(getAnalysis<CallGraphAnalysis>().getCallGraph());

    inlineCalls(mod, tables, inliner, skippedCallees);
    removeUnusedFunctions(mod);
  }

  /// Collects the current free-function call sites, then inlines them.
  /// Iterates until no inlinable calls remain: inlined bodies can expose new
  /// calls, but skipped callees are excluded, so every exposed call chain is
  /// finite. Best effort: a callee whose inlining fails is skipped with a
  /// warning, leaving its remaining call sites in place while processing
  /// continues with other callees.
  void inlineCalls(
      ModuleOp mod, SymbolTableCollection &tables, InlinerInterface &inliner,
      llvm::DenseSet<Operation *> &skippedCallees
  ) {
    SmallVector<FreeFunctionCall> callsToInline =
        collectFreeFunctionCalls(mod, tables, skippedCallees);
    while (!callsToInline.empty()) {
      LLVM_DEBUG({
        llvm::dbgs() << "[" DEBUG_TYPE "] round found " << callsToInline.size()
                     << " free-function call site(s) to inline\n";
      });

      for (auto [call, callee] : callsToInline) {
        if (skippedCallees.contains(callee)) {
          continue;
        }
        if (failed(inlineCall(inliner, call, callee, callee.getCallableRegion(), true))) {
          call.emitWarning("failed to inline free function call; skipping this callee");
          skippedCallees.insert(callee);
          continue;
        }
        call.erase();
      }

      callsToInline = collectFreeFunctionCalls(mod, tables, skippedCallees);
    }
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
