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
#include "llzk/Analysis/SymbolUseGraph.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Transforms/InliningUtils.h>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SCCIterator.h>
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

/// Walk symbol references in `type`, treating StructType names as root-resolved
/// and every other reference as unsupported for cross-symbol-table inlining.
static void walkTypeSymbolRefs(
    Type type, function_ref<void(SymbolRefAttr)> rootRef,
    function_ref<void(SymbolRefAttr)> unsupportedRef
);

/// Walk symbol references in `attr`, applying the type-specific rules above to
/// TypeAttr values.
static void walkAttrSymbolRefs(
    Attribute attr, function_ref<void(SymbolRefAttr)> rootRef,
    function_ref<void(SymbolRefAttr)> unsupportedRef
) {
  attr.walk<WalkOrder::PreOrder>([rootRef, unsupportedRef](TypeAttr typeAttr) {
    walkTypeSymbolRefs(typeAttr.getValue(), rootRef, unsupportedRef);
    return WalkResult::skip();
  }, [unsupportedRef](SymbolRefAttr ref) { unsupportedRef(ref); });
}

static void walkTypeSymbolRefs(
    Type type, function_ref<void(SymbolRefAttr)> rootRef,
    function_ref<void(SymbolRefAttr)> unsupportedRef
) {
  type.walk<WalkOrder::PreOrder>([rootRef, unsupportedRef](component::StructType structType) {
    rootRef(structType.getNameRef());
    if (ArrayAttr params = structType.getParams()) {
      walkAttrSymbolRefs(params, rootRef, unsupportedRef);
    }
    return WalkResult::skip();
  }, [unsupportedRef](SymbolRefAttr ref) { unsupportedRef(ref); });
}

/// Return whether `func` can be inlined into the root module without changing
/// the meaning of references in its body.
static bool isInlinableFreeFunction(FuncDefOp func, ModuleOp root, SymbolTableCollection &tables) {
  Operation *parent = func->getParentOp();
  if (parent == root) {
    // A normal MLIR call uses nearest-symbol lookup after cloning, so a struct
    // method can capture a callee that originally resolved at the root. Keep
    // such helpers in place until general symbol rebasing is implemented.
    bool hasFuncCall = false;
    func.walk([&hasFuncCall](func::CallOp) { hasFuncCall = true; });
    return !hasFuncCall;
  }

  // Struct methods are handled by the struct inliner, not this pass. Also
  // reject functions that are not owned by this module, such as definitions
  // resolved through include.from.
  if (!llvm::isa<ModuleOp, polymorphic::TemplateOp>(parent) ||
      !root->isAncestor(func.getOperation())) {
    return false;
  }

  // TODO: This is a temporary allowance for a common frontend pattern: a
  // helper nested in a namespace-like template or module whose symbol
  // references can all be resolved from the root module. General
  // cross-symbol-table inlining must rebase symbol references in the cloned
  // body so that they continue to resolve to the same definitions.
  bool hasUnsupportedSymbolRef = false;
  func.walk([&tables, &hasUnsupportedSymbolRef, root](Operation *op) {
    auto detectUnsupportedSymbolRef = [&hasUnsupportedSymbolRef](SymbolRefAttr) {
      hasUnsupportedSymbolRef = true;
    };
    auto detectMissingRootSymbolRef = [&tables, &hasUnsupportedSymbolRef, root](SymbolRefAttr ref) {
      if (!tables.lookupSymbolIn(root, ref)) {
        hasUnsupportedSymbolRef = true;
      }
    };
    walkAttrSymbolRefs(
        op->getDiscardableAttrDictionary(), detectMissingRootSymbolRef, detectUnsupportedSymbolRef
    );
    if (Attribute properties = op->getPropertiesAsAttribute()) {
      for (NamedAttribute property : llvm::cast<DictionaryAttr>(properties)) {
        StringRef name = property.getName().getValue();
        // Member names resolve through the component's StructType, not the
        // surrounding symbol table. The StructType itself is scanned below.
        if (llvm::isa<component::MemberRefOpInterface>(op) && name == "member_name") {
          continue;
        }
        // LLZK calls and global references intentionally resolve these
        // properties from the root module. Other properties, including
        // template parameters, retain their ordinary scope-sensitive checks.
        bool usesRootLookup = (llvm::isa<CallOp>(op) && name == "callee") ||
                              (llvm::isa<global::GlobalRefOpInterface>(op) && name == "name_ref");
        if (usesRootLookup) {
          property.getValue().walk(detectMissingRootSymbolRef);
        } else {
          walkAttrSymbolRefs(
              property.getValue(), detectMissingRootSymbolRef, detectUnsupportedSymbolRef
          );
        }
      }
    }
    for (Type type : llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
      walkTypeSymbolRefs(type, detectMissingRootSymbolRef, detectUnsupportedSymbolRef);
    }
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          walkTypeSymbolRefs(arg.getType(), detectMissingRootSymbolRef, detectUnsupportedSymbolRef);
        }
      }
    }
  });
  return !hasUnsupportedSymbolRef;
}

/// Only inline calls inside `function.def` bodies in the root module's symbol
/// namespace. Calls in nested modules or other function-like operations (such
/// as contracts) are outside this pass's scope.
static bool isInlinableCallSite(CallOp call, ModuleOp root) {
  return call->getParentOfType<FuncDefOp>() && call->getParentOfType<ModuleOp>() == root;
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
    CallOp call, ModuleOp root, SymbolTableCollection &tables,
    const llvm::DenseSet<Operation *> &skippedCallees
) {
  auto tgtRes = call.getCalleeTarget(tables);
  if (failed(tgtRes)) {
    return nullptr;
  }
  FuncDefOp callee = tgtRes->get();
  if (!isInlinableFreeFunction(callee, root, tables) || callee.isExternal() ||
      skippedCallees.contains(callee)) {
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
    if (!isInlinableCallSite(call, mod)) {
      return;
    }
    if (FuncDefOp callee = resolveFreeCallee(call, mod, tables, skippedCallees)) {
      calls.push_back({call, callee});
    }
  });
  return calls;
}

/// Collect every non-external free function in `mod` whose symbol has no
/// remaining uses anywhere in the symbol-use graph, including uses in nested
/// symbol tables.
static SmallVector<FuncDefOp> collectUnusedHelpers(ModuleOp mod) {
  SymbolUseGraph useGraph(mod.getOperation());
  SmallVector<FuncDefOp> unusedFunctions;
  for (FuncDefOp func : mod.getOps<FuncDefOp>()) {
    if (func.isExternal()) {
      continue;
    }
    const SymbolUseGraphNode *node = useGraph.lookupNode(func);
    if (!node || !node->hasPredecessor()) {
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
