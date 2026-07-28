//===-- SharedImpl.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "SharedImpl.h"

#include "llzk/Dialect/Array/IR/Dialect.h"
#include "llzk/Dialect/Bool/IR/Dialect.h"
#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/Global/IR/Dialect.h"
#include "llzk/Dialect/Include/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/Polymorphic/IR/Dialect.h"
#include "llzk/Dialect/RAM/IR/Dialect.h"
#include "llzk/Dialect/String/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Dialect.h"
#include "llzk/Util/SymbolLookup.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "poly-dialect-shared"

mlir::ConversionTarget llzk::polymorphic::detail::newBaseTarget(mlir::MLIRContext *ctx) {
  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<
      llzk::LLZKDialect, llzk::array::ArrayDialect, llzk::boolean::BoolDialect,
      llzk::cast::CastDialect, llzk::component::StructDialect, llzk::constrain::ConstrainDialect,
      llzk::felt::FeltDialect, llzk::function::FunctionDialect, llzk::global::GlobalDialect,
      llzk::include::IncludeDialect, llzk::polymorphic::PolymorphicDialect, llzk::ram::RAMDialect,
      llzk::string::StringDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
  target.addLegalOp<mlir::ModuleOp>();
  return target;
}

llzk::polymorphic::detail::CleanupBase::CleanupBase(
    mlir::ModuleOp root, const llzk::SymbolDefTree &symDefTree,
    const llzk::SymbolUseGraph &symUseGraph
)
    : rootMod(root), defTree(symDefTree), useGraph(symUseGraph) {}

bool llzk::polymorphic::detail::isErasableDefinition(mlir::Operation *op) {
  if (llvm::isa<llzk::component::StructDefOp>(op)) {
    return true;
  }
  if (llzk::function::FuncDefOp fdef = llvm::dyn_cast<llzk::function::FuncDefOp>(op)) {
    return !fdef.isInStruct();
  }
  return false;
}

llzk::polymorphic::detail::FromEraseSet::FromEraseSet(
    mlir::ModuleOp root, const llzk::SymbolDefTree &symDefTree,
    const llzk::SymbolUseGraph &symUseGraph, llvm::DenseSet<mlir::SymbolRefAttr> &&tryToErasePaths
)
    : CleanupBase(root, symDefTree, symUseGraph) {
  // Convert the set of paths targeted for erasure into a set of cleanup-candidate definitions.
  for (mlir::SymbolRefAttr path : tryToErasePaths) {
    LLVM_DEBUG(llvm::dbgs() << "[FromEraseSet] path to erase: " << path << '\n';);
    mlir::Operation *lookupFrom = rootMod.getOperation();
    auto res = lookupSymbolIn(tables, path, Within(), lookupFrom);
    assert(mlir::succeeded(res) && "inputs must be valid symbol references");
    assert(isErasableDefinition(res->get()) && "inputs must be cleanup candidates");
    if (!res->viaInclude()) { // do not remove if it's from another source file
      mlir::SymbolOpInterface op = llvm::cast<mlir::SymbolOpInterface>(res->get());
      LLVM_DEBUG(llvm::dbgs() << "[FromEraseSet]   added op to the erase set: " << op << '\n';);
      tryToErase.insert(op);
    } else {
      LLVM_DEBUG(
          llvm::dbgs() << "[FromEraseSet]   ignored op because it comes from an include: "
                       << res->get() << '\n';
      );
    }
  }
}

mlir::LogicalResult llzk::polymorphic::detail::FromEraseSet::eraseUnusedDefinitions() {
  // Collect the subset of 'tryToErase' that has no remaining uses.
  for (mlir::SymbolOpInterface sym : tryToErase) {
    collectSafeToErase(sym);
  }
  // The `visitedPlusSafetyResult` may contain child FuncDefOp within an erased StructDefOp, so
  // reduce the map to only top-level erase targets before erasing in a separate loop.
  for (auto &it : llvm::make_early_inc_range(visitedPlusSafetyResult)) {
    if (!it.second || !tryToErase.contains(it.first)) {
      visitedPlusSafetyResult.erase(it.first);
    }
  }
  for (auto &[sym, _] : visitedPlusSafetyResult) {
    LLVM_DEBUG(llvm::dbgs() << "[EraseIfUnused] removing: " << sym.getNameAttr() << '\n');
    sym.erase();
  }
  return mlir::success();
}

bool llzk::polymorphic::detail::FromEraseSet::collectSafeToErase(mlir::SymbolOpInterface check) {
  assert(check); // pre-condition

  // If previously visited, return the safety result.
  auto visited = visitedPlusSafetyResult.find(check);
  if (visited != visitedPlusSafetyResult.end()) {
    return visited->second;
  }

  // If it's an erasable definition that is not in `tryToErase` then it cannot be erased.
  if (isErasableDefinition(check.getOperation()) && !tryToErase.contains(check)) {
    visitedPlusSafetyResult[check] = false;
    return false;
  }

  // Otherwise, temporarily mark as safe b/c a node cannot keep itself live (and this prevents
  // the recursion from getting stuck in an infinite loop).
  visitedPlusSafetyResult[check] = true;

  // Check if it's safe according to both the def tree and use graph.
  // Note: Every symbol must have a def node, but symbols with no references do not have use
  // nodes. Those are safe from the use-graph perspective.
  if (collectSafeToErase(defTree.lookupNode(check))) {
    const auto *useNode = useGraph.lookupNode(check);
    if (!useNode || collectSafeToErase(useNode)) {
      return true;
    }
  }

  // Otherwise, revert the safety decision and return it.
  visitedPlusSafetyResult[check] = false;
  return false;
}

bool llzk::polymorphic::detail::FromEraseSet::collectSafeToErase(
    const llzk::SymbolDefTreeNode *check
) {
  assert(check); // pre-condition
  if (const llzk::SymbolDefTreeNode *p = check->getParent()) {
    if (mlir::SymbolOpInterface checkOp = p->getOp()) { // safe if parent is root
      return collectSafeToErase(checkOp);
    }
  }
  return true;
}

bool llzk::polymorphic::detail::FromEraseSet::collectSafeToErase(
    const llzk::SymbolUseGraphNode *check
) {
  assert(check); // pre-condition
  for (const llzk::SymbolUseGraphNode *p : check->predecessorIter()) {
    if (mlir::SymbolOpInterface checkOp = cachedLookup(p)) { // safe if via IncludeOp
      if (!collectSafeToErase(checkOp)) {
        return false;
      }
    }
  }
  return true;
}

mlir::SymbolOpInterface
llzk::polymorphic::detail::FromEraseSet::cachedLookup(const llzk::SymbolUseGraphNode *node) {
  assert(node && "must provide a node"); // pre-condition
  // Check for cached result
  auto fromCache = lookupCache.find(node);
  if (fromCache != lookupCache.end()) {
    return fromCache->second;
  }
  // Otherwise, perform lookup and cache
  auto lookupRes = node->lookupSymbol(tables);
  assert(mlir::succeeded(lookupRes) && "graph contains node with invalid path");
  assert(lookupRes->get() != nullptr && "lookup must return an Operation");
  // If loaded via an IncludeOp it's not in the current AST anyway so ignore.
  // NOTE: The SymbolUseGraph does contain nodes for struct parameters which cannot cast to
  // SymbolOpInterface. However, those will always be leaf nodes in the SymbolUseGraph and
  // therefore will not be traversed by this analysis so directly casting is fine.
  mlir::SymbolOpInterface actualRes =
      lookupRes->viaInclude() ? nullptr : llvm::cast<mlir::SymbolOpInterface>(lookupRes->get());
  // Cache and return
  lookupCache[node] = actualRes;
  assert((!actualRes == lookupRes->viaInclude()) && "not found iff included"); // post-condition
  return actualRes;
}

llzk::array::ArrayType llzk::polymorphic::detail::flattenInstantiatedArrayType(
    llzk::array::ArrayType inputTy, mlir::Type convertedElemTy
) {
  llvm::SmallVector<mlir::Attribute> mergedDims(inputTy.getDimensionSizes());
  while (auto nestedArrTy = llvm::dyn_cast<llzk::array::ArrayType>(convertedElemTy)) {
    llvm::append_range(mergedDims, nestedArrTy.getDimensionSizes());
    convertedElemTy = nestedArrTy.getElementType();
  }
  return llzk::array::ArrayType::get(convertedElemTy, mergedDims);
}

#undef DEBUG_TYPE
