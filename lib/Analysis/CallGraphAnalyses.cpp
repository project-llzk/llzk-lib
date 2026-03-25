//===-- CallGraphAnalyses.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The contents of this file are adapted from llvm/lib/Analysis/CallGraph.cpp.
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/CallGraphAnalyses.h"
#include "llzk/Dialect/Function/IR/Ops.h"

#include <llvm/ADT/DepthFirstIterator.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>

namespace llzk {

using namespace function;

CallGraphAnalysis::CallGraphAnalysis(mlir::Operation *op) : cg(nullptr) {
  if (auto modOp = llvm::dyn_cast<mlir::ModuleOp>(op)) {
    cg = std::make_unique<llzk::CallGraph>(modOp);
  } else {
    const char *error_message = "CallGraphAnalysis expects provided op to be a ModuleOp!";
    op->emitError(error_message).report();
    llvm::report_fatal_error(error_message);
  }
}

/**
 * NOTE: the need for the mlir::Operation argument is a requirement of the mlir::getAnalysis
 * method, which requires template types to define a constructor that either takes
 * only an mlir::Operation* (as in the CallGraphAnalysis above) or the signature below.
 * See:
 *  https://github.com/llvm/llvm-project/blob/415cfaf339dc4383acd44248584bcc6376213c8d/mlir/include/mlir/Pass/AnalysisManager.h#L220-L234
 *  https://mlir.llvm.org/docs/PassManagement/#querying-analyses
 */
CallGraphReachabilityAnalysis::CallGraphReachabilityAnalysis(
    mlir::Operation *, mlir::AnalysisManager &am
)
    // getting the CallGraphAnalysis will enforce the need for a module op
    : callGraph(am.getAnalysis<CallGraphAnalysis>().getCallGraph()) {}

bool CallGraphReachabilityAnalysis::isReachable(FuncDefOp &A, FuncDefOp &B) const {
  if (isReachableCached(A, B)) {
    return true;
  }

  auto *startNode = callGraph.get().lookupNode(A.getCallableRegion());
  if (!startNode) {
    const char *msg = "CallGraph contains no starting node!";
    A.emitError(msg).report();
    llvm::report_fatal_error(msg);
  }
  /**
   * NOTE: This is a potential cause of performance issues, as some circuits
   * may perform poorly for DFS. However, we don't have enough examples at this
   * time to demonstrate such an issue, so we will stick with DFS for simplicity
   * for now. If performance issues arise, it may be beneficial to switch to
   * inverse DFS or a different algorithm entirely.
   */
  auto dfsIt = llvm::df_begin<const CallGraphNode *>(startNode);
  auto dfsEnd = llvm::df_end<const CallGraphNode *>(startNode);
  for (; dfsIt != dfsEnd; ++dfsIt) {
    const CallGraphNode *currNode = *dfsIt;
    if (currNode->isExternal()) {
      continue;
    }
    FuncDefOp currFn = currNode->getCalledFunction();

    // Update the cache according to the path before checking if B is reachable.
    for (unsigned i = 0; i < dfsIt.getPathLength(); i++) {
      FuncDefOp ancestorFn = dfsIt.getPath(i)->getCalledFunction();
      reachabilityMap[ancestorFn].insert(currFn);
    }

    if (isReachableCached(currFn, B)) {
      return true;
    }
  }
  return false;
}

} // namespace llzk
