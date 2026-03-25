//===-- CallGraphAnalyses.h -------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Analysis/CallGraph.h"
#include "llzk/Dialect/Function/IR/Ops.h"

#include <mlir/Pass/AnalysisManager.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SCCIterator.h>
#include <llvm/ADT/STLExtras.h>

namespace mlir {

class Operation;

} // namespace mlir

namespace llzk {

/// An analysis wrapper to compute the \c CallGraph for a \c Module.
///
/// This class implements the concept of an analysis pass used by the \c
/// ModuleAnalysisManager to run an analysis over a module and cache the
/// resulting data.
class CallGraphAnalysis {
  std::unique_ptr<llzk::CallGraph> cg;

public:
  CallGraphAnalysis(mlir::Operation *op);

  llzk::CallGraph &getCallGraph() { return *cg; }
  const llzk::CallGraph &getCallGraph() const { return *cg; }
};

/// Lazily-constructed reachability analysis.
class CallGraphReachabilityAnalysis {

  // Maps function -> callees
  using CalleeMapTy = mlir::DenseMap<function::FuncDefOp, mlir::DenseSet<function::FuncDefOp>>;

  mutable CalleeMapTy reachabilityMap;

  std::reference_wrapper<llzk::CallGraph> callGraph;

public:
  CallGraphReachabilityAnalysis(mlir::Operation *, mlir::AnalysisManager &am);

  static bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) {
    return !pa.isPreserved<CallGraphReachabilityAnalysis>() || !pa.isPreserved<CallGraphAnalysis>();
  }

  /// Returns whether B is reachable from A.
  bool isReachable(function::FuncDefOp &A, function::FuncDefOp &B) const;

  const llzk::CallGraph &getCallGraph() const { return callGraph.get(); }

private:
  inline bool isReachableCached(function::FuncDefOp &A, function::FuncDefOp &B) const {
    auto it = reachabilityMap.find(A);
    return it != reachabilityMap.end() && it->second.find(B) != it->second.end();
  }
};

} // namespace llzk
