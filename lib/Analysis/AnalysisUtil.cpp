//===-- AnalysisUtil.cpp - Data-flow analysis utils -------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/AnalysisUtil.h"

#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>

using namespace mlir;

using Executable = mlir::dataflow::Executable;

namespace llzk::dataflow {

void loadRequiredAnalyses(DataFlowSolver &solver) {
  solver.load<mlir::dataflow::SparseConstantPropagation>();
  solver.load<mlir::dataflow::DeadCodeAnalysis>();
}

LogicalResult loadAndRunRequiredAnalyses(DataFlowSolver &solver, Operation *op) {
  loadRequiredAnalyses(solver);
  return solver.initializeAndRun(op);
}

} // namespace llzk::dataflow
