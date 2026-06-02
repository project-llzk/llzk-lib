//===-- AnalysisUtil.h - Data-flow analysis utils ---------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Analysis/DataFlowFramework.h>

namespace llzk::dataflow {

/// @brief Loads analyses required to initialize the Executable and PredecessorState
/// analysis states, which are required for the MLIR Dataflow analyses to properly
/// traverse the LLZK IR.
/// @param solver The solver.
void loadRequiredAnalyses(mlir::DataFlowSolver &solver);

/// @brief Loads and runs analyses required to initialize the Executable and PredecessorState
/// analysis states, which are required for the MLIR Dataflow analyses to properly
/// traverse the LLZK IR.
/// This function pre-runs the analyses, which is helpful in cases where early
/// region-op-body traversal is desired.
/// - the bodies of scf.for, scf.while, scf.if are usually not marked as live
///   initially, so the dataflow analysis will traverse all ops in a function
///   before traversing the insides of region ops.
/// - by pre-running the analysis, the region bodies will be explored as encountered
///   if they are marked as "live" by the dead code analysis.
/// @param solver The solver.
/// @param op The operation to pre-run the analyses on.
/// @return Whether the pre-run analysis was successful.
mlir::LogicalResult loadAndRunRequiredAnalyses(mlir::DataFlowSolver &solver, mlir::Operation *op);

/// @brief Returns whether the operation's containing block is live according to
/// the solver's dead-code analysis state. Operations without a containing block
/// or without a known executable state are treated as live.
bool isOperationLive(mlir::DataFlowSolver &solver, mlir::Operation *op);

} // namespace llzk::dataflow
