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

} // namespace llzk::dataflow
