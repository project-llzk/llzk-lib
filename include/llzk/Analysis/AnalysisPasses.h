//===-- AnalysisPasses.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Analysis/AnalysisPassEnums.h"
#include "llzk/Pass/PassBase.h"

namespace llzk {

std::unique_ptr<mlir::Pass> createCallGraphPrinterPass(llvm::raw_ostream &os);

std::unique_ptr<mlir::Pass> createCallGraphSCCsPrinterPass(llvm::raw_ostream &os);

std::unique_ptr<mlir::Pass> createConstraintDependencyGraphPrinterPass(llvm::raw_ostream &os);

std::unique_ptr<mlir::Pass> createIntervalAnalysisPrinterPass(llvm::raw_ostream &os);

std::unique_ptr<mlir::Pass> createSymbolDefTreePrinterPass();

std::unique_ptr<mlir::Pass> createSymbolUseGraphPrinterPass();

std::unique_ptr<mlir::Pass> createTestAnalysisPass();

#define GEN_PASS_REGISTRATION
#include "llzk/Analysis/AnalysisPasses.h.inc"

} // namespace llzk
