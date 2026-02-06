//===-- LLZKTransformationPasses.h ------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Config/Config.h"
#include "llzk/Pass/PassBase.h"
#include "llzk/Transforms/Parsers.h"

#include <llvm/ADT/APInt.h>
#include <llvm/Support/CommandLine.h>

namespace llzk {

std::unique_ptr<mlir::Pass> createFuseProductIfsPass();

std::unique_ptr<mlir::Pass> createFuseProductLoopsPass();

std::unique_ptr<mlir::Pass> createComputeConstrainToProductPass();

std::unique_ptr<mlir::Pass> createFlatteningPass();

std::unique_ptr<mlir::Pass> createRedundantReadAndWriteEliminationPass();

std::unique_ptr<mlir::Pass> createRedundantOperationEliminationPass();

std::unique_ptr<mlir::Pass> createUnusedDeclarationEliminationPass();

std::unique_ptr<mlir::Pass> createArrayToScalarPass();

std::unique_ptr<mlir::Pass> createPolyLoweringPass();

std::unique_ptr<mlir::Pass> createPolyLoweringPass(unsigned maxDegree);

std::unique_ptr<mlir::Pass> createInlineStructsPass();

std::unique_ptr<mlir::Pass> createR1CSLoweringPass();

#if LLZK_WITH_PCL
std::unique_ptr<mlir::Pass> createPCLLoweringPass();
#endif // LLZK_WITH_PCL

void registerTransformationPassPipelines();

void registerInliningExtensions(mlir::DialectRegistry &registry);

#define GEN_PASS_REGISTRATION
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"

}; // namespace llzk
