//===-- TransformationPasses.h ----------------------------------*- C++ -*-===//
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

namespace r1cs {

std::unique_ptr<mlir::Pass> createR1CSLoweringPass();

#if LLZK_WITH_PCL
std::unique_ptr<mlir::Pass> createPCLLoweringPass();
#endif // LLZK_WITH_PCL

void registerTransformationPassPipelines();

#define GEN_PASS_REGISTRATION
#include "r1cs/Transforms/TransformationPasses.h.inc"

}; // namespace r1cs
