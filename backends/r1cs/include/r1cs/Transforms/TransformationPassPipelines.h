//===-- TransformationPassPipelines.h ---------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassOptions.h>

namespace r1cs {

/// Select the input contract explicitly when assembling an R1CS pipeline.
enum class R1CSLoweringMode { Legacy, Direct };

/// Build a flat pipeline for either flattened legacy IR or evaluated storage IR.
void buildFullR1CSLoweringPipeline(mlir::OpPassManager &, R1CSLoweringMode mode);

void registerTransformationPassPipelines();

} // namespace r1cs
