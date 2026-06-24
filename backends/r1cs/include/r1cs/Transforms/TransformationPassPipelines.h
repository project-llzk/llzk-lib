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

void buildFullR1CSLoweringPipeline(mlir::OpPassManager &);

void registerTransformationPassPipelines();

} // namespace r1cs
