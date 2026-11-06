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

#include <string>

namespace r1cs {

/// Additional files emitted alongside the R1CS IR produced by lowering.
struct R1CSOutputConfig {
  /// LLZK with auxiliary signal fields and their assignments in compute.
  std::string outputLLZK;
  /// CSV mapping access paths and LLZK layout IDs to final R1CS wire IDs.
  std::string outputLayout;
};

/// Run full polynomial lowering followed by the existing R1CS conversion pass.
void buildFullR1CSLoweringPipeline(mlir::OpPassManager &, const R1CSOutputConfig &outputs = {});

/// Monomorphize and flatten main constraints, lower degree, and convert layout leaves to R1CS
/// wires.
void buildConstraintFlatteningR1CSLoweringPipeline(
    mlir::OpPassManager &, const R1CSOutputConfig &outputs = {}
);

void registerTransformationPassPipelines();

} // namespace r1cs
