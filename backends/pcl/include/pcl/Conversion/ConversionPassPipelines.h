//===-- ConversionPassPipelines.h ------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

namespace mlir {
class OpPassManager;
}

namespace pcl {

/// Normalize loops, flatten templates, and scalarize aggregates before PCL lowering.
/// Preserves struct boundaries and the existing PCL legality requirements.
void buildFullPCLLoweringPipeline(mlir::OpPassManager &pm);

/// Register the llzk-full-pcl-lowering pipeline.
void registerConversionPassPipelines();

} // namespace pcl
