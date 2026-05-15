//===-- WitgenLowering.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Pass/Pass.h>

#include <memory>

namespace llzk::witgen {

/// Create the pass that lowers supported LLZK compute IR into core MLIR
/// dialects suitable for LLVM lowering.
std::unique_ptr<mlir::Pass> createLowerComputeToCorePass();

/// Create the pass that synthesizes the stable llzk-witgen JIT entry wrapper.
std::unique_ptr<mlir::Pass> createCreateWitgenEntryPass(bool emitFullWitness);

/// Add the preprocessing pipeline required before witgen backend execution.
void addWitgenPreparePipeline(mlir::OpPassManager &pm);

} // namespace llzk::witgen
