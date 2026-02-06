//===-- LLZKFuseProductLoopsPass.h-------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Analysis/LightweightSignalEquivalenceAnalysis.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Region.h>
#include <mlir/Support/LogicalResult.h>

namespace llzk {
/// Identify pairs of `scf.for` loops that can be fused, fuse them, and then
/// recurse to fuse nested loops.
mlir::LogicalResult fuseMatchingLoopPairs(mlir::Region &body, mlir::MLIRContext *context);

/// Identify pairs of `scf.if` ops that can be fused, fuse them, and then recurse to fuse nested ifs
mlir::LogicalResult fuseMatchingIfPairs(
    mlir::Region &body, mlir::MLIRContext *context, LightweightSignalEquivalenceAnalysis &equiv
);

} // namespace llzk
