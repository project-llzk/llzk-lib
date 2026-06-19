//===-- InlineStructsPass.h -------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Struct/IR/Ops.h"

#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LogicalResult.h>

/// Maps caller struct to callees that should be inlined. The outer SmallVector preserves the
/// ordering from the bottom-up traversal that builds the InliningPlan so performing inlining
/// in the order given will not lose any or require doing any more than once.
/// Note: Applying in the opposite direction would reduce making repeated clones of the ops within
/// the inlined struct functions (as they are inlined further and further up the tree) but that
/// would require updating some mapping in the plan along the way to ensure it's done properly.
using InliningPlan = mlir::SmallVector<
    std::pair<llzk::component::StructDefOp, mlir::SmallVector<llzk::component::StructDefOp>>>;

mlir::LogicalResult performInlining(mlir::SymbolTableCollection &tables, InliningPlan &plan);
