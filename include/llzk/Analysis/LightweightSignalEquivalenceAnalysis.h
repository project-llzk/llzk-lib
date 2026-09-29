//===- LightweightSignalEquivalenceAnalysis.h -------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/Operation.h>

#include <llvm/ADT/EquivalenceClasses.h>

namespace llzk {

class LightweightSignalEquivalenceAnalysis {
  llvm::EquivalenceClasses<mlir::Value> equivalentSignals;

public:
  LightweightSignalEquivalenceAnalysis(mlir::Operation *op);
  bool areSignalsEquivalent(mlir::Value v1, mlir::Value v2);
};

} // namespace llzk
