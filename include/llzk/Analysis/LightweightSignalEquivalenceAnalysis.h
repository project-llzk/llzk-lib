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

/// A lightweight, positive-only equivalence analysis for signal values.
///
/// In addition to recursively comparing ordinary SSA producers, the analysis
/// understands value-copy snapshots of POD records. Two POD reads are equivalent
/// when they select the same complete (possibly nested) record path from the same
/// root POD state. Mutation checks are deliberately local to a single block:
/// writes to the selected record (or an enclosing record) invalidate a proof,
/// while writes to disjoint records do not.
///
/// POD snapshots copied through POD records and struct members are followed
/// recursively. Unknown uses, offset-bearing struct-member reads, and control-flow
/// crossings are handled conservatively and do not establish equivalence. POD reads
/// whose results require an independent mutable copy are also rejected because this
/// analysis does not track mutations of the returned snapshots. Only successful
/// queries are cached.
class LightweightSignalEquivalenceAnalysis {
  llvm::EquivalenceClasses<mlir::Value> equivalentSignals;

public:
  LightweightSignalEquivalenceAnalysis(mlir::Operation *op);
  bool areSignalsEquivalent(mlir::Value v1, mlir::Value v2);
};

} // namespace llzk
