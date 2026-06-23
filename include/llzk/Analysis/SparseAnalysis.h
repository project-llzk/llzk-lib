//===- SparseAnalysis.h - LLZK sparse data-flow adapter -------------------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Portions adapted from mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h.
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides LLZK's sparse forward data-flow analysis compatibility
/// layer. Standard sparse analysis behavior is delegated to upstream MLIR; the
/// adapter only restores LLZK's historical handling for live operations with no
/// results.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>

#include <type_traits>

namespace llzk::dataflow {

using AbstractSparseLattice = mlir::dataflow::AbstractSparseLattice;

//===----------------------------------------------------------------------===//
// AbstractSparseForwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// Compatibility adapter over MLIR sparse forward data-flow analysis.
///
/// Upstream MLIR owns the ordinary sparse forward analysis behavior, including
/// result-producing operation transfer, block-start handling, block argument
/// propagation, call/result propagation, region successor propagation, and
/// interaction with `Executable` / `PredecessorState`.
///
/// LLZK additionally needs transfer functions to run for live operations with
/// no results because several important LLZK effects are modeled that way:
/// constraints, assertions, member writes, and array writes. This adapter keeps
/// the historical LLZK class name and restores only that zero-result operation
/// path.
class AbstractSparseForwardDataFlowAnalysis
    : public mlir::dataflow::AbstractSparseForwardDataFlowAnalysis {
  using Base = mlir::dataflow::AbstractSparseForwardDataFlowAnalysis;

public:
  /// Initialize the analysis while preserving the program-order visitation of
  /// the old LLZK sparse analysis port.
  ///
  /// Result-producing operations and block starts are still visited through
  /// upstream MLIR. LLZK only handles the extra zero-result operation path.
  mlir::LogicalResult initialize(mlir::Operation *top) override;

  /// Delegate block starts and result-producing operations to upstream MLIR.
  /// Only zero-result operations need LLZK-specific handling.
  mlir::LogicalResult visit(mlir::ProgramPoint *point) override;

protected:
  explicit AbstractSparseForwardDataFlowAnalysis(mlir::DataFlowSolver &s);

  /// LLZK: Kept as a compatibility cache for analyses that derived from the old
  /// ported class and used this protected member.
  mlir::SymbolTableCollection tables;

private:
  /// Recursively build the initial dependency graph in IR program order.
  ///
  /// This mirrors upstream sparse initialization's traversal shape, but delegates
  /// all standard block and result-producing operation logic back to upstream
  /// MLIR. Interleaving zero-result operations with surrounding result-producing
  /// operations is important for LLZK analyses that maintain analysis-local
  /// state for effects such as storage writes.
  mlir::LogicalResult initializeRecursivelyInProgramOrder(mlir::Operation *op);

  /// Visit `op` during initialization using the upstream path whenever possible.
  mlir::LogicalResult visitOperationDuringInitialization(mlir::Operation *op);

  /// Return whether `op` should currently be treated as live.
  bool isOperationLive(mlir::Operation *op);

  /// Collect operand lattices and subscribe this analysis through use-def chains
  /// so updates to operand values revisit their zero-result users.
  llvm::SmallVector<const AbstractSparseLattice *, 4>
  collectOperandLatticesAndSubscribe(mlir::Operation *op);

  /// Visit a live zero-result call operation.
  mlir::LogicalResult visitZeroResultCallOperation(
      mlir::CallOpInterface call, mlir::ArrayRef<const AbstractSparseLattice *> operandLattices
  );

  /// Visit a live zero-result operation using the same operand/call dependency
  /// setup as upstream sparse forward analysis, but with an empty result lattice
  /// range.
  mlir::LogicalResult visitZeroResultOperation(mlir::Operation *op);
};

//===----------------------------------------------------------------------===//
// SparseForwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// A sparse forward data-flow analysis for propagating SSA value lattices across
/// the IR by implementing transfer functions for operations.
///
/// `StateT` is expected to be a subclass of `AbstractSparseLattice`.
template <typename StateT>
class SparseForwardDataFlowAnalysis : public AbstractSparseForwardDataFlowAnalysis {
  static_assert(
      std::is_base_of<AbstractSparseLattice, StateT>::value,
      "analysis state class expected to subclass AbstractSparseLattice"
  );

public:
  explicit SparseForwardDataFlowAnalysis(mlir::DataFlowSolver &s)
      : AbstractSparseForwardDataFlowAnalysis(s) {}

  /// Visit an operation with the lattices of its operands. This function is
  /// expected to set the lattices of the operation's results. LLZK also invokes
  /// this hook for live, non-call, non-region-branch operations with no results.
  virtual mlir::LogicalResult visitOperation(
      mlir::Operation *op, mlir::ArrayRef<const StateT *> operands, mlir::ArrayRef<StateT *> results
  ) = 0;

  /// Visit a call operation to an externally defined function given the lattices
  /// of its arguments.
  virtual void visitExternalCall(
      mlir::CallOpInterface /*call*/, mlir::ArrayRef<const StateT *> /*argumentLattices*/,
      mlir::ArrayRef<StateT *> resultLattices
  ) {
    setAllToEntryStates(resultLattices);
  }

  /// Given an operation with possible region control-flow, the lattices of the
  /// operands, and a region successor, compute the lattice values for block
  /// arguments that are not accounted for by the branching control flow (ex. the
  /// bounds of loops). By default, this method marks all such lattice elements
  /// as having reached a pessimistic fixpoint. `firstIndex` is the index of the
  /// first element of `argLattices` that is set by control-flow.
  virtual void visitNonControlFlowArguments(
      mlir::Operation * /*op*/, const mlir::RegionSuccessor &successor,
      mlir::ArrayRef<StateT *> argLattices, unsigned firstIndex
  ) {
    setAllToEntryStates(argLattices.take_front(firstIndex));
    setAllToEntryStates(argLattices.drop_front(firstIndex + successor.getSuccessorInputs().size()));
  }

protected:
  /// Get the lattice element for a value.
  StateT *getLatticeElement(mlir::Value value) override { return getOrCreate<StateT>(value); }

  /// Get the lattice element for a value and create a dependency on the provided
  /// program point.
  const StateT *getLatticeElementFor(mlir::ProgramPoint *point, mlir::Value value) {
    return static_cast<const StateT *>(
        mlir::dataflow::AbstractSparseForwardDataFlowAnalysis::getLatticeElementFor(point, value)
    );
  }

  /// Set the given lattice element(s) at control-flow entry point(s).
  virtual void setToEntryState(StateT *lattice) = 0;
  void setAllToEntryStates(mlir::ArrayRef<StateT *> lattices) {
    mlir::dataflow::AbstractSparseForwardDataFlowAnalysis::setAllToEntryStates(
        {reinterpret_cast<AbstractSparseLattice *const *>(lattices.begin()), lattices.size()}
    );
  }

private:
  /// Type-erased wrappers that convert the abstract lattice operands to derived
  /// lattices and invoke the virtual hooks operating on the derived lattices.
  mlir::LogicalResult visitOperationImpl(
      mlir::Operation *op, mlir::ArrayRef<const AbstractSparseLattice *> operandLattices,
      mlir::ArrayRef<AbstractSparseLattice *> resultLattices
  ) override {
    return visitOperation(
        op,
        {reinterpret_cast<const StateT *const *>(operandLattices.begin()), operandLattices.size()},
        {reinterpret_cast<StateT *const *>(resultLattices.begin()), resultLattices.size()}
    );
  }

  void visitExternalCallImpl(
      mlir::CallOpInterface call, mlir::ArrayRef<const AbstractSparseLattice *> argumentLattices,
      mlir::ArrayRef<AbstractSparseLattice *> resultLattices
  ) override {
    visitExternalCall(
        call,
        {reinterpret_cast<const StateT *const *>(argumentLattices.begin()),
         argumentLattices.size()},
        {reinterpret_cast<StateT *const *>(resultLattices.begin()), resultLattices.size()}
    );
  }

  void visitNonControlFlowArgumentsImpl(
      mlir::Operation *op, const mlir::RegionSuccessor &successor,
      mlir::ArrayRef<AbstractSparseLattice *> argLattices, unsigned firstIndex
  ) override {
    visitNonControlFlowArguments(
        op, successor, {reinterpret_cast<StateT *const *>(argLattices.begin()), argLattices.size()},
        firstIndex
    );
  }

  void setToEntryState(AbstractSparseLattice *lattice) override {
    return setToEntryState(reinterpret_cast<StateT *>(lattice));
  }
};

} // namespace llzk::dataflow
