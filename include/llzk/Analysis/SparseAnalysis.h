//===- SparseAnalysis.h - Sparse data-flow analysis -----------------------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h.
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements sparse data-flow analysis using the data-flow analysis
/// framework. The analysis is forward and conditional and uses the results of
/// dead code analysis to prune dead code during the analysis.
///
/// This file has been ported from the MLIR analysis so that it may be
/// tailored to work for LLZK modules,
/// as LLZK modules have different symbol lookup mechanisms that are currently
/// incompatible with the builtin MLIR dataflow analyses.
/// This file is mostly left as original in MLIR, with notes added where
/// changes have been made.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>

#include <llvm/ADT/SmallPtrSet.h>

namespace llzk::dataflow {

using AbstractSparseLattice = mlir::dataflow::AbstractSparseLattice;

//===----------------------------------------------------------------------===//
// AbstractSparseForwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// LLZK: This class has been ported from the MLIR DenseAnalysis utilities to
/// allow for the use of custom LLZK symbol lookup logic. The class has been
/// left as unmodified as possible, with explicit comments added where modifications
/// have been made.
///
/// Base class for sparse forward data-flow analyses. A sparse analysis
/// implements a transfer function on operations from the lattices of the
/// operands to the lattices of the results. This analysis will propagate
/// lattices across control-flow edges and the callgraph using liveness
/// information.
///
/// Visit a program point in sparse forward data-flow analysis will invoke the
/// transfer function of the operation preceding the program point iterator.
/// Visit a program point at the begining of block will visit the block itself.
class AbstractSparseForwardDataFlowAnalysis : public mlir::DataFlowAnalysis {
public:
  /// Initialize the analysis by visiting every owner of an SSA value: all
  /// operations and blocks.
  mlir::LogicalResult initialize(mlir::Operation *top) override;

  /// Visit a program point. If this is at beginning of block and all
  /// control-flow predecessors or callsites are known, then the arguments
  /// lattices are propagated from them. If this is after call operation or an
  /// operation with region control-flow, then its result lattices are set
  /// accordingly.  Otherwise, the operation transfer function is invoked.
  mlir::LogicalResult visit(mlir::ProgramPoint *point) override;

protected:
  explicit AbstractSparseForwardDataFlowAnalysis(mlir::DataFlowSolver &solver);

  /// The operation transfer function. Given the operand lattices, this
  /// function is expected to set the result lattices.
  virtual mlir::LogicalResult visitOperationImpl(
      mlir::Operation *op, mlir::ArrayRef<const AbstractSparseLattice *> operandLattices,
      mlir::ArrayRef<AbstractSparseLattice *> resultLattices
  ) = 0;

  /// The transfer function for calls to external functions.
  virtual void visitExternalCallImpl(
      mlir::CallOpInterface call, mlir::ArrayRef<const AbstractSparseLattice *> argumentLattices,
      mlir::ArrayRef<AbstractSparseLattice *> resultLattices
  ) = 0;

  /// Given an operation with region control-flow, the lattices of the operands,
  /// and a region successor, compute the lattice values for block arguments
  /// that are not accounted for by the branching control flow (ex. the bounds
  /// of loops).
  virtual void visitNonControlFlowArgumentsImpl(
      mlir::Operation *op, const mlir::RegionSuccessor &successor,
      mlir::ArrayRef<AbstractSparseLattice *> argLattices, unsigned firstIndex
  ) = 0;

  /// Get the lattice element of a value.
  virtual AbstractSparseLattice *getLatticeElement(mlir::Value value) = 0;

  /// Get a read-only lattice element for a value and add it as a dependency to
  /// a program point.
  const AbstractSparseLattice *getLatticeElementFor(mlir::ProgramPoint *point, mlir::Value value);

  /// Set the given lattice element(s) at control flow entry point(s).
  virtual void setToEntryState(AbstractSparseLattice *lattice) = 0;
  void setAllToEntryStates(mlir::ArrayRef<AbstractSparseLattice *> lattices);

  /// Join the lattice element and propagate and update if it changed.
  void join(AbstractSparseLattice *lhs, const AbstractSparseLattice &rhs);

  /// LLZK: Added for use of symbol helper caching.
  mlir::SymbolTableCollection tables;

private:
  /// Recursively initialize the analysis on nested operations and blocks.
  mlir::LogicalResult initializeRecursively(mlir::Operation *op);

  /// Visit an operation. If this is a call operation or an operation with
  /// region control-flow, then its result lattices are set accordingly.
  /// Otherwise, the operation transfer function is invoked.
  mlir::LogicalResult visitOperation(mlir::Operation *op);

  /// Visit a block to compute the lattice values of its arguments. If this is
  /// an entry block, then the argument values are determined from the block's
  /// "predecessors" as set by `PredecessorState`. The predecessors can be
  /// region terminators or callable callsites. Otherwise, the values are
  /// determined from block predecessors.
  void visitBlock(mlir::Block *block);

  /// Visit a program point `point` with predecessors within a region branch
  /// operation `branch`, which can either be the entry block of one of the
  /// regions or the parent operation itself, and set either the argument or
  /// parent result lattices.
  void visitRegionSuccessors(
      mlir::ProgramPoint *point, mlir::RegionBranchOpInterface branch,
      mlir::RegionBranchPoint successor, mlir::ArrayRef<AbstractSparseLattice *> lattices
  );
};

//===----------------------------------------------------------------------===//
// SparseForwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// A sparse forward data-flow analysis for propagating SSA value lattices
/// across the IR by implementing transfer functions for operations.
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
  /// expected to set the lattices of the operation's results.
  virtual mlir::LogicalResult visitOperation(
      mlir::Operation *op, mlir::ArrayRef<const StateT *> operands, mlir::ArrayRef<StateT *> results
  ) = 0;

  /// Visit a call operation to an externally defined function given the
  /// lattices of its arguments.
  virtual void visitExternalCall(
      mlir::CallOpInterface /*call*/, mlir::ArrayRef<const StateT *> /*argumentLattices*/,
      mlir::ArrayRef<StateT *> resultLattices
  ) {
    setAllToEntryStates(resultLattices);
  }

  /// Given an operation with possible region control-flow, the lattices of the
  /// operands, and a region successor, compute the lattice values for block
  /// arguments that are not accounted for by the branching control flow (ex.
  /// the bounds of loops). By default, this method marks all such lattice
  /// elements as having reached a pessimistic fixpoint. `firstIndex` is the
  /// index of the first element of `argLattices` that is set by control-flow.
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

  /// Get the lattice element for a value and create a dependency on the
  /// provided program point.
  const StateT *getLatticeElementFor(mlir::ProgramPoint *point, mlir::Value value) {
    return static_cast<const StateT *>(
        AbstractSparseForwardDataFlowAnalysis::getLatticeElementFor(point, value)
    );
  }

  /// Set the given lattice element(s) at control flow entry point(s).
  virtual void setToEntryState(StateT *lattice) = 0;
  void setAllToEntryStates(mlir::ArrayRef<StateT *> lattices) {
    AbstractSparseForwardDataFlowAnalysis::setAllToEntryStates(
        {reinterpret_cast<AbstractSparseLattice *const *>(lattices.begin()), lattices.size()}
    );
  }

private:
  /// Type-erased wrappers that convert the abstract lattice operands to derived
  /// lattices and invoke the virtual hooks operating on the derived lattices.
  llvm::LogicalResult visitOperationImpl(
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
