//===- SparseAnalysis.cpp - LLZK sparse data-flow adapter -----------------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/SparseAnalysis.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>

using namespace mlir;

namespace llzk::dataflow {

AbstractSparseForwardDataFlowAnalysis::AbstractSparseForwardDataFlowAnalysis(DataFlowSolver &s)
    : ::mlir::dataflow::AbstractSparseForwardDataFlowAnalysis(s) {}

LogicalResult AbstractSparseForwardDataFlowAnalysis::initialize(Operation *top) {
  // Match upstream sparse initialization of top-level region entry arguments.
  for (Region &region : top->getRegions()) {
    if (region.empty()) {
      continue;
    }
    for (Value argument : region.front().getArguments()) {
      setToEntryState(getLatticeElement(argument));
    }
  }

  return initializeRecursivelyInProgramOrder(top);
}

LogicalResult AbstractSparseForwardDataFlowAnalysis::visit(ProgramPoint *point) {
  if (point->isBlockStart()) {
    return ::mlir::dataflow::AbstractSparseForwardDataFlowAnalysis::visit(point);
  }

  Operation *op = point->getPrevOp();
  if (op->getNumResults() != 0) {
    return ::mlir::dataflow::AbstractSparseForwardDataFlowAnalysis::visit(point);
  }
  return visitZeroResultOperation(op);
}

LogicalResult
AbstractSparseForwardDataFlowAnalysis::initializeRecursivelyInProgramOrder(Operation *op) {
  if (failed(visitOperationDuringInitialization(op))) {
    return failure();
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      ProgramPoint *blockStart = getProgramPointBefore(&block);
      getOrCreate<::mlir::dataflow::Executable>(blockStart)->blockContentSubscribe(this);
      if (failed(::mlir::dataflow::AbstractSparseForwardDataFlowAnalysis::visit(blockStart))) {
        return failure();
      }
      for (Operation &nestedOp : block) {
        if (failed(initializeRecursivelyInProgramOrder(&nestedOp))) {
          return failure();
        }
      }
    }
  }

  return success();
}

LogicalResult
AbstractSparseForwardDataFlowAnalysis::visitOperationDuringInitialization(Operation *op) {
  if (op->getNumResults() == 0) {
    // Preserve LLZK's zero-result transfer behavior for live effect ops such as
    // constraints, assertions, and writes.
    return visitZeroResultOperation(op);
  }
  return ::mlir::dataflow::AbstractSparseForwardDataFlowAnalysis::visit(getProgramPointAfter(op));
}

bool AbstractSparseForwardDataFlowAnalysis::isOperationLive(Operation *op) {
  if (op->getBlock() == nullptr) {
    return true;
  }
  return getOrCreate<::mlir::dataflow::Executable>(getProgramPointBefore(op->getBlock()))->isLive();
}

llvm::SmallVector<const AbstractSparseLattice *, 4>
AbstractSparseForwardDataFlowAnalysis::collectOperandLatticesAndSubscribe(Operation *op) {
  llvm::SmallVector<const AbstractSparseLattice *, 4> operandLattices;
  operandLattices.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    AbstractSparseLattice *operandLattice = getLatticeElement(operand);
    operandLattice->useDefSubscribe(this);
    operandLattices.push_back(operandLattice);
  }
  return operandLattices;
}

LogicalResult AbstractSparseForwardDataFlowAnalysis::visitZeroResultCallOperation(
    CallOpInterface call, ArrayRef<const AbstractSparseLattice *> operandLattices
) {
  ArrayRef<AbstractSparseLattice *> emptyResultLattices;

  // Preserve the external-call hook. LLZK analyses may use it for no-result
  // call side effects even when there are no result lattices to update.
  auto callable = llvm::dyn_cast_if_present<CallableOpInterface>(call.resolveCallable());
  if (!getSolverConfig().isInterprocedural() || (callable && !callable.getCallableRegion())) {
    visitExternalCallImpl(call, operandLattices, emptyResultLattices);
    return success();
  }

  // Internal zero-result calls have no result lattices, but keep a callgraph
  // dependency so later predecessor updates can revisit the call site.
  Operation *callOp = call.getOperation();
  (void)getOrCreateFor<::mlir::dataflow::PredecessorState>(
      getProgramPointAfter(callOp), getProgramPointAfter(callOp)
  );
  return success();
}

LogicalResult AbstractSparseForwardDataFlowAnalysis::visitZeroResultOperation(Operation *op) {
  if (!isOperationLive(op)) {
    return success();
  }

  // Region-branch operations are fully owned by upstream control-flow
  // propagation. For zero-result region branches, there are no parent result
  // lattices for this compatibility path to update.
  if (llvm::isa<RegionBranchOpInterface>(op)) {
    return success();
  }

  auto operandLattices = collectOperandLatticesAndSubscribe(op);

  if (auto call = llvm::dyn_cast<CallOpInterface>(op)) {
    return visitZeroResultCallOperation(call, operandLattices);
  }

  // Invoke the typed operation transfer function with an empty result range.
  ArrayRef<AbstractSparseLattice *> emptyResultLattices;
  return visitOperationImpl(op, operandLattices, emptyResultLattices);
}

} // namespace llzk::dataflow
