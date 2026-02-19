//===- SparseAnalysis.cpp - Sparse data-flow analysis ---------------------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from mlir/lib/Analysis/DataFlow/SparseAnalysis.cpp.
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/SparseAnalysis.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Util/ErrorHelper.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>

#include <cassert>
#include <optional>

using namespace mlir;
using namespace mlir::dataflow;
using namespace llzk::function;

namespace llzk::dataflow {

//===----------------------------------------------------------------------===//
// AbstractSparseForwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

AbstractSparseForwardDataFlowAnalysis::AbstractSparseForwardDataFlowAnalysis(DataFlowSolver &s)
    : DataFlowAnalysis(s) {
  registerAnchorKind<CFGEdge>();
}

LogicalResult AbstractSparseForwardDataFlowAnalysis::initialize(Operation *top) {
  // Mark the entry block arguments as having reached their pessimistic
  // fixpoints.
  for (Region &region : top->getRegions()) {
    if (region.empty()) {
      continue;
    }
    for (Value argument : region.front().getArguments()) {
      setToEntryState(getLatticeElement(argument));
    }
  }

  return initializeRecursively(top);
}

LogicalResult AbstractSparseForwardDataFlowAnalysis::initializeRecursively(Operation *op) {
  // Initialize the analysis by visiting every owner of an SSA value (all
  // operations and blocks).
  if (failed(visitOperation(op))) {
    return failure();
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      getOrCreate<Executable>(getProgramPointBefore(&block))->blockContentSubscribe(this);
      visitBlock(&block);
      // LLZK: Renamed "op" -> "containedOp" to avoid shadowing.
      for (Operation &containedOp : block) {
        if (failed(initializeRecursively(&containedOp))) {
          return failure();
        }
      }
    }
  }

  return success();
}

LogicalResult AbstractSparseForwardDataFlowAnalysis::visit(ProgramPoint *point) {
  if (!point->isBlockStart()) {
    return visitOperation(point->getPrevOp());
  }
  visitBlock(point->getBlock());
  return success();
}

LogicalResult AbstractSparseForwardDataFlowAnalysis::visitOperation(Operation *op) {
  /// LLZK: Visit operations with no results, as they may still affect values
  /// (e.g., constraints and assertions).
  /// The MLIR version doesn't visit result-less operations.

  // If the containing block is not executable, bail out.
  if (op->getBlock() != nullptr &&
      !getOrCreate<Executable>(getProgramPointBefore(op->getBlock()))->isLive()) {
    return success();
  }

  // Get the result lattices.
  SmallVector<AbstractSparseLattice *> resultLattices;
  resultLattices.reserve(op->getNumResults());
  for (Value result : op->getResults()) {
    AbstractSparseLattice *resultLattice = getLatticeElement(result);
    resultLattices.push_back(resultLattice);
  }

  // The results of a region branch operation are determined by control-flow.
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    visitRegionSuccessors(
        getProgramPointAfter(branch), branch,
        /*successor=*/RegionBranchPoint::parent(), resultLattices
    );
    return success();
  }

  // Grab the lattice elements of the operands.
  SmallVector<const AbstractSparseLattice *> operandLattices;
  operandLattices.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    AbstractSparseLattice *operandLattice = getLatticeElement(operand);
    operandLattice->useDefSubscribe(this);
    operandLattices.push_back(operandLattice);
  }

  if (auto call = dyn_cast<CallOpInterface>(op)) {
    // If the call operation is to an external function, attempt to infer the
    // results from the call arguments.
    auto callable = dyn_cast_if_present<CallableOpInterface>(call.resolveCallable());
    if (!getSolverConfig().isInterprocedural() || (callable && !callable.getCallableRegion())) {
      visitExternalCallImpl(call, operandLattices, resultLattices);
      return success();
    }

    // Otherwise, the results of a call operation are determined by the
    // callgraph.
    const auto *predecessors =
        getOrCreateFor<PredecessorState>(getProgramPointAfter(op), getProgramPointAfter(call));
    // If not all return sites are known, then conservatively assume we can't
    // reason about the data-flow.
    if (!predecessors->allPredecessorsKnown()) {
      setAllToEntryStates(resultLattices);
      return success();
    }
    for (Operation *predecessor : predecessors->getKnownPredecessors()) {
      for (auto &&[operand, resLattice] : llvm::zip(predecessor->getOperands(), resultLattices)) {
        join(resLattice, *getLatticeElementFor(getProgramPointAfter(op), operand));
      }
    }
    return success();
  }

  // Invoke the operation transfer function.
  return visitOperationImpl(op, operandLattices, resultLattices);
}

void AbstractSparseForwardDataFlowAnalysis::visitBlock(Block *block) {
  // Exit early on blocks with no arguments.
  if (block->getNumArguments() == 0) {
    return;
  }

  // If the block is not executable, bail out.
  if (!getOrCreate<Executable>(getProgramPointBefore(block))->isLive()) {
    return;
  }

  // Get the argument lattices.
  SmallVector<AbstractSparseLattice *> argLattices;
  argLattices.reserve(block->getNumArguments());
  for (BlockArgument argument : block->getArguments()) {
    AbstractSparseLattice *argLattice = getLatticeElement(argument);
    argLattices.push_back(argLattice);
  }

  // The argument lattices of entry blocks are set by region control-flow or the
  // callgraph.
  if (block->isEntryBlock()) {
    // Check if this block is the entry block of a callable region.
    auto callable = dyn_cast<CallableOpInterface>(block->getParentOp());
    if (callable && callable.getCallableRegion() == block->getParent()) {
      const auto *callsites = getOrCreateFor<PredecessorState>(
          getProgramPointBefore(block), getProgramPointAfter(callable)
      );
      // If not all callsites are known, conservatively mark all lattices as
      // having reached their pessimistic fixpoints.
      if (!callsites->allPredecessorsKnown() || !getSolverConfig().isInterprocedural()) {
        return setAllToEntryStates(argLattices);
      }
      for (Operation *callsite : callsites->getKnownPredecessors()) {
        auto call = cast<CallOpInterface>(callsite);
        for (auto it : llvm::zip(call.getArgOperands(), argLattices)) {
          join(
              std::get<1>(it), *getLatticeElementFor(getProgramPointBefore(block), std::get<0>(it))
          );
        }
      }
      return;
    }

    // Check if the lattices can be determined from region control flow.
    if (auto branch = dyn_cast<RegionBranchOpInterface>(block->getParentOp())) {
      return visitRegionSuccessors(
          getProgramPointBefore(block), branch, block->getParent(), argLattices
      );
    }

    // Otherwise, we can't reason about the data-flow.
    return visitNonControlFlowArgumentsImpl(
        block->getParentOp(), RegionSuccessor(block->getParent()), argLattices, /*firstIndex=*/0
    );
  }

  // Iterate over the predecessors of the non-entry block.
  for (Block::pred_iterator it = block->pred_begin(), e = block->pred_end(); it != e; ++it) {
    Block *predecessor = *it;

    // If the edge from the predecessor block to the current block is not live,
    // bail out.
    auto *edgeExecutable = getOrCreate<Executable>(getLatticeAnchor<CFGEdge>(predecessor, block));
    edgeExecutable->blockContentSubscribe(this);
    if (!edgeExecutable->isLive()) {
      continue;
    }

    // Check if we can reason about the data-flow from the predecessor.
    if (auto branch = dyn_cast<BranchOpInterface>(predecessor->getTerminator())) {
      SuccessorOperands operands = branch.getSuccessorOperands(it.getSuccessorIndex());
      for (auto [idx, lattice] : llvm::enumerate(argLattices)) {
        if (Value operand = operands[idx]) {
          join(lattice, *getLatticeElementFor(getProgramPointBefore(block), operand));
        } else {
          // Conservatively consider internally produced arguments as entry
          // points.
          setAllToEntryStates(lattice);
        }
      }
    } else {
      return setAllToEntryStates(argLattices);
    }
  }
}

void AbstractSparseForwardDataFlowAnalysis::visitRegionSuccessors(
    ProgramPoint *point, RegionBranchOpInterface branch, RegionBranchPoint successor,
    ArrayRef<AbstractSparseLattice *> lattices
) {
  const auto *predecessors = getOrCreateFor<PredecessorState>(point, point);
  assert(predecessors->allPredecessorsKnown() && "unexpected unresolved region successors");

  for (Operation *op : predecessors->getKnownPredecessors()) {
    // Get the incoming successor operands.
    std::optional<OperandRange> operands;

    // Check if the predecessor is the parent op.
    if (op == branch) {
      operands = branch.getEntrySuccessorOperands(successor);
      // Otherwise, try to deduce the operands from a region return-like op.
    } else if (auto regionTerminator = dyn_cast<RegionBranchTerminatorOpInterface>(op)) {
      operands = regionTerminator.getSuccessorOperands(successor);
    }

    if (!operands) {
      // We can't reason about the data-flow.
      return setAllToEntryStates(lattices);
    }

    ValueRange inputs = predecessors->getSuccessorInputs(op);
    assert(
        inputs.size() == operands->size() &&
        "expected the same number of successor inputs as operands"
    );

    unsigned firstIndex = 0;
    if (inputs.size() != lattices.size()) {
      if (!point->isBlockStart()) {
        if (!inputs.empty()) {
          firstIndex = cast<OpResult>(inputs.front()).getResultNumber();
        }
        visitNonControlFlowArgumentsImpl(
            branch, RegionSuccessor(branch->getResults().slice(firstIndex, inputs.size())),
            lattices, firstIndex
        );
      } else {
        if (!inputs.empty()) {
          firstIndex = cast<BlockArgument>(inputs.front()).getArgNumber();
        }
        Region *region = point->getBlock()->getParent();
        visitNonControlFlowArgumentsImpl(
            branch,
            RegionSuccessor(region, region->getArguments().slice(firstIndex, inputs.size())),
            lattices, firstIndex
        );
      }
    }

    for (auto it : llvm::zip(*operands, lattices.drop_front(firstIndex))) {
      join(std::get<1>(it), *getLatticeElementFor(point, std::get<0>(it)));
    }
  }
}

const AbstractSparseLattice *
AbstractSparseForwardDataFlowAnalysis::getLatticeElementFor(ProgramPoint *point, Value value) {
  AbstractSparseLattice *state = getLatticeElement(value);
  addDependency(state, point);
  return state;
}

void AbstractSparseForwardDataFlowAnalysis::setAllToEntryStates(
    ArrayRef<AbstractSparseLattice *> lattices
) {
  for (AbstractSparseLattice *lattice : lattices) {
    setToEntryState(lattice);
  }
}

void AbstractSparseForwardDataFlowAnalysis::join(
    AbstractSparseLattice *lhs, const AbstractSparseLattice &rhs
) {
  propagateIfChanged(lhs, lhs->join(rhs));
}

} // namespace llzk::dataflow
