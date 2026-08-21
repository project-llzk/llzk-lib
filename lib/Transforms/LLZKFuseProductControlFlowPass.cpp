//===-- LLZKFuseProductControlFlowPass.cpp ----------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-fuse-product-control-flow` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/AlignmentHelper.h"
#include "llzk/Util/Constants.h"
#include "llzk/Util/ProductSourceHelper.h"

#include <mlir/Dialect/SCF/Utils/Utils.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/SMTAPI.h>

#include <optional>
#include <string>

// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_FUSEPRODUCTCONTROLFLOWPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

namespace {

using namespace mlir;
using namespace llzk;
using namespace llzk::component;

static void fuseMatchingRegionControlFlow(
    Region &body, MLIRContext *context, SymbolTableCollection &symbolTables
);

/// Return whether one operation is compute-sourced and the other is constrain-sourced.
static inline bool areOppositeProductSources(Operation *a, Operation *b) {
  std::optional<llvm::StringRef> sourceA = getProductSource(a);
  std::optional<llvm::StringRef> sourceB = getProductSource(b);
  if (!sourceA || !sourceB) {
    return false;
  }
  return (*sourceA == FUNC_NAME_COMPUTE && *sourceB == FUNC_NAME_CONSTRAIN) ||
         (*sourceA == FUNC_NAME_CONSTRAIN && *sourceB == FUNC_NAME_COMPUTE);
}

// Bitwidth of `index` for instantiating SMT variables
constexpr int INDEX_WIDTH = 64;

static inline bool isConstOrStructParam(Value val) {
  // TODO: doing arithmetic over constants should also be fine?
  return llvm::isa<arith::ConstantIndexOp, polymorphic::ConstReadOp, felt::FeltConstantOp>(
      val.getDefiningOp()
  );
}

static llvm::SMTExprRef mkExpr(Value value, llvm::SMTSolver *solver) {
  if (auto constOp = value.getDefiningOp<arith::ConstantIndexOp>()) {
    return solver->mkBitvector(llvm::APSInt::get(constOp.value()), INDEX_WIDTH);
  } else if (auto polyReadOp = value.getDefiningOp<polymorphic::ConstReadOp>()) {

    return solver->mkSymbol(
        std::string {polyReadOp.getConstName()}.c_str(), solver->getBitvectorSort(INDEX_WIDTH)
    );
  }
  assert(false && "unsupported: checking non-constant trip counts");
  return nullptr; // Unreachable
}

static llvm::SMTExprRef tripCount(scf::ForOp op, llvm::SMTSolver *solver) {
  const auto *one = solver->mkBitvector(llvm::APSInt::get(1), INDEX_WIDTH);
  return solver->mkBVSDiv(
      solver->mkBVAdd(
          one,
          solver->mkBVSub(mkExpr(op.getUpperBound(), solver), mkExpr(op.getLowerBound(), solver))
      ),
      mkExpr(op.getStep(), solver)
  );
}

/// Return whether two marked loops have the same parent region, have opposite product roles, and
/// have provably equal trip counts.
static inline bool canLoopsBeFused(scf::ForOp a, scf::ForOp b) {
  if (a->getParentRegion() != b->getParentRegion()) {
    return false;
  }

  if (!areOppositeProductSources(a, b)) {
    return false;
  }

  // Compare literal trip counts directly; use the solver only when every bound is a constant or
  // struct parameter and equality can therefore be proved symbolically.
  auto tripCountA = constantTripCount(a.getLowerBound(), a.getUpperBound(), a.getStep());
  auto tripCountB = constantTripCount(b.getLowerBound(), b.getUpperBound(), b.getStep());
  if (tripCountA.has_value() && tripCountB.has_value() && *tripCountA == *tripCountB) {
    return true;
  }

  if (!isConstOrStructParam(a.getLowerBound()) || !isConstOrStructParam(a.getUpperBound()) ||
      !isConstOrStructParam(a.getStep()) || !isConstOrStructParam(b.getLowerBound()) ||
      !isConstOrStructParam(b.getUpperBound()) || !isConstOrStructParam(b.getStep())) {
    return false;
  }

  llvm::SMTSolverRef solver = llvm::CreateZ3Solver();
  solver->addConstraint(/* (actually ask if they "can't be different") */ solver->mkNot(
      solver->mkEqual(tripCount(a, solver.get()), tripCount(b, solver.get()))
  ));

  return !*solver->check();
}

/// Return whether every operand of `op` is available at `insertionPoint` using the conservative
/// same-block dominance proof accepted by this pass. Block arguments are available by definition;
/// operation results must be defined earlier in the insertion block.
static bool operandsDominateInsertion(Operation *op, Operation *insertionPoint) {
  for (Value operand : op->getOperands()) {
    if (Operation *def = operand.getDefiningOp()) {
      if (def->getBlock() != insertionPoint->getBlock() || !def->isBeforeInBlock(insertionPoint)) {
        return false;
      }
    }
  }
  return true;
}

/// Return whether a signal member read can be hoisted before `computeIf` without changing the
/// constraint's signal identity. The referenced member must resolve to an explicit signal
/// definition, match a preceding direct compute-if-result write, have no table offset, have
/// operands available before the compute if, and be used only inside the paired constrain if.
static bool canHoistMemberRead(
    MemberReadOp read, scf::IfOp computeIf, scf::IfOp constrainIf,
    ArrayRef<MemberWriteOp> precedingWrites, SymbolTableCollection &symbolTables
) {
  if (!hasProductSource(read, FUNC_NAME_CONSTRAIN) || read.getTableOffset().has_value() ||
      read.getVal().use_empty() || !operandsDominateInsertion(read, computeIf)) {
    return false;
  }

  FailureOr<SymbolLookupResult<MemberDefOp>> memberDef = read.getMemberDefOp(symbolTables);
  if (failed(memberDef) || !memberDef->get().getSignal()) {
    return false;
  }

  bool matchesWrite = llvm::any_of(precedingWrites, [&](MemberWriteOp write) {
    std::optional<llvm::StringRef> source = getProductSource(write);
    return (!source || *source == FUNC_NAME_COMPUTE) &&
           write.getComponent() == read.getComponent() &&
           write.getMemberNameAttr() == read.getMemberNameAttr();
  });
  if (!matchesWrite) {
    return false;
  }

  for (Operation *user : read.getVal().getUsers()) {
    if (!constrainIf->isAncestor(user)) {
      return false;
    }
  }
  return true;
}

/// Return whether `op` may move with the constrain branch across compute-side operations.
static bool isSafeToMoveConstrainOp(Operation *op) {
  // ConstraintOpInterface identifies constraint-producing operations but does not guarantee
  // movement safety. Keep this whitelist explicit until the interface carries that contract.
  if (isa<constrain::EmitEqualityOp, constrain::EmitContainmentOp, NonDetOp>(op)) {
    return true;
  }

  // Nested operations are checked by the walk separately. Admit only the structured control-flow
  // operations this pass recurses into; scf.for must also pass MLIR's speculatability check.
  if (isa<scf::IfOp>(op)) {
    return true;
  }
  if (isa<scf::WhileOp>(op)) {
    return false;
  }
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    return forOp.getSpeculatability() != Speculation::NotSpeculatable;
  }

  return isPure(op);
}

/// Return whether the constrain branch contains an operation unsafe to move across compute-side
/// operations.
///
/// The branch is cloned into the earlier compute branch. An operation is movable only when this
/// pass explicitly admits it or MLIR proves it pure; the walk rejects reads, writes, calls, traps,
/// and unknown effects.
static bool hasUnsafeMovedConstrainOp(scf::IfOp constrainIf) {
  auto result = constrainIf->walk([&](Operation *op) {
    if (op == constrainIf.getOperation() || isa<scf::YieldOp>(op)) {
      return WalkResult::advance();
    }

    if (isSafeToMoveConstrainOp(op)) {
      return WalkResult::advance();
    }
    return WalkResult::interrupt();
  });
  return result.wasInterrupted();
}

/// Return attributes that can be preserved on an operation created by `scf.if` fusion.
///
/// `product_source` identifies the source role of each input operation and cannot be copied to
/// the operation that combines both roles. All other attributes must agree exactly; otherwise the
/// transformation declines to guess which source attribute semantics apply to the fused operation.
static std::optional<DictionaryAttr> getCompatibleFusedAttrs(Operation *a, Operation *b) {
  NamedAttrList attrsA(a->getAttrs());
  attrsA.erase(PRODUCT_SOURCE);
  NamedAttrList attrsB(b->getAttrs());
  attrsB.erase(PRODUCT_SOURCE);
  if (attrsA != attrsB) {
    return std::nullopt;
  }
  return attrsA.getDictionary(a->getContext());
}

/// Collect the compute-if result mappings needed by `constrainIf` and reject unsafe crossings.
///
/// The mapping lets cloned constrain operands use branch-local compute values; return false when
/// intervening definitions or effects would make the move observable. Matching constrain-side
/// signal member reads are returned in `readsToHoist` for movement before `computeIf`.
static bool collectConstrainValueMappings(
    scf::IfOp computeIf, scf::IfOp constrainIf, llvm::DenseMap<Value, unsigned> &valueToResult,
    SmallVector<MemberReadOp> &readsToHoist, SymbolTableCollection &symbolTables
) {
  SmallVector<MemberWriteOp> precedingWrites;
  for (Operation *op = computeIf->getNextNode(); op != constrainIf; op = op->getNextNode()) {
    if (auto writeOp = dyn_cast<MemberWriteOp>(op)) {
      if (std::optional<llvm::StringRef> source = getProductSource(writeOp);
          source && *source != FUNC_NAME_COMPUTE) {
        return false;
      }
      if (!llvm::is_contained(computeIf.getResults(), writeOp.getVal())) {
        return false;
      }
      precedingWrites.push_back(writeOp);
      continue;
    }

    if (auto readOp = dyn_cast<MemberReadOp>(op)) {
      if (!canHoistMemberRead(readOp, computeIf, constrainIf, precedingWrites, symbolTables)) {
        return false;
      }
      readsToHoist.push_back(readOp);
      continue;
    }

    // Only direct member writes and matching signal member reads are currently proven safe to
    // cross. Unknown storage operations, calls, and other intervening definitions are rejected.
    return false;
  }

  for (auto [idx, result] : llvm::enumerate(computeIf.getResults())) {
    valueToResult[result] = idx;
  }

  return !hasUnsafeMovedConstrainOp(constrainIf);
}

/// Return whether two marked sibling `scf.if` ops satisfy the conservative fusion contract.
static bool canIfsBeFused(scf::IfOp a, scf::IfOp b, SymbolTableCollection &symbolTables) {
  if (a->getBlock() != b->getBlock()) {
    return false;
  }
  if (!areOppositeProductSources(a, b)) {
    return false;
  }

  scf::IfOp computeIf = hasProductSource(a, FUNC_NAME_COMPUTE) ? a : b;
  scf::IfOp constrainIf = computeIf == a ? b : a;
  if (!computeIf->isBeforeInBlock(constrainIf)) {
    return false;
  }
  if (!constrainIf->getResults().empty()) {
    return false;
  }
  if (computeIf.getElseRegion().empty() != constrainIf.getElseRegion().empty()) {
    return false;
  }
  if (computeIf.getCondition() != constrainIf.getCondition()) {
    return false;
  }
  if (!getCompatibleFusedAttrs(computeIf, constrainIf)) {
    return false;
  }
  if (!getCompatibleFusedAttrs(
          computeIf.thenBlock()->getTerminator(), constrainIf.thenBlock()->getTerminator()
      )) {
    return false;
  }
  if (!computeIf.getElseRegion().empty() &&
      !getCompatibleFusedAttrs(
          computeIf.elseBlock()->getTerminator(), constrainIf.elseBlock()->getTerminator()
      )) {
    return false;
  }

  llvm::DenseMap<Value, unsigned> valueToResult;
  SmallVector<MemberReadOp> readsToHoist;
  return collectConstrainValueMappings(
      computeIf, constrainIf, valueToResult, readsToHoist, symbolTables
  );
}

/// Remove the destination block's existing `scf.yield` before appending a cloned branch.
static void eraseDefaultTerminator(Block *block) {
  if (!block->empty()) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(block->back())) {
      yieldOp.erase();
    }
  }
}

/// Clone compute operations before constrain operations in `destBlock`, then rebuild its yield.
/// Validated compute results are remapped to the corresponding branch-local yield values.
static void cloneIfBranch(
    Block *computeBlock, Block *constrainBlock, Block *destBlock,
    const llvm::DenseMap<Value, unsigned> &valueToResult, OpBuilder &builder
) {
  eraseDefaultTerminator(destBlock);
  IRMapping mapper;
  builder.setInsertionPointToEnd(destBlock);

  scf::YieldOp computeYield = llvm::cast<scf::YieldOp>(computeBlock->getTerminator());
  scf::YieldOp constrainYield = llvm::cast<scf::YieldOp>(constrainBlock->getTerminator());
  for (Operation &op : computeBlock->without_terminator()) {
    builder.clone(op, mapper);
  }
  for (auto [value, resultIndex] : valueToResult) {
    Value branchValue = computeYield.getResults()[resultIndex];
    mapper.map(value, mapper.lookupOrDefault(branchValue));
  }
  for (Operation &op : constrainBlock->without_terminator()) {
    builder.clone(op, mapper);
  }

  llvm::SmallVector<Value> yieldOperands;
  yieldOperands.reserve(computeYield.getResults().size());
  for (Value operand : computeYield.getResults()) {
    yieldOperands.push_back(mapper.lookupOrDefault(operand));
  }
  auto fusedYield = builder.create<scf::YieldOp>(
      builder.getFusedLoc({computeYield.getLoc(), constrainYield.getLoc()}), yieldOperands
  );
  std::optional<DictionaryAttr> yieldAttrs = getCompatibleFusedAttrs(computeYield, constrainYield);
  assert(yieldAttrs && "fusion candidates must have compatible yield attributes");
  fusedYield->setAttrs(*yieldAttrs);
}

/// Replace a checked compute/constrain `scf.if` pair with one fused `scf.if`.
static void fuseIfPair(
    scf::IfOp a, scf::IfOp b, MLIRContext *context, SymbolTableCollection &symbolTables,
    IRRewriter &rewriter
) {
  scf::IfOp computeIf = hasProductSource(a, FUNC_NAME_COMPUTE) ? a : b;
  scf::IfOp constrainIf = computeIf == a ? b : a;

  llvm::DenseMap<Value, unsigned> valueToResult;
  SmallVector<MemberReadOp> readsToHoist;
  if (!collectConstrainValueMappings(
          computeIf, constrainIf, valueToResult, readsToHoist, symbolTables
      )) {
    assert(false && "fusion candidates must have already been checked");
    return;
  }

  // Preserve the signal member used by the constrain branch. Insert reads in source order before
  // the fused conditional; the matching writes stay after it.
  for (MemberReadOp read : readsToHoist) {
    rewriter.moveOpBefore(read, computeIf);
  }

  rewriter.setInsertionPoint(computeIf);
  std::optional<DictionaryAttr> fusedAttrs = getCompatibleFusedAttrs(computeIf, constrainIf);
  assert(fusedAttrs && "fusion candidates must have compatible if attributes");
  scf::IfOp fusedIf = rewriter.create<scf::IfOp>(
      rewriter.getFusedLoc({computeIf.getLoc(), constrainIf.getLoc()}), computeIf.getResultTypes(),
      computeIf.getCondition(), !computeIf.getElseRegion().empty()
  );
  fusedIf->setAttrs(*fusedAttrs);
  setProductSource(fusedIf, "fused");

  cloneIfBranch(
      computeIf.thenBlock(), constrainIf.thenBlock(), fusedIf.thenBlock(), valueToResult, rewriter
  );
  if (!computeIf.getElseRegion().empty()) {
    cloneIfBranch(
        computeIf.elseBlock(), constrainIf.elseBlock(), fusedIf.elseBlock(), valueToResult, rewriter
    );
  }

  fuseMatchingRegionControlFlow(fusedIf.getThenRegion(), context, symbolTables);
  if (!fusedIf.getElseRegion().empty()) {
    fuseMatchingRegionControlFlow(fusedIf.getElseRegion(), context, symbolTables);
  }

  computeIf->replaceAllUsesWith(fusedIf->getResults());
  rewriter.eraseOp(constrainIf);
  rewriter.eraseOp(computeIf);
}

/// Fuse uniquely matchable marked compute/constrain `scf.if` pairs in `body`; leave unmatched pairs
/// unchanged.
static void
fuseMatchingIfPairs(Region &body, MLIRContext *context, SymbolTableCollection &symbolTables) {
  llvm::SmallVector<scf::IfOp> computeIfs, constrainIfs;
  body.walk<WalkOrder::PreOrder>([&](scf::IfOp ifOp) {
    std::optional<llvm::StringRef> productSource = getProductSource(ifOp);
    if (!productSource) {
      return WalkResult::advance();
    }
    if (*productSource == FUNC_NAME_COMPUTE) {
      computeIfs.push_back(ifOp);
    } else if (*productSource == FUNC_NAME_CONSTRAIN) {
      constrainIfs.push_back(ifOp);
    }
    // Defer nested `if` until their enclosing pair has been fused.
    return WalkResult::skip();
  });

  auto fusionCandidates = *alignmentHelpers::getMatchingPairs<scf::IfOp>(
      computeIfs, constrainIfs,
      [&](scf::IfOp a, scf::IfOp b) { return canIfsBeFused(a, b, symbolTables); },
      /*allowPartial=*/true
  );

  IRRewriter rewriter {context};
  for (auto [computeIf, constrainIf] : fusionCandidates) {
    fuseIfPair(computeIf, constrainIf, context, symbolTables, rewriter);
  }
}

/// Return whether sinking `op` across the constrain loop preserves observable effects.
///
/// Direct member writes are the one stateful operation this pass deliberately moves as part of
/// the existing product layout. Every other moved operation must be recursively pure so
/// global/RAM accesses, allocations, calls, traps, and unknown effects remain ordered.
static bool isSafeToSinkComputeOp(Operation *op) { return isa<MemberWriteOp>(op) || isPure(op); }

/// Collect compute-sourced operations between sibling loops that must move after `constrainLoop`.
/// Fail if the loops do not share a block, an intervening operation has observable effects, the
/// move would cross already-fused constraint work, or sinking a result would place it below a
/// surviving user.
static FailureOr<SmallVector<Operation *>>
canPrepareForFusion(scf::ForOp computeLoop, scf::ForOp constrainLoop) {
  if (computeLoop->getBlock() != constrainLoop->getBlock()) {
    return failure();
  }

  SmallVector<Operation *> opsToSink;
  for (auto *op = computeLoop->getNextNode(); op != constrainLoop; op = op->getNextNode()) {
    if (hasProductSource(op, "fused")) {
      // A fused op contains constrain work and cannot move with compute-only operations.
      return failure();
    }
    if (hasProductSource(op, FUNC_NAME_COMPUTE)) {
      if (!isSafeToSinkComputeOp(op)) {
        return failure();
      }
      opsToSink.push_back(op);
    }
  }

  DominanceInfo dominanceInfo(computeLoop->getParentOp());
  auto isMovedWithSink = [&opsToSink](Operation *user) {
    return llvm::any_of(opsToSink, [user](Operation *sink) {
      return sink == user || sink->isAncestor(user);
    });
  };
  for (Operation *op : opsToSink) {
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (isMovedWithSink(user)) {
          continue;
        }
        // The result definition moves after `constrainLoop`; every surviving user must therefore
        // be dominated by that insertion point and cannot be inside the constrain loop itself.
        if (user == constrainLoop.getOperation() || constrainLoop->isAncestor(user) ||
            !dominanceInfo.dominates(constrainLoop, user)) {
          return failure();
        }
      }
    }
  }
  return opsToSink;
}

/// Move the preflighted compute-only operations after `constrainLoop` in their original order.
/// Because collection finishes before the first move, failure leaves the IR unchanged.
static LogicalResult
prepareForFusion(scf::ForOp computeLoop, scf::ForOp constrainLoop, IRRewriter &rewriter) {
  auto computeOpsToSink = canPrepareForFusion(computeLoop, constrainLoop);
  if (failed(computeOpsToSink)) {
    return failure();
  }

  Operation *insertionPoint = constrainLoop.getOperation();
  for (Operation *op : *computeOpsToSink) {
    rewriter.moveOpAfter(op, insertionPoint);
    insertionPoint = op;
  }

  return success();
}

/// Fuse uniquely matchable marked compute/constrain `scf.for` pairs in `body`; leave unmatched or
/// unpreparable pairs unchanged.
static void
fuseMatchingLoopPairs(Region &body, MLIRContext *context, SymbolTableCollection &symbolTables) {
  // Collect marked loops before matching unique compute/constrain pairs.
  llvm::SmallVector<scf::ForOp> computeLoops, constrainLoops;
  body.walk<WalkOrder::PreOrder>([&computeLoops, &constrainLoops](scf::ForOp forOp) {
    std::optional<llvm::StringRef> productSource = getProductSource(forOp);
    if (!productSource) {
      return WalkResult::skip();
    }
    if (*productSource == FUNC_NAME_COMPUTE) {
      computeLoops.push_back(forOp);
    } else if (*productSource == FUNC_NAME_CONSTRAIN) {
      constrainLoops.push_back(forOp);
    }
    // Defer nested loops until their enclosing pair has been fused.
    return WalkResult::skip();
  });

  // Select only pairs that match uniquely in both directions.
  auto fusionCandidates = *alignmentHelpers::getMatchingPairs<scf::ForOp>(
      computeLoops, constrainLoops, canLoopsBeFused, /*allowPartial=*/true
  );

  // Fuse each unambiguous pair; leave preparation failures unchanged.
  IRRewriter rewriter {context};
  for (auto [computeLoop, constrainLoop] : fusionCandidates) {
    if (failed(prepareForFusion(computeLoop, constrainLoop, rewriter))) {
      continue;
    }
    auto fusedLoop = fuseIndependentSiblingForLoops(computeLoop, constrainLoop, rewriter);
    setProductSource(fusedLoop, "fused");
    // Recurse so nested if/loop pairs become eligible after loop fusion.
    fuseMatchingRegionControlFlow(fusedLoop.getBodyRegion(), context, symbolTables);
  }
}

/// Fuse marked `scf.for` pairs before marked `scf.if` pairs in `body`.
static void fuseMatchingRegionControlFlow(
    Region &body, MLIRContext *context, SymbolTableCollection &symbolTables
) {
  // Loop fusion has priority because preparation may legally sink a compute-sourced if, while a
  // prior fused if is an immovable barrier. The later conditional sweep also catches pairs made
  // adjacent by that preparation.
  fuseMatchingLoopPairs(body, context, symbolTables);
  fuseMatchingIfPairs(body, context, symbolTables);
}

class PassImpl : public llzk::impl::FuseProductControlFlowPassBase<PassImpl> {
  using Base = FuseProductControlFlowPassBase<PassImpl>;
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SymbolTableCollection symbolTables;
    mod.walk([this, &symbolTables](function::FuncDefOp funcDef) {
      if (funcDef.isStructProduct()) {
        fuseMatchingRegionControlFlow(funcDef.getFunctionBody(), &getContext(), symbolTables);
      }
    });
  }
};

} // namespace
