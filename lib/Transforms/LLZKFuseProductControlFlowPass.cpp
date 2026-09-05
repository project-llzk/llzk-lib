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
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

#include <optional>

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

/// Return whether one operation has the compute role and the other has the constrain role.
static inline bool areOppositeProductSources(Operation *a, Operation *b) {
  std::optional<llvm::StringRef> sourceA = getProductSource(a);
  std::optional<llvm::StringRef> sourceB = getProductSource(b);
  if (!sourceA || !sourceB) {
    return false;
  }
  return (*sourceA == FUNC_NAME_COMPUTE && *sourceB == FUNC_NAME_CONSTRAIN) ||
         (*sourceA == FUNC_NAME_CONSTRAIN && *sourceB == FUNC_NAME_COMPUTE);
}

/// Return whether two loop bounds or steps match under this pass's conservative rules.
///
/// Distinct `poly.read_const` operations are equivalent only when they read the same binding from
/// the same block. Matching trip counts are not sufficient because the fused loop uses one
/// induction variable for both source bodies.
static bool sameLoopControlValue(Value a, Value b) {
  if (a.getType() != b.getType()) {
    return false;
  }
  if (a == b) {
    return true;
  }

  llvm::APInt aConstant;
  if (matchPattern(a, m_ConstantInt(&aConstant))) {
    llvm::APInt bConstant;
    return matchPattern(b, m_ConstantInt(&bConstant)) && aConstant == bConstant;
  }

  auto aConstRead = a.getDefiningOp<polymorphic::ConstReadOp>();
  auto bConstRead = b.getDefiningOp<polymorphic::ConstReadOp>();
  return aConstRead && bConstRead && aConstRead->getBlock() == bConstRead->getBlock() &&
         aConstRead.getConstName() == bConstRead.getConstName();
}

/// Return whether LLZK's `unsignedCmp` attribute selects unsigned bound comparisons for this loop.
static bool usesUnsignedCmp(scf::ForOp loop) {
  if (auto boolAttr = loop->getAttrOfType<BoolAttr>("unsignedCmp")) {
    return boolAttr.getValue();
  }
  return loop->hasAttr("unsignedCmp");
}

/// Return whether two marked loops have the same parent region, opposite product roles, and the
/// same lower bound, upper bound, step, and signed or unsigned comparison.
static inline bool canLoopsBeFused(scf::ForOp a, scf::ForOp b) {
  if (a->getParentRegion() != b->getParentRegion()) {
    return false;
  }

  if (!areOppositeProductSources(a, b)) {
    return false;
  }

  if (usesUnsignedCmp(a) != usesUnsignedCmp(b)) {
    return false;
  }

  return sameLoopControlValue(a.getLowerBound(), b.getLowerBound()) &&
         sameLoopControlValue(a.getUpperBound(), b.getUpperBound()) &&
         sameLoopControlValue(a.getStep(), b.getStep());
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

/// Return whether `write` targets the same member as `read` and is eligible as its compute-side
/// direct write.
static bool isMatchingComputeWrite(MemberWriteOp write, MemberReadOp read) {
  std::optional<llvm::StringRef> source = getProductSource(write);
  return (!source || *source == FUNC_NAME_COMPUTE) && write.getComponent() == read.getComponent() &&
         write.getMemberNameAttr() == read.getMemberNameAttr();
}

/// Return whether a signal member read can be hoisted before `computeIf` while remaining the
/// constraint operand. The referenced member must resolve to an explicit signal definition, match
/// a preceding direct compute-if-result write, have no table offset, have operands available before
/// the compute if, and be used only inside the paired constrain if.
static bool canHoistMemberRead(
    MemberReadOp read, scf::IfOp computeIf, scf::IfOp constrainIf,
    ArrayRef<MemberWriteOp> priorWrites, SymbolTableCollection &symbolTables
) {
  if (!hasProductSource(read, FUNC_NAME_CONSTRAIN) || read.getTableOffset().has_value() ||
      read.getVal().use_empty() || !operandsDominateInsertion(read, computeIf)) {
    return false;
  }

  FailureOr<SymbolLookupResult<MemberDefOp>> memberDef = read.getMemberDefOp(symbolTables);
  if (failed(memberDef) || !memberDef->get().getSignal()) {
    return false;
  }

  bool matchesWrite = llvm::any_of(priorWrites, [read](MemberWriteOp write) {
    return isMatchingComputeWrite(write, read);
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

/// Return whether `root` contains an operation unsafe to cross with compute-side operations.
///
/// An operation may move across compute-side operations only when this pass explicitly admits it
/// or MLIR proves it pure; the walk rejects reads, writes, calls, traps, and unknown effects.
static bool hasUnsafeCrossedConstrainOp(Operation *root) {
  auto result = root->walk([root](Operation *op) {
    if (op == root || isa<scf::YieldOp>(op)) {
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
  SmallVector<MemberWriteOp> candidateWrites;
  for (Operation *op = computeIf->getNextNode(); op != constrainIf; op = op->getNextNode()) {
    if (auto writeOp = dyn_cast<MemberWriteOp>(op)) {
      if (std::optional<llvm::StringRef> source = getProductSource(writeOp);
          source && *source != FUNC_NAME_COMPUTE) {
        return false;
      }
      if (!llvm::is_contained(computeIf.getResults(), writeOp.getVal())) {
        return false;
      }
      candidateWrites.push_back(writeOp);
      continue;
    }

    if (auto readOp = dyn_cast<MemberReadOp>(op)) {
      if (!canHoistMemberRead(readOp, computeIf, constrainIf, candidateWrites, symbolTables)) {
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

  // A hoisted read must identify one unambiguous write. Any additional matching write in the
  // interval leaves the pass unable to preserve which write the original read observed.
  for (MemberReadOp read : readsToHoist) {
    if (llvm::count_if(candidateWrites, [read](MemberWriteOp write) {
      return isMatchingComputeWrite(write, read);
    }) != 1) {
      return false;
    }
  }

  // Hoisting a read crosses the entire compute conditional, so MLIR must prove it pure.
  if (!readsToHoist.empty() && !isPure(computeIf.getOperation())) {
    return false;
  }

  return !hasUnsafeCrossedConstrainOp(constrainIf.getOperation());
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

/// Fuse uniquely matchable marked compute/constrain `scf.if` pairs in `body`.
static void
fuseMatchingIfPairs(Region &body, MLIRContext *context, SymbolTableCollection &symbolTables) {
  llvm::SmallVector<scf::IfOp> computeIfs;
  body.walk<WalkOrder::PreOrder>([&computeIfs](scf::IfOp ifOp) {
    std::optional<llvm::StringRef> productSource = getProductSource(ifOp);
    if (!productSource) {
      return WalkResult::advance();
    }
    if (*productSource == FUNC_NAME_COMPUTE) {
      computeIfs.push_back(ifOp);
    }
    // Defer nested `if` until their enclosing pair has been fused.
    return WalkResult::skip();
  });

  IRRewriter rewriter {context};
  for (scf::IfOp computeIf : computeIfs) {
    // Only member writes and reads may separate a pair. Every other operation is a barrier,
    // so the first following non-member operation is the only possible constrain partner.
    Operation *next = computeIf->getNextNode();
    while (next && isa<MemberWriteOp, MemberReadOp>(next)) {
      next = next->getNextNode();
    }
    auto constrainIf = dyn_cast_if_present<scf::IfOp>(next);
    if (constrainIf && canIfsBeFused(computeIf, constrainIf, symbolTables)) {
      fuseIfPair(computeIf, constrainIf, context, symbolTables, rewriter);
    }
  }
}

/// Return whether `op` is an admissible compute-side sink candidate.
///
/// Direct member writes are the one stateful operation this pass deliberately moves as part of
/// the existing product layout; their cross-loop effects are checked separately. Every other moved
/// operation must be recursively pure so global/RAM accesses, allocations, calls, traps, and
/// unknown effects remain ordered.
static bool isSafeToSinkComputeOp(Operation *op) { return isa<MemberWriteOp>(op) || isPure(op); }

/// Collect operations marked `product_source = "compute"` between sibling loops that must move
/// after `constrainLoop`.
/// Fail if the compute loop is not before its constrain partner in the same block, a crossed
/// non-compute operation is not recursively pure, a compute-sourced operation is not sinkable, the
/// move would cross already-fused constraint work, or a relocated result would lose dominance.
static FailureOr<SmallVector<Operation *>>
canPrepareForFusion(scf::ForOp computeLoop, scf::ForOp constrainLoop) {
  if (computeLoop->getBlock() != constrainLoop->getBlock() ||
      !computeLoop->isBeforeInBlock(constrainLoop)) {
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
    } else if (!isPure(op)) {
      return failure();
    }
  }

  DominanceInfo dominanceInfo(computeLoop->getParentOp());
  auto isMovedWithSink = [&opsToSink](Operation *user) {
    return llvm::any_of(opsToSink, [user](Operation *sink) {
      return sink == user || sink->isAncestor(user);
    });
  };
  auto hasLegalUsers = [&isMovedWithSink, &constrainLoop, &dominanceInfo](Value result) {
    for (Operation *user : result.getUsers()) {
      if (isMovedWithSink(user)) {
        continue;
      }
      // The result definition moves after `constrainLoop`; every surviving user must therefore be
      // dominated by that insertion point and cannot be inside the constrain loop itself.
      if (user == constrainLoop.getOperation() || constrainLoop->isAncestor(user) ||
          !dominanceInfo.dominates(constrainLoop, user)) {
        return false;
      }
    }
    return true;
  };

  for (Value result : computeLoop.getResults()) {
    if (!hasLegalUsers(result)) {
      return failure();
    }
  }
  for (Operation *op : opsToSink) {
    for (Value result : op->getResults()) {
      if (!hasLegalUsers(result)) {
        return failure();
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

/// Fuse uniquely matchable marked compute/constrain `scf.for` pairs in `body` when preparation is
/// legal.
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

  // Matching uses a DenseMap internally. Walk the original lexical compute-loop list when
  // applying pairs, so the result does not depend on container iteration order.
  IRRewriter rewriter {context};
  for (scf::ForOp lexicalComputeLoop : computeLoops) {
    auto candidate = llvm::find_if(fusionCandidates, [lexicalComputeLoop](auto pair) {
      return pair.first == lexicalComputeLoop;
    });
    if (candidate == fusionCandidates.end()) {
      continue;
    }
    auto [computeLoop, constrainLoop] = *candidate;

    // Fusion interleaves the two loop bodies. Require the compute body to be recursively pure and
    // the constrain body to contain only operations admitted by the crossing contract before
    // preparation handles interstitial operations and relocated-result dominance.
    if (!isPure(computeLoop.getOperation()) ||
        hasUnsafeCrossedConstrainOp(constrainLoop.getOperation())) {
      continue;
    }

    Attribute unsignedCmpAttr = computeLoop->getAttr("unsignedCmp");
    if (!unsignedCmpAttr) {
      unsignedCmpAttr = constrainLoop->getAttr("unsignedCmp");
    }
    if (failed(prepareForFusion(computeLoop, constrainLoop, rewriter))) {
      continue;
    }
    auto fusedLoop = fuseIndependentSiblingForLoops(computeLoop, constrainLoop, rewriter);
    if (unsignedCmpAttr) {
      // LLZK records `unsignedCmp` as a discardable `scf.for` attribute. The generic SCF fusion
      // helper does not copy it, so retain the accepted input spelling.
      fusedLoop->setAttr("unsignedCmp", unsignedCmpAttr);
    }
    setProductSource(fusedLoop, "fused");
    // Recurse so nested if/loop pairs become eligible after loop fusion.
    fuseMatchingRegionControlFlow(fusedLoop.getBodyRegion(), context, symbolTables);
  }
}

/// Fuse marked `scf.for` pairs before marked `scf.if` pairs in `body`.
static void fuseMatchingRegionControlFlow(
    Region &body, MLIRContext *context, SymbolTableCollection &symbolTables
) {
  // Loop fusion has priority because preparation may legally move an if marked
  // `product_source = "compute"`, while a prior fused if is an immovable barrier. The later
  // conditional sweep also catches pairs made adjacent by that preparation.
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
