//===-- LLZKFuseProductLoopsPass.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-fuse-product-loops` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/RAM/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/AlignmentHelper.h"
#include "llzk/Util/Constants.h"

#include <mlir/Dialect/SCF/Utils/Utils.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/SMTAPI.h>

#include <memory>
#include <optional>

// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_FUSEPRODUCTLOOPSPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

namespace {

using namespace mlir;
using namespace llzk;
using namespace llzk::component;

static LogicalResult fuseMatchingRegionControlFlow(Region &body, MLIRContext *context);
static std::optional<llvm::StringRef> getProductSource(Operation *op);
static inline bool areOppositeProductSources(Operation *a, Operation *b);

// Bitwidth of `index` for instantiating SMT variables
constexpr int INDEX_WIDTH = 64;

static inline bool isConstOrStructParam(Value val) {
  // TODO: doing arithmetic over constants should also be fine?
  return val.getDefiningOp<arith::ConstantIndexOp>() ||
         val.getDefiningOp<polymorphic::ConstReadOp>();
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

static inline bool canLoopsBeFused(scf::ForOp a, scf::ForOp b) {
  // A priori, two loops can be fused if:
  // 1. They live in the same parent region,
  // 2. One comes from witgen and the other comes from constraint gen, and
  // 3. They have the same trip count

  // Check 1.
  if (a->getParentRegion() != b->getParentRegion()) {
    return false;
  }

  // Check 2.
  if (!areOppositeProductSources(a, b)) {
    return false;
  }

  // Check 3.
  // Easy case: both have a constant trip-count. If the trip counts are not "constant up to a struct
  // param", we definitely can't tell if they're equal. If the trip counts are only "constant up to
  // a struct param" but not actually constant, we can ask a solver if the equations are guaranteed
  // to be the same
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

static std::optional<llvm::StringRef> getProductSource(Operation *op) {
  if (StringAttr source = op->getAttrOfType<StringAttr>(PRODUCT_SOURCE)) {
    return source.getValue();
  }
  return std::nullopt;
}

static inline bool hasProductSource(Operation *op, llvm::StringRef source) {
  std::optional<llvm::StringRef> productSource = getProductSource(op);
  return productSource && *productSource == source;
}

static inline bool areOppositeProductSources(Operation *a, Operation *b) {
  std::optional<llvm::StringRef> sourceA = getProductSource(a);
  std::optional<llvm::StringRef> sourceB = getProductSource(b);
  if (!sourceA || !sourceB) {
    return false;
  }
  return (*sourceA == FUNC_NAME_COMPUTE && *sourceB == FUNC_NAME_CONSTRAIN) ||
         (*sourceA == FUNC_NAME_CONSTRAIN && *sourceB == FUNC_NAME_COMPUTE);
}

static bool isBetweenInBlock(Operation *op, Operation *before, Operation *after) {
  return op->getBlock() == before->getBlock() && op->getBlock() == after->getBlock() &&
         before->isBeforeInBlock(op) && op->isBeforeInBlock(after);
}

static std::optional<unsigned> getIfResultIndex(scf::IfOp ifOp, Value value) {
  for (auto [idx, result] : llvm::enumerate(ifOp.getResults())) {
    if (result == value) {
      return idx;
    }
  }
  return std::nullopt;
}

static bool hasWriteLikeMemoryEffect(Operation *op) {
  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectOp) {
    return false;
  }
  return effectOp.hasEffect<MemoryEffects::Write>() ||
         effectOp.hasEffect<MemoryEffects::Allocate>() || effectOp.hasEffect<MemoryEffects::Free>();
}

static bool hasUnsafeMovedConstrainOp(scf::IfOp constrainIf) {
  auto result = constrainIf->walk([&](Operation *op) {
    if (op == constrainIf.getOperation() || isa<scf::YieldOp>(op)) {
      return WalkResult::advance();
    }

    if (isa<MemberReadOp, MemberWriteOp, llzk::global::GlobalWriteOp, llzk::array::WriteArrayOp,
            llzk::array::InsertArrayOp, llzk::pod::WritePodOp, llzk::ram::StoreOp>(op)) {
      return WalkResult::interrupt();
    }

    if (op->hasTrait<llzk::function::WitnessGen>() || hasWriteLikeMemoryEffect(op)) {
      return WalkResult::interrupt();
    }

    if (isa<CallOpInterface>(op)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

static bool collectConstrainValueMappings(
    scf::IfOp computeIf, scf::IfOp constrainIf, llvm::DenseMap<Value, unsigned> &valueToResult
) {
  // `scf.if` fusion moves the constrain branch before intervening member writes of compute-if
  // results. The moved branch must not read members, write storage, call functions, or have
  // write-like effects.
  for (Operation *op = computeIf->getNextNode(); op != constrainIf; op = op->getNextNode()) {
    if (auto writeOp = dyn_cast<MemberWriteOp>(op)) {
      std::optional<unsigned> resultIndex = getIfResultIndex(computeIf, writeOp.getVal());
      if (!resultIndex) {
        return false;
      }
      continue;
    }

    if (isa<MemberReadOp>(op)) {
      // Replacing a member read with the branch-local computed value can remove the member signal
      // from emitted constraints. Keep this fusion conservative until reads can be preserved.
      return false;
    }

    return false;
  }

  for (auto [idx, result] : llvm::enumerate(computeIf.getResults())) {
    valueToResult[result] = idx;
  }

  if (hasUnsafeMovedConstrainOp(constrainIf)) {
    return false;
  }

  auto result = constrainIf->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def || constrainIf->isAncestor(def)) {
        continue;
      }
      if (valueToResult.contains(operand)) {
        continue;
      }
      if (!isBetweenInBlock(def, computeIf, constrainIf)) {
        continue;
      }
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !result.wasInterrupted();
}

static bool canIfsBeFused(scf::IfOp a, scf::IfOp b) {
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

  llvm::DenseMap<Value, unsigned> valueToResult;
  return collectConstrainValueMappings(computeIf, constrainIf, valueToResult);
}

static void eraseDefaultTerminator(Block *block) {
  if (!block->empty()) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(block->back())) {
      yieldOp.erase();
    }
  }
}

static void cloneIfBranch(
    scf::IfOp computeIf, Block *computeBlock, Block *constrainBlock, Block *destBlock,
    const llvm::DenseMap<Value, unsigned> &valueToResult, OpBuilder &builder
) {
  eraseDefaultTerminator(destBlock);
  IRMapping mapper;
  builder.setInsertionPointToEnd(destBlock);

  scf::YieldOp computeYield = cast<scf::YieldOp>(computeBlock->getTerminator());
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
  builder.create<scf::YieldOp>(computeIf.getLoc(), yieldOperands);
}

static LogicalResult
fuseIfPair(scf::IfOp a, scf::IfOp b, MLIRContext *context, IRRewriter &rewriter) {
  scf::IfOp computeIf = hasProductSource(a, FUNC_NAME_COMPUTE) ? a : b;
  scf::IfOp constrainIf = computeIf == a ? b : a;

  llvm::DenseMap<Value, unsigned> valueToResult;
  [[maybe_unused]] bool canMap =
      collectConstrainValueMappings(computeIf, constrainIf, valueToResult);
  assert(canMap && "fusion candidates must have already been checked");

  rewriter.setInsertionPoint(computeIf);
  scf::IfOp fusedIf = rewriter.create<scf::IfOp>(
      computeIf.getLoc(), computeIf.getResultTypes(), computeIf.getCondition(),
      !computeIf.getElseRegion().empty()
  );
  fusedIf->setAttr(PRODUCT_SOURCE, rewriter.getStringAttr("fused"));

  cloneIfBranch(
      computeIf, computeIf.thenBlock(), constrainIf.thenBlock(), fusedIf.thenBlock(), valueToResult,
      rewriter
  );
  if (!computeIf.getElseRegion().empty()) {
    cloneIfBranch(
        computeIf, computeIf.elseBlock(), constrainIf.elseBlock(), fusedIf.elseBlock(),
        valueToResult, rewriter
    );
  }

  if (failed(fuseMatchingRegionControlFlow(fusedIf.getThenRegion(), context))) {
    return failure();
  }
  if (!fusedIf.getElseRegion().empty() &&
      failed(fuseMatchingRegionControlFlow(fusedIf.getElseRegion(), context))) {
    return failure();
  }

  computeIf->replaceAllUsesWith(fusedIf->getResults());
  rewriter.eraseOp(constrainIf);
  rewriter.eraseOp(computeIf);
  return success();
}

static LogicalResult fuseMatchingIfPairs(Region &body, MLIRContext *context) {
  llvm::SmallVector<scf::IfOp> witnessIfs, constraintIfs;
  body.walk<WalkOrder::PreOrder>([&](scf::IfOp ifOp) {
    std::optional<llvm::StringRef> productSource = getProductSource(ifOp);
    if (!productSource) {
      return WalkResult::advance();
    }
    if (*productSource == FUNC_NAME_COMPUTE) {
      witnessIfs.push_back(ifOp);
    } else if (*productSource == FUNC_NAME_CONSTRAIN) {
      constraintIfs.push_back(ifOp);
    }
    return WalkResult::skip();
  });

  auto fusionCandidates =
      alignmentHelpers::getMatchingPairs<scf::IfOp>(witnessIfs, constraintIfs, canIfsBeFused);
  if (failed(fusionCandidates)) {
    return failure();
  }

  IRRewriter rewriter {context};
  for (auto [w, c] : *fusionCandidates) {
    if (failed(fuseIfPair(w, c, context, rewriter))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult fuseMatchingLoopPairs(Region &body, MLIRContext *context) {
  // Start by collecting all possible loops
  llvm::SmallVector<scf::ForOp> witnessLoops, constraintLoops;
  body.walk<WalkOrder::PreOrder>([&witnessLoops, &constraintLoops](scf::ForOp forOp) {
    std::optional<llvm::StringRef> productSource = getProductSource(forOp);
    if (!productSource) {
      return WalkResult::skip();
    }
    if (*productSource == FUNC_NAME_COMPUTE) {
      witnessLoops.push_back(forOp);
    } else if (*productSource == FUNC_NAME_CONSTRAIN) {
      constraintLoops.push_back(forOp);
    }
    // Skipping here, because any nested loops can't possibly be fused at this stage
    return WalkResult::skip();
  });

  // A pair of loops will be fused iff (1) they can be fused according to the rules above, and (2)
  // neither can be fused with anything else (so there's no ambiguity)
  auto fusionCandidates = alignmentHelpers::getMatchingPairs<scf::ForOp>(
      witnessLoops, constraintLoops, canLoopsBeFused
  );

  // This shouldn't happen, since we allow partial matches
  if (failed(fusionCandidates)) {
    return failure();
  }

  // Finally, fuse all the marked loops...
  IRRewriter rewriter {context};
  for (auto [w, c] : *fusionCandidates) {
    auto fusedLoop = fuseIndependentSiblingForLoops(w, c, rewriter);
    fusedLoop->setAttr(PRODUCT_SOURCE, rewriter.getAttr<StringAttr>("fused"));
    // ...and recurse to fuse nested control flow
    if (failed(fuseMatchingRegionControlFlow(fusedLoop.getBodyRegion(), context))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult fuseMatchingRegionControlFlow(Region &body, MLIRContext *context) {
  if (failed(fuseMatchingIfPairs(body, context))) {
    return failure();
  }
  return fuseMatchingLoopPairs(body, context);
}

class PassImpl : public llzk::impl::FuseProductLoopsPassBase<PassImpl> {
  using Base = FuseProductLoopsPassBase<PassImpl>;
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([this](function::FuncDefOp funcDef) {
      if (funcDef.isStructProduct()) {
        if (failed(fuseMatchingRegionControlFlow(funcDef.getFunctionBody(), &getContext()))) {
          signalPassFailure();
        }
      }
    });
  }
};

} // namespace
