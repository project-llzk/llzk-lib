//===-- LLZKFuseProductLoopsPass.cpp -----------------------------*- C++ -*-===//
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

#include "llzk/Transforms/LLZKFuseProductLoopsPass.h"

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/AlignmentHelper.h"
#include "llzk/Util/Constants.h"

#include <mlir/Dialect/SCF/Utils/Utils.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/SMTAPI.h>

#include <memory>
#include <optional>

namespace llzk {

#define GEN_PASS_DECL_FUSEPRODUCTLOOPSPASS
#define GEN_PASS_DEF_FUSEPRODUCTLOOPSPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"

using namespace llzk::function;
using namespace llzk::component;

static mlir::LogicalResult
fuseMatchingRegionControlFlow(mlir::Region &body, mlir::MLIRContext *context);

// Bitwidth of `index` for instantiating SMT variables
constexpr int INDEX_WIDTH = 64;

class FuseProductLoopsPass : public impl::FuseProductLoopsPassBase<FuseProductLoopsPass> {

public:
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    mod.walk([this](FuncDefOp funcDef) {
      if (funcDef.isStructProduct()) {
        if (mlir::failed(fuseMatchingRegionControlFlow(funcDef.getFunctionBody(), &getContext()))) {
          signalPassFailure();
        }
      }
    });
  }
};

static inline bool isConstOrStructParam(mlir::Value val) {
  // TODO: doing arithmetic over constants should also be fine?
  return val.getDefiningOp<mlir::arith::ConstantIndexOp>() ||
         val.getDefiningOp<llzk::polymorphic::ConstReadOp>();
}

llvm::SMTExprRef mkExpr(mlir::Value value, llvm::SMTSolver *solver) {
  if (auto constOp = value.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
    return solver->mkBitvector(llvm::APSInt::get(constOp.value()), INDEX_WIDTH);
  } else if (auto polyReadOp = value.getDefiningOp<llzk::polymorphic::ConstReadOp>()) {

    return solver->mkSymbol(
        std::string {polyReadOp.getConstName()}.c_str(), solver->getBitvectorSort(INDEX_WIDTH)
    );
  }
  assert(false && "unsupported: checking non-constant trip counts");
  return nullptr; // Unreachable
}

llvm::SMTExprRef tripCount(mlir::scf::ForOp op, llvm::SMTSolver *solver) {
  const auto *one = solver->mkBitvector(llvm::APSInt::get(1), INDEX_WIDTH);
  return solver->mkBVSDiv(
      solver->mkBVAdd(
          one,
          solver->mkBVSub(mkExpr(op.getUpperBound(), solver), mkExpr(op.getLowerBound(), solver))
      ),
      mkExpr(op.getStep(), solver)
  );
}

static inline bool canLoopsBeFused(mlir::scf::ForOp a, mlir::scf::ForOp b) {
  // A priori, two loops can be fused if:
  // 1. They live in the same parent region,
  // 2. One comes from witgen and the other comes from constraint gen, and
  // 3. They have the same trip count

  // Check 1.
  if (a->getParentRegion() != b->getParentRegion()) {
    return false;
  }

  // Check 2.
  if (!a->hasAttrOfType<mlir::StringAttr>(PRODUCT_SOURCE) ||
      !b->hasAttrOfType<mlir::StringAttr>(PRODUCT_SOURCE)) {
    // Ideally this should never happen, since the pass only runs on fused @product functions, but
    // check anyway just to be safe
    return false;
  }
  if (a->getAttrOfType<mlir::StringAttr>(PRODUCT_SOURCE) ==
      b->getAttrOfType<mlir::StringAttr>(PRODUCT_SOURCE)) {
    return false;
  }

  // Check 3.
  // Easy case: both have a constant trip-count. If the trip counts are not "constant up to a struct
  // param", we definitely can't tell if they're equal. If the trip counts are only "constant up to
  // a struct param" but not actually constant, we can ask a solver if the equations are guaranteed
  // to be the same
  auto tripCountA = mlir::constantTripCount(a.getLowerBound(), a.getUpperBound(), a.getStep());
  auto tripCountB = mlir::constantTripCount(b.getLowerBound(), b.getUpperBound(), b.getStep());
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

static std::optional<llvm::StringRef> getProductSource(mlir::Operation *op) {
  if (mlir::StringAttr source = op->getAttrOfType<mlir::StringAttr>(PRODUCT_SOURCE)) {
    return source.getValue();
  }

  // Loop fusion can preserve descendant provenance while dropping it from the control op.
  std::optional<llvm::StringRef> inferredSource;
  bool sourceConflict = false;
  op->walk([&](mlir::Operation *nestedOp) {
    if (nestedOp == op) {
      return mlir::WalkResult::advance();
    }

    mlir::StringAttr nestedSource = nestedOp->getAttrOfType<mlir::StringAttr>(PRODUCT_SOURCE);
    if (!nestedSource) {
      return mlir::WalkResult::advance();
    }

    llvm::StringRef source = nestedSource.getValue();
    if (source != FUNC_NAME_COMPUTE && source != FUNC_NAME_CONSTRAIN) {
      sourceConflict = true;
      return mlir::WalkResult::interrupt();
    }
    if (!inferredSource) {
      inferredSource = source;
      return mlir::WalkResult::advance();
    }
    if (*inferredSource != source) {
      sourceConflict = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  if (sourceConflict) {
    return std::nullopt;
  }
  return inferredSource;
}

static inline bool hasProductSource(mlir::Operation *op, llvm::StringRef source) {
  std::optional<llvm::StringRef> productSource = getProductSource(op);
  return productSource && *productSource == source;
}

static inline bool areOppositeProductSources(mlir::Operation *a, mlir::Operation *b) {
  std::optional<llvm::StringRef> sourceA = getProductSource(a);
  std::optional<llvm::StringRef> sourceB = getProductSource(b);
  if (!sourceA || !sourceB) {
    return false;
  }
  return *sourceA != *sourceB;
}

static bool isBetweenInBlock(mlir::Operation *op, mlir::Operation *before, mlir::Operation *after) {
  return op->getBlock() == before->getBlock() && op->getBlock() == after->getBlock() &&
         before->isBeforeInBlock(op) && op->isBeforeInBlock(after);
}

struct ResultWrite {
  mlir::Value component;
  mlir::FlatSymbolRefAttr memberName;
  unsigned resultIndex;
};

static std::optional<unsigned> getIfResultIndex(mlir::scf::IfOp ifOp, mlir::Value value) {
  for (auto [idx, result] : llvm::enumerate(ifOp.getResults())) {
    if (result == value) {
      return idx;
    }
  }
  return std::nullopt;
}

static bool isPlainReadOfWrite(MemberReadOp readOp, const ResultWrite &write) {
  return readOp.getComponent() == write.component &&
         readOp.getMemberNameAttr() == write.memberName && readOp->getNumOperands() == 1 &&
         !readOp->hasAttr("tableOffset");
}

static bool collectConstrainValueMappings(
    mlir::scf::IfOp computeIf, mlir::scf::IfOp constrainIf,
    llvm::DenseMap<mlir::Value, unsigned> &valueToResult,
    llvm::SmallVectorImpl<MemberReadOp> *mappedReads = nullptr
) {
  llvm::SmallVector<ResultWrite> writes;
  for (mlir::Operation *op = computeIf->getNextNode(); op != constrainIf; op = op->getNextNode()) {
    if (auto writeOp = llvm::dyn_cast<MemberWriteOp>(op)) {
      std::optional<unsigned> resultIndex = getIfResultIndex(computeIf, writeOp.getVal());
      if (!resultIndex) {
        return false;
      }
      writes.push_back({writeOp.getComponent(), writeOp.getMemberNameAttr(), *resultIndex});
      continue;
    }

    if (auto readOp = llvm::dyn_cast<MemberReadOp>(op)) {
      const ResultWrite *mappedWrite = nullptr;
      for (const ResultWrite &write : llvm::reverse(writes)) {
        if (isPlainReadOfWrite(readOp, write)) {
          mappedWrite = &write;
          break;
        }
      }
      if (!mappedWrite) {
        return false;
      }
      if (!llvm::all_of(readOp.getVal().getUsers(), [&](mlir::Operation *user) {
        return constrainIf->isAncestor(user);
      })) {
        return false;
      }
      valueToResult[readOp.getVal()] = mappedWrite->resultIndex;
      if (mappedReads) {
        mappedReads->push_back(readOp);
      }
      continue;
    }

    return false;
  }

  for (auto [idx, result] : llvm::enumerate(computeIf.getResults())) {
    valueToResult[result] = idx;
  }

  auto hasInternalRead =
      constrainIf->walk([](MemberReadOp) { return mlir::WalkResult::interrupt(); });
  if (hasInternalRead.wasInterrupted()) {
    return false;
  }

  auto result = constrainIf->walk([&](mlir::Operation *op) {
    for (mlir::Value operand : op->getOperands()) {
      mlir::Operation *def = operand.getDefiningOp();
      if (!def || constrainIf->isAncestor(def)) {
        continue;
      }
      if (valueToResult.contains(operand)) {
        continue;
      }
      if (!isBetweenInBlock(def, computeIf, constrainIf)) {
        continue;
      }
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return !result.wasInterrupted();
}

static bool canIfsBeFused(mlir::scf::IfOp a, mlir::scf::IfOp b) {
  if (a->getBlock() != b->getBlock()) {
    return false;
  }
  if (!areOppositeProductSources(a, b)) {
    return false;
  }

  mlir::scf::IfOp computeIf = hasProductSource(a, FUNC_NAME_COMPUTE) ? a : b;
  mlir::scf::IfOp constrainIf = computeIf == a ? b : a;
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

  llvm::DenseMap<mlir::Value, unsigned> valueToResult;
  return collectConstrainValueMappings(computeIf, constrainIf, valueToResult);
}

static void eraseDefaultTerminator(mlir::Block *block) {
  if (!block->empty()) {
    if (auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(block->back())) {
      yieldOp.erase();
    }
  }
}

static void cloneIfBranch(
    mlir::scf::IfOp computeIf, mlir::Block *computeBlock, mlir::Block *constrainBlock,
    mlir::Block *destBlock, const llvm::DenseMap<mlir::Value, unsigned> &valueToResult,
    mlir::OpBuilder &builder
) {
  eraseDefaultTerminator(destBlock);
  mlir::IRMapping mapper;
  builder.setInsertionPointToEnd(destBlock);

  mlir::scf::YieldOp computeYield = llvm::cast<mlir::scf::YieldOp>(computeBlock->getTerminator());
  for (mlir::Operation &op : computeBlock->without_terminator()) {
    builder.clone(op, mapper);
  }
  for (auto [value, resultIndex] : valueToResult) {
    mlir::Value branchValue = computeYield.getResults()[resultIndex];
    mapper.map(value, mapper.lookupOrDefault(branchValue));
  }
  for (mlir::Operation &op : constrainBlock->without_terminator()) {
    builder.clone(op, mapper);
  }

  llvm::SmallVector<mlir::Value> yieldOperands;
  yieldOperands.reserve(computeYield.getResults().size());
  for (mlir::Value operand : computeYield.getResults()) {
    yieldOperands.push_back(mapper.lookupOrDefault(operand));
  }
  builder.create<mlir::scf::YieldOp>(computeIf.getLoc(), yieldOperands);
}

static mlir::LogicalResult fuseIfPair(
    mlir::scf::IfOp a, mlir::scf::IfOp b, mlir::MLIRContext *context, mlir::IRRewriter &rewriter
) {
  mlir::scf::IfOp computeIf = hasProductSource(a, FUNC_NAME_COMPUTE) ? a : b;
  mlir::scf::IfOp constrainIf = computeIf == a ? b : a;

  llvm::DenseMap<mlir::Value, unsigned> valueToResult;
  llvm::SmallVector<MemberReadOp> mappedReads;
  [[maybe_unused]] bool canMap =
      collectConstrainValueMappings(computeIf, constrainIf, valueToResult, &mappedReads);
  assert(canMap && "fusion candidates must have already been checked");

  rewriter.setInsertionPoint(computeIf);
  mlir::scf::IfOp fusedIf = rewriter.create<mlir::scf::IfOp>(
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

  if (mlir::failed(fuseMatchingRegionControlFlow(fusedIf.getThenRegion(), context))) {
    return mlir::failure();
  }
  if (!fusedIf.getElseRegion().empty() &&
      mlir::failed(fuseMatchingRegionControlFlow(fusedIf.getElseRegion(), context))) {
    return mlir::failure();
  }

  computeIf->replaceAllUsesWith(fusedIf->getResults());
  rewriter.eraseOp(constrainIf);
  rewriter.eraseOp(computeIf);
  for (MemberReadOp readOp : mappedReads) {
    rewriter.eraseOp(readOp);
  }
  return mlir::success();
}

static mlir::LogicalResult fuseMatchingIfPairs(mlir::Region &body, mlir::MLIRContext *context) {
  llvm::SmallVector<mlir::scf::IfOp> witnessIfs, constraintIfs;
  body.walk<mlir::WalkOrder::PreOrder>([&](mlir::scf::IfOp ifOp) {
    std::optional<llvm::StringRef> productSource = getProductSource(ifOp);
    if (!productSource) {
      return mlir::WalkResult::advance();
    }
    if (*productSource == FUNC_NAME_COMPUTE) {
      witnessIfs.push_back(ifOp);
    } else if (*productSource == FUNC_NAME_CONSTRAIN) {
      constraintIfs.push_back(ifOp);
    }
    return mlir::WalkResult::skip();
  });

  auto fusionCandidates =
      alignmentHelpers::getMatchingPairs<mlir::scf::IfOp>(witnessIfs, constraintIfs, canIfsBeFused);
  if (mlir::failed(fusionCandidates)) {
    return mlir::failure();
  }

  mlir::IRRewriter rewriter {context};
  for (auto [w, c] : *fusionCandidates) {
    if (mlir::failed(fuseIfPair(w, c, context, rewriter))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

mlir::LogicalResult fuseMatchingLoopPairs(mlir::Region &body, mlir::MLIRContext *context) {
  // Start by collecting all possible loops
  llvm::SmallVector<mlir::scf::ForOp> witnessLoops, constraintLoops;
  body.walk<mlir::WalkOrder::PreOrder>([&witnessLoops, &constraintLoops](mlir::scf::ForOp forOp) {
    std::optional<llvm::StringRef> productSource = getProductSource(forOp);
    if (!productSource) {
      return mlir::WalkResult::skip();
    }
    if (*productSource == FUNC_NAME_COMPUTE) {
      witnessLoops.push_back(forOp);
    } else if (*productSource == FUNC_NAME_CONSTRAIN) {
      constraintLoops.push_back(forOp);
    }
    // Skipping here, because any nested loops can't possibly be fused at this stage
    return mlir::WalkResult::skip();
  });

  // A pair of loops will be fused iff (1) they can be fused according to the rules above, and (2)
  // neither can be fused with anything else (so there's no ambiguity)
  auto fusionCandidates = alignmentHelpers::getMatchingPairs<mlir::scf::ForOp>(
      witnessLoops, constraintLoops, canLoopsBeFused
  );

  // This shouldn't happen, since we allow partial matches
  if (mlir::failed(fusionCandidates)) {
    return mlir::failure();
  }

  // Finally, fuse all the marked loops...
  mlir::IRRewriter rewriter {context};
  for (auto [w, c] : *fusionCandidates) {
    auto fusedLoop = mlir::fuseIndependentSiblingForLoops(w, c, rewriter);
    fusedLoop->setAttr(PRODUCT_SOURCE, rewriter.getAttr<mlir::StringAttr>("fused"));
    // ...and recurse to fuse nested control flow
    if (mlir::failed(fuseMatchingRegionControlFlow(fusedLoop.getBodyRegion(), context))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

mlir::LogicalResult fuseMatchingRegionControlFlow(mlir::Region &body, mlir::MLIRContext *context) {
  if (mlir::failed(fuseMatchingIfPairs(body, context))) {
    return mlir::failure();
  }
  return fuseMatchingLoopPairs(body, context);
}

std::unique_ptr<mlir::Pass> createFuseProductLoopsPass() {
  return std::make_unique<FuseProductLoopsPass>();
}
} // namespace llzk
