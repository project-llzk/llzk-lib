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

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKProductFusion.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/AlignmentHelper.h"
#include "llzk/Util/Constants.h"

#include <mlir/Dialect/SCF/Utils/Utils.h>

#include <llvm/Support/Debug.h>
#include <llvm/Support/SMTAPI.h>

#include <memory>

namespace llzk {

#define GEN_PASS_DECL_FUSEPRODUCTLOOPSPASS
#define GEN_PASS_DEF_FUSEPRODUCTLOOPSPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"

using namespace llzk::function;

// Bitwidth of `index` for instantiating SMT variables
constexpr int INDEX_WIDTH = 64;

class FuseProductLoopsPass : public impl::FuseProductLoopsPassBase<FuseProductLoopsPass> {

public:
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    mod.walk([this](FuncDefOp funcDef) {
      if (funcDef.isStructProduct()) {
        if (mlir::failed(fuseMatchingLoopPairs(funcDef.getFunctionBody(), &getContext()))) {
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

mlir::LogicalResult fuseMatchingLoopPairs(mlir::Region &body, mlir::MLIRContext *context) {
  // Start by collecting all possible loops
  llvm::SmallVector<mlir::scf::ForOp> witnessLoops, constraintLoops;
  body.walk<mlir::WalkOrder::PreOrder>([&witnessLoops, &constraintLoops](mlir::scf::ForOp forOp) {
    if (!forOp->hasAttrOfType<mlir::StringAttr>(PRODUCT_SOURCE)) {
      return mlir::WalkResult::skip();
    }
    auto productSource = forOp->getAttrOfType<mlir::StringAttr>(PRODUCT_SOURCE);
    if (productSource == FUNC_NAME_COMPUTE) {
      witnessLoops.push_back(forOp);
    } else if (productSource == FUNC_NAME_CONSTRAIN) {
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
    // ...and recurse to fuse nested loops
    if (mlir::failed(fuseMatchingLoopPairs(fusedLoop.getBodyRegion(), context))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

std::unique_ptr<mlir::Pass> createFuseProductLoopsPass() {
  return std::make_unique<FuseProductLoopsPass>();
}
} // namespace llzk
