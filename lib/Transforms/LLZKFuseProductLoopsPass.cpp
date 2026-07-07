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

#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/AlignmentHelper.h"
#include "llzk/Util/Constants.h"

#include <mlir/Dialect/SCF/Utils/Utils.h>

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

// Bitwidth of `index` for instantiating SMT variables
constexpr int INDEX_WIDTH = 64;

static inline bool isConstOrStructParam(Value val) {
  // TODO: doing arithmetic over constants should also be fine?
  return val.getDefiningOp<arith::ConstantIndexOp>() ||
         val.getDefiningOp<polymorphic::ConstReadOp>() || val.getDefiningOp<felt::FeltConstantOp>();
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
  if (!a->hasAttrOfType<StringAttr>(PRODUCT_SOURCE) ||
      !b->hasAttrOfType<StringAttr>(PRODUCT_SOURCE)) {
    // Ideally this should never happen, since the pass only runs on fused @product functions, but
    // check anyway just to be safe
    return false;
  }
  if (a->getAttrOfType<StringAttr>(PRODUCT_SOURCE) ==
      b->getAttrOfType<StringAttr>(PRODUCT_SOURCE)) {
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

/// Determine which ops need to sink past `constraintLoop`, or return failure() if some of these
/// ops can't be sunk. Conservatively tries to sink all compute ops, but we could do a more precise
/// analysis here
static FailureOr<SmallVector<Operation *>>
canPrepareForFusion(scf::ForOp witnessLoop, scf::ForOp constraintLoop) {
  if (witnessLoop->getBlock() != constraintLoop->getBlock()) {
    return failure();
  }

  SmallVector<Operation *> opsToSink;
  for (auto *op = witnessLoop->getNextNode(); op != constraintLoop; op = op->getNextNode()) {
    if (op->getAttrOfType<StringAttr>(PRODUCT_SOURCE) == "fused") {
      // "fused" means "compute" + "constrain". Conservatively, a "compute" op we want to sink can't
      // be sunk if it also has "constrain" since we need to preserve the relative orders within
      // compute/constrain
      return failure();
    }
    if (op->getAttrOfType<StringAttr>(PRODUCT_SOURCE) == FUNC_NAME_COMPUTE) {
      opsToSink.push_back(op);
    }
  }
  return opsToSink;
}

static LogicalResult
prepareForFusion(scf::ForOp witnessLoop, scf::ForOp constraintLoop, IRRewriter &rewriter) {
  auto computeOpsToSink = canPrepareForFusion(witnessLoop, constraintLoop);
  if (failed(computeOpsToSink)) {
    return failure();
  }

  Operation *insertionPoint = constraintLoop.getOperation();
  for (Operation *op : *computeOpsToSink) {
    rewriter.moveOpAfter(op, insertionPoint);
    insertionPoint = op;
  }

  return success();
}

static LogicalResult fuseMatchingLoopPairs(Region &body, MLIRContext *context) {
  // Start by collecting all possible loops
  llvm::SmallVector<scf::ForOp> witnessLoops, constraintLoops;
  body.walk<WalkOrder::PreOrder>([&witnessLoops, &constraintLoops](scf::ForOp forOp) {
    if (!forOp->hasAttrOfType<StringAttr>(PRODUCT_SOURCE)) {
      return WalkResult::skip();
    }
    auto productSource = forOp->getAttrOfType<StringAttr>(PRODUCT_SOURCE);
    if (productSource == FUNC_NAME_COMPUTE) {
      witnessLoops.push_back(forOp);
    } else if (productSource == FUNC_NAME_CONSTRAIN) {
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
    if (failed(prepareForFusion(w, c, rewriter))) {
      continue;
    }
    auto fusedLoop = fuseIndependentSiblingForLoops(w, c, rewriter);
    fusedLoop->setAttr(PRODUCT_SOURCE, rewriter.getAttr<StringAttr>("fused"));
    // ...and recurse to fuse nested loops
    if (failed(fuseMatchingLoopPairs(fusedLoop.getBodyRegion(), context))) {
      return failure();
    }
  }
  return success();
}

class PassImpl : public llzk::impl::FuseProductLoopsPassBase<PassImpl> {
  using Base = FuseProductLoopsPassBase<PassImpl>;
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([this](function::FuncDefOp funcDef) {
      if (funcDef.isStructProduct()) {
        if (failed(fuseMatchingLoopPairs(funcDef.getFunctionBody(), &getContext()))) {
          signalPassFailure();
        }
      }
    });
  }
};

} // namespace
