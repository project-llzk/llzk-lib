
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
/// This file implements the `-llzk-fuse-product-ifs` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/LightweightSignalEquivalenceAnalysis.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Transforms/LLZKProductFusion.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/AlignmentHelper.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>

#include <llvm/Support/Debug.h>

#include <memory>

#define GEN_PASS_DECL_FUSEPRODUCTIFSPASS
#define GEN_PASS_DEF_FUSEPRODUCTIFSPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"

namespace llzk {

using namespace llzk::function;
using namespace mlir::scf;

class FuseProductIfsPass : public impl::FuseProductIfsPassBase<FuseProductIfsPass> {
public:
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();

    mod->walk([this](FuncDefOp funcDef) {
      LightweightSignalEquivalenceAnalysis equiv {funcDef};
      if (mlir::failed(fuseMatchingIfPairs(funcDef.getBody(), &getContext(), equiv))) {
        signalPassFailure();
      }
    });
    mod->dumpPretty();
  }
};

bool canIfsBeFused(IfOp a, IfOp b, LightweightSignalEquivalenceAnalysis &equiv) {
  // Two if-statements can be fused if:
  // 1. They have the same parent region,
  // 2. One is compute and the other is constrain, and
  // 3. Their conditions are equivalent

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
  return equiv.areSignalsEquivalent(a.getCondition(), b.getCondition());
}

IfOp fuseSiblingIfs(IfOp &target, IfOp source, mlir::RewriterBase &rewriter) {
  [[maybe_unused]] unsigned numTargetOuts = target.getNumResults();
  [[maybe_unused]] unsigned numSourceOuts = source.getNumResults();

  mlir::SmallVector<mlir::Type> resultTypes;
  for (auto type : source->getResultTypes()) {
    resultTypes.push_back(type);
  }
  for (auto type : target.getResultTypes()) {
    resultTypes.push_back(type);
  }

  rewriter.setInsertionPointAfter(source);
  bool hasElse = !source.getElseRegion().empty();

  IfOp fused = rewriter.create<IfOp>(source.getLoc(), resultTypes, source.getCondition(), hasElse);
  mlir::IRMapping mapping;

  auto fuseBlocks = [&rewriter, &mapping](mlir::Block *target, mlir::Block *source) {
    for (auto &op : source->without_terminator()) {
      rewriter.clone(op, mapping);
    }
    for (auto &op : target->without_terminator()) {
      rewriter.clone(op, mapping);
    }
    mlir::SmallVector<mlir::Value> results;
    for (auto result : source->getTerminator()->getOperands()) {
      results.push_back(mapping.lookupOrDefault(result));
    }
    for (auto result : target->getTerminator()->getOperands()) {
      results.push_back(mapping.lookupOrDefault(result));
    }
    rewriter.create<mlir::scf::YieldOp>(source->getTerminator()->getLoc(), results);
  };

  rewriter.setInsertionPointToStart(fused.thenBlock());
  fuseBlocks(target.thenBlock(), source.thenBlock());
  if (hasElse) {
    rewriter.setInsertionPointToStart(fused.elseBlock());
    fuseBlocks(target.elseBlock(), source.elseBlock());
  }

  rewriter.replaceOp(source, fused.getResults().take_front(numSourceOuts));
  rewriter.replaceOp(target, fused.getResults().take_back(numTargetOuts));

  return fused;
}

mlir::LogicalResult fuseMatchingIfPairs(
    mlir::Region &body, mlir::MLIRContext *context, LightweightSignalEquivalenceAnalysis &equiv
) {
  // TODO: A lot of this logic is repeated with LLZKFuseProductLoopsPass.cpp
  // Start by collecting all possible ifs
  llvm::SmallVector<mlir::scf::IfOp> witnessIfs, constraintIfs;
  body.walk<mlir::WalkOrder::PreOrder>([&witnessIfs, &constraintIfs](mlir::scf::IfOp ifOp) {
    if (!ifOp->hasAttrOfType<mlir::StringAttr>(PRODUCT_SOURCE)) {
      return mlir::WalkResult::skip();
    }
    auto productSource = ifOp->getAttrOfType<mlir::StringAttr>(PRODUCT_SOURCE);
    if (productSource == FUNC_NAME_COMPUTE) {
      witnessIfs.push_back(ifOp);
    } else if (productSource == FUNC_NAME_CONSTRAIN) {
      constraintIfs.push_back(ifOp);
    }
    // Skipping here, because any nested loops can't possibly be fused at this stage
    return mlir::WalkResult::skip();
  });

  // A pair of scf.ifs will be fused iff (1) they can be fused according to the rules above,
  // and (2) neither can be fused with anything else (so there's no ambiguity)
  auto fusionCandidates = alignmentHelpers::getMatchingPairs<mlir::scf::IfOp>(
      witnessIfs, constraintIfs, [&equiv](auto a, auto b) { return canIfsBeFused(a, b, equiv); }
  );

  // This shouldn't happen, since we allow partial matches
  if (mlir::failed(fusionCandidates)) {
    return mlir::failure();
  }

  mlir::IRRewriter rewriter {context};
  for (auto [w, c] : *fusionCandidates) {
    auto fusedIf = fuseSiblingIfs(w, c, rewriter);
    fusedIf->setAttr(PRODUCT_SOURCE, rewriter.getAttr<mlir::StringAttr>("fused"));
    // ...and recurse to fuse nested loops
    if (mlir::failed(fuseMatchingIfPairs(fusedIf.getBodyRegion(), context, equiv))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

std::unique_ptr<mlir::Pass> createFuseProductIfsPass() {
  return std::make_unique<FuseProductIfsPass>();
}
} // namespace llzk
