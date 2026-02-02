//===-- LLZKComputeConstrainToProductPass.cpp -------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-product-program` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/LightweightSignalEquivalenceAnalysis.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKComputeConstrainToProductPass.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/Constants.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/IR/Builders.h>
#include <mlir/Transforms/InliningUtils.h>

#include <llvm/Support/Debug.h>

#include <memory>
#include <ranges>

namespace llzk {
#define GEN_PASS_DECL_COMPUTECONSTRAINTOPRODUCTPASS
#define GEN_PASS_DEF_COMPUTECONSTRAINTOPRODUCTPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

#define DEBUG_TYPE "llzk-compute-constrain-to-product-pass"

using namespace llzk::component;
using namespace llzk::function;
using namespace mlir;

using std::make_unique;

namespace llzk {

bool isValidRoot(StructDefOp root) {
  FuncDefOp computeFunc = root.getComputeFuncOp();
  FuncDefOp constrainFunc = root.getConstrainFuncOp();

  if (!computeFunc || !constrainFunc) {
    root->emitError() << "no " << FUNC_NAME_COMPUTE << "/" << FUNC_NAME_CONSTRAIN << " to align";
    return false;
  }

  /// TODO: If root::@compute and root::@constrain are called anywhere else, this is not a valid
  /// root to start aligning from (issue #241)

  return true;
}

LogicalResult alignStartingAt(
    component::StructDefOp root, SymbolTableCollection &tables,
    LightweightSignalEquivalenceAnalysis &equivalence
) {
  if (!isValidRoot(root)) {
    return failure();
  }

  ProductAligner aligner {tables, equivalence};
  if (!aligner.alignFuncs(root, root.getComputeFuncOp(), root.getConstrainFuncOp())) {
    return failure();
  }

  for (auto s : aligner.alignedStructs) {
    s.getComputeFuncOp()->erase();
    s.getConstrainFuncOp()->erase();
  }

  return success();
}

class ComputeConstrainToProductPass
    : public llzk::impl::ComputeConstrainToProductPassBase<ComputeConstrainToProductPass> {

public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    StructDefOp root;

    SymbolTableCollection tables;
    LightweightSignalEquivalenceAnalysis equivalence {
        getAnalysis<LightweightSignalEquivalenceAnalysis>()
    };

    // Find the indicated root struct and make sure its a valid place to start aligning
    mod.walk([&root, this](StructDefOp structDef) {
      if (structDef.getSymName() == rootStruct) {
        root = structDef;
      }
    });

    if (failed(alignStartingAt(root, tables, equivalence))) {
      signalPassFailure();
    }
  }
};

FuncDefOp ProductAligner::alignFuncs(StructDefOp root, FuncDefOp compute, FuncDefOp constrain) {
  OpBuilder funcBuilder(compute);

  // Add compute/constrain attributes
  compute.walk([&funcBuilder](Operation *op) {
    op->setAttr("product_source", funcBuilder.getStringAttr(FUNC_NAME_COMPUTE));
  });

  constrain.walk([&funcBuilder](Operation *op) {
    op->setAttr("product_source", funcBuilder.getStringAttr(FUNC_NAME_CONSTRAIN));
  });

  // Create an empty @product func...
  FuncDefOp productFunc = funcBuilder.create<FuncDefOp>(
      funcBuilder.getFusedLoc({compute.getLoc(), constrain.getLoc()}), FUNC_NAME_PRODUCT,
      compute.getFunctionType()
  );
  Block *entryBlock = productFunc.addEntryBlock();
  funcBuilder.setInsertionPointToStart(entryBlock);

  // ...with the right arguments
  llvm::SmallVector<Value> args {productFunc.getArguments()};

  // Add calls to @compute and @constrain...
  CallOp computeCall = funcBuilder.create<CallOp>(funcBuilder.getUnknownLoc(), compute, args);
  args.insert(args.begin(), computeCall->getResult(0));
  CallOp constrainCall = funcBuilder.create<CallOp>(funcBuilder.getUnknownLoc(), constrain, args);
  funcBuilder.create<ReturnOp>(funcBuilder.getUnknownLoc(), computeCall->getResult(0));

  // ..and inline them
  InlinerInterface inliner(productFunc.getContext());
  if (failed(inlineCall(inliner, computeCall, compute, &compute.getBody(), true))) {
    root->emitError() << "failed to inline " << FUNC_NAME_COMPUTE;
    return nullptr;
  }
  if (failed(inlineCall(inliner, constrainCall, constrain, &constrain.getBody(), true))) {
    root->emitError() << "failed to inline " << FUNC_NAME_CONSTRAIN;
    return nullptr;
  }
  computeCall->erase();
  constrainCall->erase();

  // Mark the compute/constrain for deletion
  alignedStructs.push_back(root);

  // Make sure we can align sub-calls to @compute and @constrain
  if (failed(alignCalls(productFunc))) {
    return nullptr;
  }
  return productFunc;
}

LogicalResult ProductAligner::alignCalls(FuncDefOp product) {
  // Gather up all the remaining calls to @compute and @constrain
  llvm::SetVector<CallOp> computeCalls, constrainCalls;
  product.walk([&](CallOp callOp) {
    if (callOp.calleeIsStructCompute()) {
      computeCalls.insert(callOp);
    } else if (callOp.calleeIsStructConstrain()) {
      constrainCalls.insert(callOp);
    }
  });

  llvm::SetVector<std::pair<CallOp, CallOp>> alignedCalls;

  // A @compute matches a @constrain if they belong to the same struct and all their input signals
  // are pairwise equivalent
  auto doCallsMatch = [&](CallOp compute, CallOp constrain) -> bool {
    LLVM_DEBUG({
      llvm::outs() << "Asking for equivalence between calls\n"
                   << compute << "\nand\n"
                   << constrain << "\n\n";
      llvm::outs() << "In block:\n\n" << *compute->getBlock() << "\n";
    });

    auto computeStruct = getPrefixAsSymbolRefAttr(compute.getCallee());
    auto constrainStruct = getPrefixAsSymbolRefAttr(constrain.getCallee());
    if (computeStruct != constrainStruct) {
      return false;
    }
    for (unsigned i = 0, e = compute->getNumOperands() - 1; i < e; i++) {
      if (!equivalence.areSignalsEquivalent(compute->getOperand(i), constrain->getOperand(i + 1))) {
        return false;
      }
    }

    return true;
  };

  for (auto compute : computeCalls) {
    // If there is exactly one @compute that matches a given @constrain, we can align them
    auto matches = llvm::filter_to_vector(constrainCalls, [&](CallOp constrain) {
      return doCallsMatch(compute, constrain);
    });

    if (matches.size() == 1) {
      alignedCalls.insert({compute, matches[0]});
      computeCalls.remove(compute);
      constrainCalls.remove(matches[0]);
    }
  }

  // TODO: If unaligned calls remain, fully inline their structs and continue instead of failing
  if (!computeCalls.empty() && constrainCalls.empty()) {
    product->emitError() << "failed to align some @" << FUNC_NAME_COMPUTE << " and @"
                         << FUNC_NAME_CONSTRAIN;
    return failure();
  }

  for (auto [compute, constrain] : alignedCalls) {
    // If @A::@compute matches @A::@constrain, recursively align the functions in @A...
    auto newRoot = compute.getCalleeTarget(tables)->get()->getParentOfType<StructDefOp>();
    assert(newRoot);
    FuncDefOp newProduct =
        alignFuncs(newRoot, newRoot.getComputeFuncOp(), newRoot.getConstrainFuncOp());
    if (!newProduct) {
      return failure();
    }

    // ...and replace the two calls with a single call to @A::@product
    OpBuilder callBuilder(compute);
    CallOp newCall = callBuilder.create<CallOp>(
        callBuilder.getFusedLoc({compute.getLoc(), constrain.getLoc()}), newProduct,
        compute.getOperands()
    );
    compute->replaceAllUsesWith(newCall.getResults());
    compute->erase();
    constrain->erase();
  }

  return success();
}

std::unique_ptr<Pass> createComputeConstrainToProductPass() {
  return make_unique<ComputeConstrainToProductPass>();
}

} // namespace llzk
