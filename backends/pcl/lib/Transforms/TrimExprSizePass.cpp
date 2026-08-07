//===-- TrimExprSizePass.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "pcl/Dialect/IR/Dialect.h"
#include "pcl/Dialect/IR/Ops.h"
#include "pcl/Transforms/TransformationPasses.h"

#include <mlir/IR/Builders.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>

#include <algorithm>

using namespace mlir;

// Include the generated base pass class definitions.
namespace pcl {
#define GEN_PASS_DEF_TRIMEXPRSIZEPASS
#include "pcl/Transforms/TransformationPasses.h.inc"
} // namespace pcl

namespace {

/// Maps the expresion sizes of Values in the IR.
class ExprSizes {
  llvm::DenseMap<Value, unsigned> sizes;
  unsigned maxSize;
  llvm::DenseSet<Operation *> &cuts;

public:
  ExprSizes(unsigned MAX, llvm::DenseSet<Operation *> &C) : maxSize(MAX), cuts(C) {}

  unsigned operator[](Value v) { return get(v); }

  unsigned get(Value v) {
    auto *defOp = v.getDefiningOp();
    if (!defOp || isa<pcl::VarOp, func::CallOp>(defOp) || cuts.contains(defOp)) {
      return 1;
    }
    if (sizes.contains(v)) {
      return sizes.at(v);
    }

    unsigned size = 1;
    for (auto operand : defOp->getOperands()) {
      size += get(operand);
      if (size > maxSize) {
        break;
      }
    }
    sizes[v] = size;
    return size;
  }
};

class PassImpl : public pcl::impl::TrimExprSizePassBase<PassImpl> {
  using pcl::impl::TrimExprSizePassBase<PassImpl>::TrimExprSizePassBase;

  struct Plan {
    llvm::DenseSet<Operation *> cuts;
    llvm::StringSet<> names;
    unsigned bindingsCount = 0;

    std::string nextBinding() {
      auto name = ("t" + Twine(bindingsCount)).str();
      while (names.contains(name)) {
        bindingsCount++;
        name = ("t" + Twine(bindingsCount)).str();
      }
      bindingsCount++;
      return name;
    }
  };

  void runOnOperation() override {
    auto plan = collectPlan();
    OpBuilder builder(getOperation());

    for (auto *cut : plan.cuts) {
      auto loc = cut->getLoc();
      if (cut->getNumResults() != 1) {
        cut->emitOpError() << "cannot trim op with " << cut->getNumResults() << " results";
        signalPassFailure();
        return;
      }
      Value cutOpResult = cut->getResult(0);

      auto name = plan.nextBinding();
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(cut);
      auto varOp = builder.create<pcl::VarOp>(loc, name, false);

      if (isa<pcl::BoolType>(cutOpResult.getType())) {
        auto value = pcl::FeltAttr::get(&getContext(), llvm::APInt::getZero(2));
        auto zeroOp = builder.create<pcl::ConstOp>(loc, value);
        auto cmpEqOp = builder.create<pcl::CmpEqOp>(loc, varOp, zeroOp);
        auto notOp = builder.create<pcl::NotOp>(loc, cmpEqOp);
        cut->replaceAllUsesWith(notOp);
      } else {
        cut->replaceAllUsesWith(varOp);
      }
      if (isa<pcl::BoolType>(cutOpResult.getType())) {
        cutOpResult = builder.create<pcl::AsFeltOp>(loc, cutOpResult);
      }
      auto cmpEqOp = builder.create<pcl::CmpEqOp>(loc, varOp, cutOpResult);
      builder.create<pcl::AssertOp>(loc, cmpEqOp);
    }
  }

  Plan collectPlan() {
    auto max = safeMaxSize();
    llvm::DenseSet<Operation *> cuts;
    llvm::StringSet<> names;
    ExprSizes sizes(max, cuts);

    getOperation()->walk([&sizes, &cuts, &names, max](Operation *op) {
      if (auto varOp = mlir::dyn_cast_if_present<pcl::VarOp>(op)) {
        names.insert(varOp.getName());
      }
      if (op->getNumResults() == 0) {
        return;
      }

      for (auto result : op->getResults()) {
        auto size = sizes[result];
        if (size > max) {
          cuts.insert(op);
        }
      }
    });
    return {.cuts = cuts, .names = names};
  }

  /// Returns the maximum size, capping to a minimum value of 2 if the user passed a smaller size.
  unsigned safeMaxSize() {
    auto size = maxSize.getValue();
    if (size < 2) {
      getOperation()->emitWarning("maximum sizes smaller than 2 are capped to that value");
    }
    return std::max({size, static_cast<unsigned>(2)});
  }
};

} // namespace
