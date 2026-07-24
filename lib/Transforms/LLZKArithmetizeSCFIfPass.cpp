//===-- LLZKArithmetizeSCFIfPass.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-arithmetize-scf-if` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/OpTraits.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

#include <utility>

namespace llzk {
#define GEN_PASS_DEF_ARITHMETIZESCFIFPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

#define DEBUG_TYPE "llzk-arithmetize-scf-if"

using namespace mlir;
using namespace llzk;
using namespace llzk::cast;
using namespace llzk::constrain;
using namespace llzk::felt;
using namespace llzk::function;

namespace {

/// Rewrites every `scf.if` in one constraint-bearing function. The rewrite is
/// bottom-up so an outer branch can add its gate to constraints produced while
/// lowering nested conditionals.
class ConstraintIfArithmetizer {
public:
  explicit ConstraintIfArithmetizer(FuncDefOp func) : fn(func), rewriter(func.getContext()) {}

  LogicalResult run();

private:
  struct Gates {
    Value whenTrue;
    Value whenFalse;
  };

  FuncDefOp fn;
  IRRewriter rewriter;

  /// Constants shared by the entire function. They live in an entry-block
  /// prelude so they dominate conditionals in sibling regions.
  llvm::DenseMap<std::pair<Type, uint64_t>, Value> constPool;
  Operation *lastPreludeOp = nullptr;

  LogicalResult validate(mlir::scf::IfOp ifOp);
  LogicalResult validateBranch(mlir::scf::IfOp ifOp, Block &block);
  bool isSafeToExecuteUnconditionally(Operation *op);
  bool requiresFeltAlgebraization(Operation *op);

  Value feltConst(Location loc, FeltType type, uint64_t value);
  Gates getGates(mlir::scf::IfOp ifOp, FeltType type, llvm::DenseMap<Type, Gates> &gates);
  void gateConstraints(
      mlir::scf::IfOp ifOp, Block &block, bool thenBranch, llvm::DenseMap<Type, Gates> &gates
  );
  void inlineBranch(mlir::scf::IfOp ifOp, Block &block);
  void rewrite(mlir::scf::IfOp ifOp);
};

bool ConstraintIfArithmetizer::requiresFeltAlgebraization(Operation *op) {
  if (llvm::isa<DivFeltOp>(op)) {
    return true;
  }
  return op->getName().getDialectNamespace() != "bool" && op->hasTrait<NotFieldNative>();
}

bool ConstraintIfArithmetizer::isSafeToExecuteUnconditionally(Operation *op) {
  return mlir::isSpeculatable(op);
}

LogicalResult ConstraintIfArithmetizer::validateBranch(mlir::scf::IfOp ifOp, Block &block) {
  auto yield = llvm::dyn_cast<mlir::scf::YieldOp>(block.getTerminator());
  if (!yield) {
    ifOp.emitError("expected each scf.if branch to end with scf.yield");
    return failure();
  }

  for (Operation &op : block.without_terminator()) {
    if (llvm::isa<mlir::scf::IfOp>(op)) {
      continue;
    }
    if (auto eq = llvm::dyn_cast<EmitEqualityOp>(op)) {
      if (!llvm::isa<FeltType>(eq.getLhs().getType())) {
        eq.emitError("scf.if arithmetization supports only felt equality constraints");
        return failure();
      }
      continue;
    }
    if (requiresFeltAlgebraization(&op)) {
      op.emitError("felt division and non-native operations must be lowered before scf.if");
      return failure();
    }
    if (op.getNumRegions() != 0 || op.getNumSuccessors() != 0 ||
        !isSafeToExecuteUnconditionally(&op)) {
      op.emitError() << "cannot arithmetize scf.if branch operation '" << op.getName()
                     << "' because executing both branches may change its effects";
      return failure();
    }
  }
  return success();
}

LogicalResult ConstraintIfArithmetizer::validate(mlir::scf::IfOp ifOp) {
  if (!ifOp.getThenRegion().hasOneBlock() ||
      (!ifOp.getElseRegion().empty() && !ifOp.getElseRegion().hasOneBlock())) {
    ifOp.emitError("scf.if arithmetization requires single-block branches");
    return failure();
  }
  if (llvm::any_of(ifOp.getResultTypes(), [](Type type) { return !llvm::isa<FeltType>(type); })) {
    ifOp.emitError("scf.if arithmetization supports only felt results");
    return failure();
  }
  if (failed(validateBranch(ifOp, ifOp.getThenRegion().front()))) {
    return failure();
  }
  if (!ifOp.getElseRegion().empty() && failed(validateBranch(ifOp, ifOp.getElseRegion().front()))) {
    return failure();
  }
  return success();
}

Value ConstraintIfArithmetizer::feltConst(Location loc, FeltType type, uint64_t value) {
  auto [it, inserted] = constPool.try_emplace({type, value});
  if (!inserted) {
    return it->second;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  if (lastPreludeOp) {
    rewriter.setInsertionPointAfter(lastPreludeOp);
  } else {
    rewriter.setInsertionPointToStart(&fn.getBody().front());
  }
  auto attr = FeltConstAttr::get(rewriter.getContext(), llvm::APInt(2, value), type);
  it->second = rewriter.create<FeltConstantOp>(loc, type, attr).getResult();
  lastPreludeOp = it->second.getDefiningOp();
  return it->second;
}

ConstraintIfArithmetizer::Gates ConstraintIfArithmetizer::getGates(
    mlir::scf::IfOp ifOp, FeltType type, llvm::DenseMap<Type, Gates> &gates
) {
  auto [it, inserted] = gates.try_emplace(type);
  if (inserted) {
    rewriter.setInsertionPoint(ifOp);
    Value whenTrue =
        rewriter.create<IntToFeltOp>(ifOp.getLoc(), type, ifOp.getCondition()).getResult();
    Value whenFalse =
        rewriter.create<SubFeltOp>(ifOp.getLoc(), feltConst(ifOp.getLoc(), type, 1), whenTrue);
    it->second = {whenTrue, whenFalse};
  }
  return it->second;
}

void ConstraintIfArithmetizer::gateConstraints(
    mlir::scf::IfOp ifOp, Block &block, bool thenBranch, llvm::DenseMap<Type, Gates> &gates
) {
  for (Operation &op : llvm::make_early_inc_range(block.without_terminator())) {
    auto eq = llvm::dyn_cast<EmitEqualityOp>(op);
    if (!eq) {
      continue;
    }

    auto type = llvm::cast<FeltType>(eq.getLhs().getType());
    Gates branchGates = getGates(ifOp, type, gates);
    Value gate = thenBranch ? branchGates.whenTrue : branchGates.whenFalse;

    rewriter.setInsertionPoint(eq);
    Value difference = rewriter.create<SubFeltOp>(eq.getLoc(), eq.getLhs(), eq.getRhs());
    Value gated = rewriter.create<MulFeltOp>(eq.getLoc(), gate, difference);
    rewriter.create<EmitEqualityOp>(eq.getLoc(), gated, feltConst(eq.getLoc(), type, 0));
    rewriter.eraseOp(eq);
  }
}

void ConstraintIfArithmetizer::inlineBranch(mlir::scf::IfOp ifOp, Block &block) {
  rewriter.eraseOp(block.getTerminator());
  rewriter.inlineBlockBefore(&block, ifOp);
}

void ConstraintIfArithmetizer::rewrite(mlir::scf::IfOp ifOp) {
  llvm::DenseMap<Type, Gates> gates;

  Block &thenBlock = ifOp.getThenRegion().front();
  auto thenYield = llvm::cast<mlir::scf::YieldOp>(thenBlock.getTerminator());
  SmallVector<Value> thenValues(thenYield.getOperands());
  gateConstraints(ifOp, thenBlock, /*thenBranch=*/true, gates);

  SmallVector<Value> elseValues;
  Block *elseBlock = nullptr;
  if (!ifOp.getElseRegion().empty()) {
    elseBlock = &ifOp.getElseRegion().front();
    auto elseYield = llvm::cast<mlir::scf::YieldOp>(elseBlock->getTerminator());
    elseValues.append(elseYield.getOperands().begin(), elseYield.getOperands().end());
    gateConstraints(ifOp, *elseBlock, /*thenBranch=*/false, gates);
  }

  inlineBranch(ifOp, thenBlock);
  if (elseBlock) {
    inlineBranch(ifOp, *elseBlock);
  }

  rewriter.setInsertionPoint(ifOp);
  SmallVector<Value> replacements;
  replacements.reserve(ifOp.getNumResults());
  for (auto [type, thenValue, elseValue] :
       llvm::zip_equal(ifOp.getResultTypes(), thenValues, elseValues)) {
    Gates resultGates = getGates(ifOp, llvm::cast<FeltType>(type), gates);
    Value difference = rewriter.create<SubFeltOp>(ifOp.getLoc(), thenValue, elseValue);
    Value selected = rewriter.create<MulFeltOp>(ifOp.getLoc(), resultGates.whenTrue, difference);
    replacements.push_back(rewriter.create<AddFeltOp>(ifOp.getLoc(), elseValue, selected));
  }
  rewriter.replaceOp(ifOp, replacements);
}

LogicalResult ConstraintIfArithmetizer::run() {
  SmallVector<mlir::scf::IfOp> ifOps;
  fn.walk<WalkOrder::PostOrder>([&](mlir::scf::IfOp ifOp) { ifOps.push_back(ifOp); });

  // Validate the whole function before changing it. Nested scf.if operations
  // are allowed here because the rewrite list is already bottom-up.
  for (mlir::scf::IfOp ifOp : ifOps) {
    if (failed(validate(ifOp))) {
      return failure();
    }
  }
  for (mlir::scf::IfOp ifOp : ifOps) {
    rewrite(ifOp);
  }
  return success();
}

class PassImpl : public llzk::impl::ArithmetizeSCFIfPassBase<PassImpl> {
  using Base = ArithmetizeSCFIfPassBase<PassImpl>;
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        llzk::cast::CastDialect, llzk::constrain::ConstrainDialect, llzk::felt::FeltDialect>();
  }

  void runOnOperation() override {
    SmallVector<FuncDefOp> funcs;
    getOperation()->walk([&](FuncDefOp fn) {
      if (fn.hasAllowConstraintAttr() && !fn.getBody().empty()) {
        funcs.push_back(fn);
      }
    });

    for (FuncDefOp fn : funcs) {
      if (failed(ConstraintIfArithmetizer(fn).run())) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace
