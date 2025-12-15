//===-- LLZKToZKLean.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Conversions/Passes.h"

#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderOps.h"
#include "llzk/Dialect/ZKExpr/IR/ZKExprOps.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;

namespace llzk {
#define GEN_PASS_DEF_CONVERTLLZKTOZKLEANPASS
#include "llzk/Conversions/LLZKConversionPasses.h.inc"
} // namespace llzk

namespace {

static LogicalResult convertModule(ModuleOp module) {
  OpBuilder builder(module.getContext());
  auto zkType = mlir::zkexpr::ZKExprType::get(module.getContext());

  module.walk([&](llzk::function::FuncDefOp func) {
    if (func.getBody().empty())
      return;
    if (!func->hasAttr("function.allow_constraint"))
      return;

    Block &oldBlock = func.getBody().front();
    SmallVector<Operation *, 16> ops;
    for (Operation &op : oldBlock)
      ops.push_back(&op);

    auto *newBlock = new Block();
    DenseMap<Value, Value> zkValues;
    DenseMap<Value, Value> argMapping;

    for (auto [idx, oldArg] : llvm::enumerate(oldBlock.getArguments())) {
      auto newArg = newBlock->addArgument(oldArg.getType(), oldArg.getLoc());
      argMapping[oldArg] = newArg;
    }

    auto mapValue = [&](Value v) -> Value {
      if (auto it = zkValues.find(v); it != zkValues.end())
        return it->second;

      auto newArg = argMapping.lookup(v);
      if (!newArg)
        return Value();

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(newBlock);
      auto literal = builder.create<mlir::zkexpr::LiteralOp>(v.getLoc(), zkType,
                                                             newArg);
      zkValues[v] = literal.getOutput();
      return literal.getOutput();
    };

    for (Operation *op : ops) {
      if (auto constOp = dyn_cast<llzk::felt::FeltConstantOp>(op)) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        auto newConst = builder.create<llzk::felt::FeltConstantOp>(
            constOp.getLoc(), constOp.getResult().getType(),
            constOp.getValueAttr());
        auto literal =
            builder.create<mlir::zkexpr::LiteralOp>(constOp.getLoc(), zkType,
                                                    newConst.getResult());
        zkValues[constOp.getResult()] = literal.getOutput();
        continue;
      }

      if (auto add = dyn_cast<llzk::felt::AddFeltOp>(op)) {
        Value lhs = mapValue(add.getLhs());
        Value rhs = mapValue(add.getRhs());
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        auto zkAdd =
            builder.create<mlir::zkexpr::AddOp>(add.getLoc(), lhs, rhs);
        zkValues[add.getResult()] = zkAdd.getOutput();
        continue;
      }

      if (auto mul = dyn_cast<llzk::felt::MulFeltOp>(op)) {
        Value lhs = mapValue(mul.getLhs());
        Value rhs = mapValue(mul.getRhs());
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        auto zkMul =
            builder.create<mlir::zkexpr::MulOp>(mul.getLoc(), lhs, rhs);
        zkValues[mul.getResult()] = zkMul.getOutput();
        continue;
      }

      if (auto neg = dyn_cast<llzk::felt::NegFeltOp>(op)) {
        Value operand = mapValue(neg.getOperand());
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        auto zkNeg =
            builder.create<mlir::zkexpr::NegOp>(neg.getLoc(), operand);
        zkValues[neg.getResult()] = zkNeg.getOutput();
        continue;
      }

      if (auto read = dyn_cast<llzk::component::FieldReadOp>(op)) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        auto witness = builder.create<mlir::zkexpr::WitnessOp>(read.getLoc());
        zkValues[read.getResult()] = witness.getOutput();
        continue;
      }

      if (auto eq = dyn_cast<llzk::constrain::EmitEqualityOp>(op)) {
        Value lhs = mapValue(eq.getLhs());
        Value rhs = mapValue(eq.getRhs());
        auto stateType =
            mlir::zkbuilder::ZKBuilderStateType::get(module.getContext());
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        builder.create<mlir::zkbuilder::ConstrainEqOp>(eq.getLoc(), stateType,
                                                       lhs, rhs);
        continue;
      }
    }

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(newBlock);
    builder.create<llzk::function::ReturnOp>(func.getLoc());

    while (!func.getBody().empty())
      func.getBody().front().erase();
    func.getBody().push_back(newBlock);
  });

  return success();
}

class ConvertLLZKToZKLeanPass
    : public llzk::impl::ConvertLLZKToZKLeanPassBase<
          ConvertLLZKToZKLeanPass> {
public:
  void runOnOperation() override {
    ModuleOp original = getOperation();
    ModuleOp zkLeanClone = original.clone();
    auto symName = StringAttr::get(&getContext(), "ZKLean");
    zkLeanClone->setAttr(SymbolTable::getSymbolAttrName(), symName);
    if (failed(convertModule(zkLeanClone))) {
      original.emitError("failed to produce ZKLean module");
      signalPassFailure();
      return;
    }
    original.getBody()->push_back(zkLeanClone.getOperation());
  }
};

} // namespace

namespace llzk {

std::unique_ptr<Pass> createConvertLLZKToZKLeanPass() {
  return std::make_unique<ConvertLLZKToZKLeanPass>();
}

void registerConversionPasses() {
  ::mlir::registerPass([] { return createConvertLLZKToZKLeanPass(); });
}

} // namespace llzk
