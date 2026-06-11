//===-- LLZKVerifToSmtPass.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-verif-to-smt` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Bool/IR/Enums.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/SMT/IR/SMTDialect.h"
#include "llzk/Dialect/SMT/IR/SMTOps.h"
#include "llzk/Dialect/SMT/IR/SMTTypes.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "llzk/Dialect/Verif/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/SymbolTable.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>

#include <memory>
#include <optional>
#include <string>

namespace llzk {
#define GEN_PASS_DECL_VERIFTOSMTPASS
#define GEN_PASS_DEF_VERIFTOSMTPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk::component;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::verif;

namespace {

struct TargetHelperNames {
  std::string compute;
  std::string constrain;
};

struct ContractHelperNames {
  std::string pre;
  std::string post;
  std::optional<std::string> includes;
};

struct LoweredSignature {
  SmallVector<Type> argTypes;
  SmallVector<Type> resultTypes;
};

struct LoweringContext {
  MLIRContext *context = nullptr;
  SymbolTableCollection tables {};
  DenseMap<Operation *, func::FuncOp> functionHelpers {};
  DenseMap<Operation *, TargetHelperNames> structHelpers {};
  DenseMap<Operation *, ContractHelperNames> contractHelpers {};
};

static Type lowerType(MLIRContext *context, Type type) {
  if (isa<FeltType>(type)) {
    return llzk::smt::IntType::get(context);
  }
  if (auto intType = dyn_cast<IntegerType>(type);
      intType && intType.isSignless() && intType.getWidth() == 1) {
    return llzk::smt::BoolType::get(context);
  }
  return type;
}

static FailureOr<StructDefOp>
resolveStructDef(LoweringContext &state, StructType type, Operation *origin) {
  return llzk::verifyStructTypeResolution(state.tables, type, origin);
}

static FailureOr<SmallVector<MemberDefOp>>
getStructMembers(LoweringContext &state, StructType type, Operation *origin) {
  auto structDef = resolveStructDef(state, type, origin);
  if (failed(structDef)) {
    return failure();
  }
  SmallVector<MemberDefOp> members;
  for (MemberDefOp member : structDef->getMemberDefs()) {
    members.push_back(member);
  }
  return members;
}

static FailureOr<LoweredSignature>
lowerCallableSignature(LoweringContext &state, FunctionType type, Operation *origin) {
  LoweredSignature lowered;
  for (Type input : type.getInputs()) {
    if (auto structType = dyn_cast<StructType>(input)) {
      auto members = getStructMembers(state, structType, origin);
      if (failed(members)) {
        return failure();
      }
      for (MemberDefOp member : *members) {
        lowered.argTypes.push_back(lowerType(state.context, member.getType()));
      }
      continue;
    }
    lowered.argTypes.push_back(lowerType(state.context, input));
  }
  for (Type result : type.getResults()) {
    lowered.resultTypes.push_back(lowerType(state.context, result));
  }
  return lowered;
}

static Value buildConjunction(OpBuilder &builder, Location loc, ArrayRef<Value> conditions) {
  if (conditions.empty()) {
    return builder.create<llzk::smt::BoolConstantOp>(loc, builder.getBoolAttr(true)).getResult();
  }

  Value current = conditions.front();
  for (Value condition : conditions.drop_front()) {
    current = builder.create<llzk::smt::AndOp>(loc, current, condition).getResult();
  }
  return current;
}

class ExprLowerer {
public:
  ExprLowerer(
      LoweringContext &state, OpBuilder &builder, DenseMap<Value, Value> &valueMap,
      DenseMap<StringRef, Value> &selfMemberMap
  )
      : state(state), builder(builder), valueMap(valueMap), selfMemberMap(selfMemberMap) {}

  FailureOr<Value> lower(Value value) {
    if (auto it = valueMap.find(value); it != valueMap.end()) {
      return it->second;
    }

    Operation *definingOp = value.getDefiningOp();
    if (definingOp == nullptr) {
      emitError(value.getLoc(), "missing lowered block argument mapping");
      return failure();
    }

    if (auto feltConst = dyn_cast<FeltConstantOp>(definingOp)) {
      auto lowered = builder.create<llzk::smt::IntConstantOp>(
          feltConst.getLoc(),
          IntegerAttr::get(builder.getContext(), APSInt(feltConst.getValue().getValue()))
      );
      valueMap[value] = lowered.getResult();
      return lowered.getResult();
    }

    if (auto constOp = dyn_cast<arith::ConstantOp>(definingOp)) {
      if (auto boolAttr = dyn_cast<BoolAttr>(constOp.getValue())) {
        auto lowered =
            builder.create<llzk::smt::BoolConstantOp>(constOp.getLoc(), boolAttr).getResult();
        valueMap[value] = lowered;
        return lowered;
      }
      auto lowered = builder.create<arith::ConstantOp>(constOp.getLoc(), constOp.getValue());
      valueMap[value] = lowered.getResult();
      return lowered.getResult();
    }

    if (auto memberRead = dyn_cast<MemberReadOp>(definingOp)) {
      if (memberRead.getComponent() != nullptr &&
          memberRead.getComponent().getDefiningOp() == nullptr) {
        auto it = selfMemberMap.find(memberRead.getMemberName());
        if (it != selfMemberMap.end()) {
          valueMap[value] = it->second;
          return it->second;
        }
      }
    }

    if (auto boolCmp = dyn_cast<llzk::boolean::CmpOp>(definingOp)) {
      auto lhs = lower(boolCmp.getLhs());
      auto rhs = lower(boolCmp.getRhs());
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }

      Value cmp = [&]() -> Value {
        using llzk::boolean::FeltCmpPredicate;
        switch (boolCmp.getPredicate()) {
        case FeltCmpPredicate::EQ:
          return builder.create<llzk::smt::EqOp>(boolCmp.getLoc(), *lhs, *rhs).getResult();
        case FeltCmpPredicate::NE: {
          Value eq = builder.create<llzk::smt::EqOp>(boolCmp.getLoc(), *lhs, *rhs).getResult();
          return builder.create<llzk::smt::NotOp>(boolCmp.getLoc(), eq).getResult();
        }
        case FeltCmpPredicate::LT:
          return builder
              .create<llzk::smt::IntCmpOp>(
                  boolCmp.getLoc(), llzk::smt::IntPredicate::lt, *lhs, *rhs
              )
              .getResult();
        case FeltCmpPredicate::LE:
          return builder
              .create<llzk::smt::IntCmpOp>(
                  boolCmp.getLoc(), llzk::smt::IntPredicate::le, *lhs, *rhs
              )
              .getResult();
        case FeltCmpPredicate::GT:
          return builder
              .create<llzk::smt::IntCmpOp>(
                  boolCmp.getLoc(), llzk::smt::IntPredicate::gt, *lhs, *rhs
              )
              .getResult();
        case FeltCmpPredicate::GE:
          return builder
              .create<llzk::smt::IntCmpOp>(
                  boolCmp.getLoc(), llzk::smt::IntPredicate::ge, *lhs, *rhs
              )
              .getResult();
        }
        llvm_unreachable("unknown bool.cmp predicate");
      }();

      valueMap[value] = cmp;
      return cmp;
    }

    if (auto call = dyn_cast<CallOp>(definingOp)) {
      auto calleeTarget = call.getCalleeTarget(state.tables);
      if (failed(calleeTarget)) {
        return failure();
      }

      auto it = state.functionHelpers.find(calleeTarget->get().getOperation());
      if (it == state.functionHelpers.end()) {
        call.emitError("verif-to-smt requires a lowered helper for called function");
        return failure();
      }

      SmallVector<Value> loweredOperands;
      loweredOperands.reserve(call.getNumOperands());
      for (Value operand : call.getOperands()) {
        auto loweredOperand = lower(operand);
        if (failed(loweredOperand)) {
          return failure();
        }
        loweredOperands.push_back(*loweredOperand);
      }

      auto loweredCall = builder.create<func::CallOp>(
          call.getLoc(), it->second.getSymName(), it->second.getFunctionType().getResults(),
          loweredOperands
      );
      for (auto [orig, lowered] : llvm::zip(call.getResults(), loweredCall.getResults())) {
        valueMap[orig] = lowered;
      }
      return loweredCall.getResult(mlir::cast<OpResult>(value).getResultNumber());
    }

    definingOp->emitError("unsupported expression in verif-to-smt lowering");
    return failure();
  }

private:
  LoweringContext &state;
  OpBuilder &builder;
  DenseMap<Value, Value> &valueMap;
  DenseMap<StringRef, Value> &selfMemberMap;
};

static FailureOr<func::FuncOp>
getOrCreateFunctionHelper(LoweringContext &state, ModuleOp module, FuncDefOp func) {
  if (auto it = state.functionHelpers.find(func.getOperation());
      it != state.functionHelpers.end()) {
    return it->second;
  }

  auto loweredSig = lowerCallableSignature(state, func.getFunctionType(), func);
  if (failed(loweredSig)) {
    return failure();
  }

  OpBuilder moduleBuilder(module.getBodyRegion());
  // modules don't have a terminator, so just insert at the back
  moduleBuilder.setInsertionPoint(&module.getBody()->back());
  std::string helperName = ("smt_" + func.getSymName()).str();
  auto helper = moduleBuilder.create<func::FuncOp>(
      func.getLoc(), helperName,
      FunctionType::get(state.context, loweredSig->argTypes, loweredSig->resultTypes)
  );

  Block *entry = helper.addEntryBlock();
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
  DenseMap<Value, Value> valueMap;
  DenseMap<StringRef, Value> selfMemberMap;
  for (auto [original, lowered] : llvm::zip(func.getArguments(), entry->getArguments())) {
    valueMap[original] = lowered;
  }

  ExprLowerer lowerer(state, bodyBuilder, valueMap, selfMemberMap);
  auto returnOp = dyn_cast<ReturnOp>(func.getBody().front().getTerminator());
  if (!returnOp) {
    func.emitError("expected function.return terminator");
    return failure();
  }

  SmallVector<Value> returnedValues;
  for (Value operand : returnOp.getOperands()) {
    auto loweredValue = lowerer.lower(operand);
    if (failed(loweredValue)) {
      return failure();
    }
    returnedValues.push_back(*loweredValue);
  }
  bodyBuilder.create<func::ReturnOp>(returnOp.getLoc(), returnedValues);

  state.functionHelpers[func.getOperation()] = helper;
  return helper;
}

static FailureOr<TargetHelperNames>
getOrCreateStructHelpers(LoweringContext &state, ModuleOp module, StructDefOp structDef) {
  if (auto it = state.structHelpers.find(structDef.getOperation());
      it != state.structHelpers.end()) {
    return it->second;
  }

  FuncDefOp computeFunc = structDef.getComputeFuncOp();
  FuncDefOp constrainFunc = structDef.getConstrainFuncOp();
  if (!computeFunc || !constrainFunc) {
    structDef.emitError("verif-to-smt requires both @compute and @constrain");
    return failure();
  }

  SmallVector<MemberDefOp> members;
  for (MemberDefOp member : structDef.getMemberDefs()) {
    members.push_back(member);
  }

  SmallVector<Type> memberTypes;
  memberTypes.reserve(members.size());
  for (MemberDefOp member : members) {
    memberTypes.push_back(lowerType(state.context, member.getType()));
  }

  auto computeSig = lowerCallableSignature(state, computeFunc.getFunctionType(), computeFunc);
  if (failed(computeSig)) {
    return failure();
  }

  OpBuilder moduleBuilder(module.getBodyRegion());
  // modules don't have a terminator, so just insert at the back
  moduleBuilder.setInsertionPoint(&module.getBody()->back());

  std::string computeName = ("smt_" + structDef.getSymName() + "_compute").str();
  auto computeHelper = moduleBuilder.create<func::FuncOp>(
      computeFunc.getLoc(), computeName,
      FunctionType::get(state.context, computeSig->argTypes, memberTypes)
  );
  {
    Block *entry = computeHelper.addEntryBlock();
    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
    DenseMap<Value, Value> valueMap;
    DenseMap<StringRef, Value> selfMemberMap;
    for (auto [original, lowered] : llvm::zip(computeFunc.getArguments(), entry->getArguments())) {
      valueMap[original] = lowered;
    }

    ExprLowerer lowerer(state, bodyBuilder, valueMap, selfMemberMap);
    DenseMap<StringRef, Value> writtenMembers;
    computeFunc.walk([&](MemberWriteOp writeOp) {
      auto loweredValue = lowerer.lower(writeOp.getVal());
      if (succeeded(loweredValue)) {
        writtenMembers[writeOp.getMemberName()] = *loweredValue;
      }
    });

    SmallVector<Value> results;
    for (MemberDefOp member : members) {
      auto it = writtenMembers.find(member.getSymName());
      if (it == writtenMembers.end()) {
        computeFunc.emitError().append("missing write for member @", member.getSymName());
        return failure();
      }
      results.push_back(it->second);
    }
    bodyBuilder.create<func::ReturnOp>(computeFunc.getLoc(), results);
  }

  std::string constrainName = ("smt_" + structDef.getSymName() + "_constrain").str();
  SmallVector<Type> constrainArgs = memberTypes;
  for (Type input : constrainFunc.getArgumentTypes().drop_front()) {
    constrainArgs.push_back(lowerType(state.context, input));
  }
  auto constrainHelper = moduleBuilder.create<func::FuncOp>(
      constrainFunc.getLoc(), constrainName,
      FunctionType::get(state.context, constrainArgs, memberTypes)
  );
  {
    Block *entry = constrainHelper.addEntryBlock();
    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
    bodyBuilder.create<func::ReturnOp>(
        constrainFunc.getLoc(), ValueRange(entry->getArguments()).take_front(memberTypes.size())
    );
  }

  TargetHelperNames helperNames {computeName, constrainName};
  state.structHelpers[structDef.getOperation()] = helperNames;
  return helperNames;
}

static FailureOr<LoweredSignature>
lowerContractSignature(LoweringContext &state, ContractOp contract) {
  return lowerCallableSignature(state, contract.getFunctionType(), contract);
}

static void seedContractArgumentMaps(
    LoweringContext &state, ContractOp contract, Block *entry, DenseMap<Value, Value> &valueMap,
    DenseMap<StringRef, Value> &selfMemberMap
) {
  unsigned nextArg = 0;
  for (BlockArgument originalArg : contract.getArguments()) {
    if (auto structType = dyn_cast<StructType>(originalArg.getType())) {
      auto members = *getStructMembers(state, structType, contract);
      for (MemberDefOp member : members) {
        selfMemberMap[member.getSymName()] = entry->getArgument(nextArg++);
      }
      continue;
    }
    valueMap[originalArg] = entry->getArgument(nextArg++);
  }
}

static FailureOr<func::FuncOp> createContractConditionHelper(
    LoweringContext &state, ModuleOp module, ContractOp contract, StringRef helperName,
    llvm::function_ref<bool(Operation *)> predicate
) {
  auto loweredSig = lowerContractSignature(state, contract);
  if (failed(loweredSig)) {
    return failure();
  }

  SmallVector<Type> resultTypes {llzk::smt::BoolType::get(state.context)};
  OpBuilder moduleBuilder(module.getBodyRegion());
  // modules don't have a terminator, so just insert at the back
  moduleBuilder.setInsertionPoint(&module.getBody()->back());
  auto helper = moduleBuilder.create<func::FuncOp>(
      contract.getLoc(), helperName,
      FunctionType::get(state.context, loweredSig->argTypes, resultTypes)
  );

  Block *entry = helper.addEntryBlock();
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
  DenseMap<Value, Value> valueMap;
  DenseMap<StringRef, Value> selfMemberMap;
  seedContractArgumentMaps(state, contract, entry, valueMap, selfMemberMap);
  ExprLowerer lowerer(state, bodyBuilder, valueMap, selfMemberMap);

  SmallVector<Value> conditions;
  contract.walk([&](Operation *op) {
    if (!predicate(op)) {
      return;
    }

    auto requireOp = dyn_cast<ConditionOpInterface>(op);
    auto lowered = lowerer.lower(requireOp.getCondition());
    if (succeeded(lowered)) {
      conditions.push_back(*lowered);
    }
  });

  Value combined = buildConjunction(bodyBuilder, contract.getLoc(), conditions);
  bodyBuilder.create<func::ReturnOp>(contract.getLoc(), combined);
  return helper;
}

static FailureOr<ContractHelperNames>
getOrCreateContractHelpers(LoweringContext &state, ModuleOp module, ContractOp contract) {
  if (auto it = state.contractHelpers.find(contract.getOperation());
      it != state.contractHelpers.end()) {
    return it->second;
  }

  std::string prefix = ("smt_verif_" + contract.getSymName()).str();
  auto preHelper =
      createContractConditionHelper(state, module, contract, prefix + "_pre", [](Operation *op) {
    return isa<RequireComputeOp, RequireConstrainOp>(op);
  });
  if (failed(preHelper)) {
    return failure();
  }

  auto postHelper =
      createContractConditionHelper(state, module, contract, prefix + "_post", [](Operation *op) {
    return isa<EnsureComputeOp, EnsureConstrainOp>(op);
  });
  if (failed(postHelper)) {
    return failure();
  }

  ContractHelperNames names {prefix + "_pre", prefix + "_post", std::nullopt};
  state.contractHelpers[contract.getOperation()] = names;
  return names;
}

static FailureOr<SmallVector<Value>> lowerIncludeHelperOperands(
    LoweringContext &state, ContractOp owningContract, ValueRange operands, OpBuilder &builder,
    DenseMap<Value, Value> &valueMap, DenseMap<StringRef, Value> &selfMemberMap
) {
  ExprLowerer lowerer(state, builder, valueMap, selfMemberMap);
  SmallVector<Value> loweredOperands;
  loweredOperands.reserve(operands.size());
  for (Value operand : operands) {
    auto lowered = lowerer.lower(operand);
    if (failed(lowered)) {
      return failure();
    }
    loweredOperands.push_back(*lowered);
  }
  return loweredOperands;
}

static LogicalResult
maybeCreateIncludeHelper(LoweringContext &state, ModuleOp module, ContractOp contract) {
  SmallVector<IncludeOp> includes;
  contract.walk([&](IncludeOp includeOp) { includes.push_back(includeOp); });
  if (includes.empty()) {
    return success();
  }

  auto loweredSig = lowerContractSignature(state, contract);
  if (failed(loweredSig)) {
    return failure();
  }

  std::string helperName = ("smt_verif_" + contract.getSymName() + "_includes").str();
  OpBuilder moduleBuilder(module.getBodyRegion());
  // modules don't have a terminator, so just insert at the back
  moduleBuilder.setInsertionPoint(&module.getBody()->back());
  auto helper = moduleBuilder.create<func::FuncOp>(
      contract.getLoc(), helperName,
      FunctionType::get(state.context, loweredSig->argTypes, TypeRange {})
  );

  Block *entry = helper.addEntryBlock();
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
  DenseMap<Value, Value> valueMap;
  DenseMap<StringRef, Value> selfMemberMap;
  seedContractArgumentMaps(state, contract, entry, valueMap, selfMemberMap);

  for (IncludeOp includeOp : includes) {
    auto calleeTarget = includeOp.getCalleeTarget(state.tables);
    if (failed(calleeTarget)) {
      return failure();
    }
    ContractOp callee = calleeTarget->get();
    auto calleeHelpers = getOrCreateContractHelpers(state, module, callee);
    if (failed(calleeHelpers)) {
      return failure();
    }

    auto loweredOperands = lowerIncludeHelperOperands(
        state, contract, includeOp.getArgOperands(), bodyBuilder, valueMap, selfMemberMap
    );
    if (failed(loweredOperands)) {
      return failure();
    }

    SmallVector<Value> helperArgs = *loweredOperands;
    if (!callee.hasStructTarget()) {
      FailureOr<llzk::SymbolLookupResult<FuncDefOp>> calleeTargetFunc =
          llzk::lookupSymbolIn<FuncDefOp>(
              state.tables, callee.getTarget(), llzk::Within(module.getOperation()), includeOp,
              /*reportMissing=*/true
          );
      if (failed(calleeTargetFunc)) {
        return failure();
      }
      auto targetHelper = getOrCreateFunctionHelper(state, module, calleeTargetFunc->get());
      if (failed(targetHelper)) {
        return failure();
      }
      auto targetCall = bodyBuilder.create<func::CallOp>(
          includeOp.getLoc(), targetHelper->getSymName(),
          targetHelper->getFunctionType().getResults(), *loweredOperands
      );
      llvm::append_range(helperArgs, targetCall.getResults());
    }

    auto preCall = bodyBuilder.create<func::CallOp>(
        includeOp.getLoc(), calleeHelpers->pre, TypeRange {llzk::smt::BoolType::get(state.context)},
        helperArgs
    );
    auto postCall = bodyBuilder.create<func::CallOp>(
        includeOp.getLoc(), calleeHelpers->post,
        TypeRange {llzk::smt::BoolType::get(state.context)}, helperArgs
    );
    Value implication = bodyBuilder.create<llzk::smt::ImpliesOp>(
        includeOp.getLoc(), preCall.getResult(0), postCall.getResult(0)
    );
    bodyBuilder.create<llzk::smt::AssertOp>(includeOp.getLoc(), implication);
  }

  bodyBuilder.create<func::ReturnOp>(contract.getLoc());
  state.contractHelpers[contract.getOperation()].includes = helperName;
  return success();
}

struct VerifToSmtPass : public llzk::impl::VerifToSmtPassBase<VerifToSmtPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, llzk::smt::SMTDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    LoweringContext state {.context = &getContext()};

    SmallVector<ContractOp> contracts;
    module.walk([&](ContractOp contract) { contracts.push_back(contract); });

    for (ContractOp contract : contracts) {
      if (contract.hasStructTarget()) {
        auto structTarget = contract.getStructTarget(state.tables);
        if (failed(structTarget) ||
            failed(getOrCreateStructHelpers(state, module, structTarget->get()))) {
          signalPassFailure();
          return;
        }
      } else {
        auto targetFunc = llzk::lookupSymbolIn<FuncDefOp>(
            state.tables, contract.getTarget(), llzk::Within(module.getOperation()), contract, true
        );
        if (failed(targetFunc) ||
            failed(getOrCreateFunctionHelper(state, module, targetFunc->get()))) {
          signalPassFailure();
          return;
        }
      }

      if (failed(getOrCreateContractHelpers(state, module, contract)) ||
          failed(maybeCreateIncludeHelper(state, module, contract))) {
        signalPassFailure();
        return;
      }
    }

    for (ContractOp contract : contracts) {
      contract.erase();
    }
  }
};

} // namespace

namespace llzk {

std::unique_ptr<Pass> createVerifToSmtPass() { return std::make_unique<VerifToSmtPass>(); }

} // namespace llzk
