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
/// This file implements the `-llzk-scalar-verif-to-smt` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Bool/IR/Enums.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
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
#include <llvm/ADT/DenseSet.h>
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
  std::string target;
  std::string post;
  std::string entry;
  SmallVector<std::string> includeHelpers;
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

static FailureOr<SmallVector<MemberDefOp>>
getStructMembers(LoweringContext &state, StructType type, Operation *origin);

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

static bool typeContainsUnsupportedAggregate(
    LoweringContext &state, Type type, Operation *origin, llvm::DenseSet<Type> &visited
) {
  if (!visited.insert(type).second) {
    return false;
  }
  if (isa<llzk::array::ArrayType, llzk::pod::PodType>(type)) {
    return true;
  }
  if (auto structType = dyn_cast<StructType>(type)) {
    auto members = getStructMembers(state, structType, origin);
    if (failed(members)) {
      return true;
    }
    for (MemberDefOp member : *members) {
      if (typeContainsUnsupportedAggregate(state, member.getType(), origin, visited)) {
        return true;
      }
    }
  }
  return false;
}

static bool typeContainsUnsupportedAggregate(LoweringContext &state, Type type, Operation *origin) {
  llvm::DenseSet<Type> visited;
  return typeContainsUnsupportedAggregate(state, type, origin, visited);
}

static LogicalResult ensureScalarTypeSupported(
    LoweringContext &state, Type type, Operation *origin, StringRef description
) {
  if (!typeContainsUnsupportedAggregate(state, type, origin)) {
    return success();
  }
  origin->emitError() << "llzk-scalar-verif-to-smt requires array/pod-free IR for " << description
                      << "; run -llzk-verif-to-smt instead";
  return failure();
}

static LogicalResult ensureScalarSignatureSupported(
    LoweringContext &state, FunctionType type, Operation *origin, StringRef description
) {
  for (Type input : type.getInputs()) {
    if (failed(ensureScalarTypeSupported(state, input, origin, description))) {
      return failure();
    }
  }
  for (Type result : type.getResults()) {
    if (failed(ensureScalarTypeSupported(state, result, origin, description))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult ensureScalarContractSupported(LoweringContext &state, ContractOp contract) {
  if (failed(ensureScalarSignatureSupported(
          state, contract.getFunctionType(), contract, "contract signature"
      ))) {
    return failure();
  }

  if (contract.hasStructTarget()) {
    auto structTarget = contract.getStructTarget(state.tables);
    if (failed(structTarget)) {
      return failure();
    }
    for (MemberDefOp member : structTarget->get().getMemberDefs()) {
      if (failed(ensureScalarTypeSupported(
              state, member.getType(), contract, "struct target member types"
          ))) {
        return failure();
      }
    }
    return success();
  }

  auto targetFunc = llzk::lookupSymbolIn<FuncDefOp>(
      state.tables, contract.getTarget(), llzk::Within(contract->getParentOp()), contract, true
  );
  if (failed(targetFunc)) {
    return failure();
  }
  return ensureScalarSignatureSupported(
      state, targetFunc->get().getFunctionType(), contract, "target function signature"
  );
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

static SmallVector<Value>
buildEqualityConditions(OpBuilder &builder, Location loc, ValueRange lhs, ValueRange rhs) {
  SmallVector<Value> conditions;
  for (auto [lhsValue, rhsValue] : llvm::zip(lhs, rhs)) {
    conditions.push_back(builder.create<llzk::smt::EqOp>(loc, lhsValue, rhsValue).getResult());
  }
  return conditions;
}

static void populateVoidCheckRegion(
    Region &region, Location loc, llvm::function_ref<void(OpBuilder &)> buildBody
) {
  Block *block = new Block();
  region.push_back(block);
  OpBuilder builder = OpBuilder::atBlockBegin(block);
  buildBody(builder);
  builder.create<llzk::smt::YieldOp>(loc);
}

static void proveByUnsatAndAssert(
    OpBuilder &builder, Location loc, Value condition, StringRef contractName, StringRef stageName
) {
  builder.create<llzk::smt::PushOp>(loc, 1);
  Value negated = builder.create<llzk::smt::NotOp>(loc, condition).getResult();
  builder.create<llzk::smt::AssertOp>(loc, negated);
  auto check = builder.create<llzk::smt::CheckOp>(loc, TypeRange {});

  auto makeFailureMessage = [&](StringRef outcome) {
    return builder.getStringAttr(
        (Twine("verification failed in ") + contractName + " " + stageName + ": " + outcome).str()
    );
  };

  populateVoidCheckRegion(check.getSatRegion(), loc, [&](OpBuilder &regionBuilder) {
    regionBuilder.create<llzk::smt::PopOp>(loc, 1);
    Value failed = regionBuilder.create<arith::ConstantOp>(loc, regionBuilder.getBoolAttr(false));
    regionBuilder.create<llzk::boolean::AssertOp>(
        loc, failed, makeFailureMessage("counterexample found")
    );
  });
  populateVoidCheckRegion(check.getUnknownRegion(), loc, [&](OpBuilder &regionBuilder) {
    regionBuilder.create<llzk::smt::PopOp>(loc, 1);
    Value failed = regionBuilder.create<arith::ConstantOp>(loc, regionBuilder.getBoolAttr(false));
    regionBuilder.create<llzk::boolean::AssertOp>(
        loc, failed, makeFailureMessage("condition unprovable")
    );
  });
  populateVoidCheckRegion(check.getUnsatRegion(), loc, [&](OpBuilder &regionBuilder) {
    regionBuilder.create<llzk::smt::PopOp>(loc, 1);
    regionBuilder.create<llzk::smt::AssertOp>(loc, condition);
  });
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

static bool contractUsesComputeTarget(ContractOp contract) {
  bool usesCompute = false;
  contract.walk([&](Operation *op) {
    if (isa<RequireComputeOp, EnsureComputeOp>(op)) {
      usesCompute = true;
    }
  });
  return usesCompute;
}

static bool contractUsesConstrainTarget(ContractOp contract) {
  bool usesConstrain = false;
  contract.walk([&](Operation *op) {
    if (isa<RequireConstrainOp, EnsureConstrainOp>(op)) {
      usesConstrain = true;
    }
  });
  return usesConstrain;
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

static FailureOr<func::FuncOp>
createFreeFunctionTargetHelper(LoweringContext &state, ModuleOp module, ContractOp contract) {
  auto loweredSig = lowerContractSignature(state, contract);
  if (failed(loweredSig)) {
    return failure();
  }

  auto targetFunc = llzk::lookupSymbolIn<FuncDefOp>(
      state.tables, contract.getTarget(), llzk::Within(module.getOperation()), contract,
      /*reportMissing=*/true
  );
  if (failed(targetFunc)) {
    return failure();
  }

  auto rawTargetHelper = getOrCreateFunctionHelper(state, module, targetFunc->get());
  if (failed(rawTargetHelper)) {
    return failure();
  }

  auto targetLoweredSig =
      lowerCallableSignature(state, targetFunc->get().getFunctionType(), targetFunc->get());
  if (failed(targetLoweredSig)) {
    return failure();
  }

  std::string helperName = ("smt_verif_" + contract.getSymName() + "_target").str();
  OpBuilder moduleBuilder(module.getBodyRegion());
  moduleBuilder.setInsertionPoint(&module.getBody()->back());
  auto helper = moduleBuilder.create<func::FuncOp>(
      contract.getLoc(), helperName,
      FunctionType::get(
          state.context, loweredSig->argTypes, TypeRange {llzk::smt::BoolType::get(state.context)}
      )
  );

  Block *entry = helper.addEntryBlock();
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);

  ValueRange args = entry->getArguments();
  ValueRange targetInputs = args.take_front(targetLoweredSig->argTypes.size());
  auto targetCall = bodyBuilder.create<func::CallOp>(
      contract.getLoc(), rawTargetHelper->getSymName(),
      rawTargetHelper->getFunctionType().getResults(), targetInputs
  );

  ValueRange expectedResults = args.drop_front(targetLoweredSig->argTypes.size())
                                   .take_front(targetLoweredSig->resultTypes.size());
  SmallVector<Value> conditions = buildEqualityConditions(
      bodyBuilder, contract.getLoc(), targetCall.getResults(), expectedResults
  );
  Value combined = buildConjunction(bodyBuilder, contract.getLoc(), conditions);
  bodyBuilder.create<func::ReturnOp>(contract.getLoc(), combined);
  return helper;
}

static FailureOr<func::FuncOp>
createStructTargetHelper(LoweringContext &state, ModuleOp module, ContractOp contract) {
  auto loweredSig = lowerContractSignature(state, contract);
  if (failed(loweredSig)) {
    return failure();
  }

  auto structTarget = contract.getStructTarget(state.tables);
  if (failed(structTarget)) {
    return failure();
  }
  auto targetHelpers = getOrCreateStructHelpers(state, module, structTarget->get());
  if (failed(targetHelpers)) {
    return failure();
  }

  auto members = getStructMembers(state, structTarget->get().getType(), contract);
  if (failed(members)) {
    return failure();
  }
  unsigned numFlattenedSelfMembers = members->size();

  bool useCompute = contractUsesComputeTarget(contract);
  bool useConstrain = contractUsesConstrainTarget(contract);

  std::string helperName = ("smt_verif_" + contract.getSymName() + "_target").str();
  OpBuilder moduleBuilder(module.getBodyRegion());
  moduleBuilder.setInsertionPoint(&module.getBody()->back());
  auto helper = moduleBuilder.create<func::FuncOp>(
      contract.getLoc(), helperName,
      FunctionType::get(
          state.context, loweredSig->argTypes, TypeRange {llzk::smt::BoolType::get(state.context)}
      )
  );

  Block *entry = helper.addEntryBlock();
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
  ValueRange args = entry->getArguments();
  ValueRange selfArgs = args.take_front(numFlattenedSelfMembers);
  ValueRange nonSelfArgs = args.drop_front(numFlattenedSelfMembers);

  SmallVector<Value> conditions;
  if (useCompute) {
    auto computeCall = bodyBuilder.create<func::CallOp>(
        contract.getLoc(), targetHelpers->compute, TypeRange(selfArgs.getTypes()), nonSelfArgs
    );
    SmallVector<Value> computeConditions =
        buildEqualityConditions(bodyBuilder, contract.getLoc(), computeCall.getResults(), selfArgs);
    llvm::append_range(conditions, computeConditions);
  }
  if (useConstrain) {
    auto constrainCall = bodyBuilder.create<func::CallOp>(
        contract.getLoc(), targetHelpers->constrain, TypeRange(selfArgs.getTypes()), args
    );
    SmallVector<Value> constrainConditions = buildEqualityConditions(
        bodyBuilder, contract.getLoc(), constrainCall.getResults(), selfArgs
    );
    llvm::append_range(conditions, constrainConditions);
  }

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
  std::string preName = prefix + "_pre";
  std::string targetName = prefix + "_target";
  std::string postName = prefix + "_post";
  std::string entryName = prefix + "_entry";

  SmallVector<IncludeOp> includes;
  contract.walk([&](IncludeOp includeOp) { includes.push_back(includeOp); });

  SmallVector<std::string> includeHelperNames;
  includeHelperNames.reserve(includes.size());
  for (auto [index, _] : llvm::enumerate(includes)) {
    includeHelperNames.push_back(prefix + "_include_" + std::to_string(index));
  }

  ContractHelperNames names {preName, targetName, postName, entryName, includeHelperNames};
  state.contractHelpers[contract.getOperation()] = names;

  auto preHelper =
      createContractConditionHelper(state, module, contract, preName, [](Operation *op) {
    return isa<RequireComputeOp, RequireConstrainOp>(op);
  });
  if (failed(preHelper)) {
    return failure();
  }

  auto postHelper =
      createContractConditionHelper(state, module, contract, postName, [](Operation *op) {
    return isa<EnsureComputeOp, EnsureConstrainOp>(op);
  });
  if (failed(postHelper)) {
    return failure();
  }

  FailureOr<func::FuncOp> targetHelper =
      contract.hasStructTarget() ? createStructTargetHelper(state, module, contract)
                                 : createFreeFunctionTargetHelper(state, module, contract);
  if (failed(targetHelper)) {
    return failure();
  }

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

static FailureOr<func::FuncOp> createIncludeWrapperHelper(
    LoweringContext &state, ModuleOp module, IncludeOp includeOp, StringRef helperName
) {
  auto calleeTarget = includeOp.getCalleeTarget(state.tables);
  if (failed(calleeTarget)) {
    return failure();
  }
  ContractOp callee = calleeTarget->get();
  auto calleeHelpers = getOrCreateContractHelpers(state, module, callee);
  if (failed(calleeHelpers)) {
    return failure();
  }

  auto loweredSig = lowerContractSignature(state, callee);
  if (failed(loweredSig)) {
    return failure();
  }

  OpBuilder moduleBuilder(module.getBodyRegion());
  moduleBuilder.setInsertionPoint(&module.getBody()->back());
  auto helper = moduleBuilder.create<func::FuncOp>(
      includeOp.getLoc(), helperName,
      FunctionType::get(state.context, loweredSig->argTypes, TypeRange {})
  );

  Block *entry = helper.addEntryBlock();
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
  bodyBuilder.create<func::CallOp>(
      includeOp.getLoc(), calleeHelpers->entry, TypeRange {}, entry->getArguments()
  );
  bodyBuilder.create<func::ReturnOp>(includeOp.getLoc());
  return helper;
}

static LogicalResult createContractEntryHelper(
    LoweringContext &state, ModuleOp module, ContractOp contract,
    SmallVectorImpl<IncludeOp> &includes
) {
  auto loweredSig = lowerContractSignature(state, contract);
  if (failed(loweredSig)) {
    return failure();
  }

  auto helperInfoIt = state.contractHelpers.find(contract.getOperation());
  if (helperInfoIt == state.contractHelpers.end()) {
    contract.emitError("missing contract helper names for entry helper generation");
    return failure();
  }
  const ContractHelperNames &helperInfo = helperInfoIt->second;

  OpBuilder moduleBuilder(module.getBodyRegion());
  moduleBuilder.setInsertionPoint(&module.getBody()->back());
  auto helper = moduleBuilder.create<func::FuncOp>(
      contract.getLoc(), helperInfo.entry,
      FunctionType::get(state.context, loweredSig->argTypes, TypeRange {})
  );

  Block *entry = helper.addEntryBlock();
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
  DenseMap<Value, Value> valueMap;
  DenseMap<StringRef, Value> selfMemberMap;
  seedContractArgumentMaps(state, contract, entry, valueMap, selfMemberMap);

  auto preCall = bodyBuilder.create<func::CallOp>(
      contract.getLoc(), helperInfo.pre, TypeRange {llzk::smt::BoolType::get(state.context)},
      entry->getArguments()
  );
  proveByUnsatAndAssert(
      bodyBuilder, contract.getLoc(), preCall.getResult(0), contract.getSymName(), "pre"
  );

  auto targetCall = bodyBuilder.create<func::CallOp>(
      contract.getLoc(), helperInfo.target, TypeRange {llzk::smt::BoolType::get(state.context)},
      entry->getArguments()
  );
  proveByUnsatAndAssert(
      bodyBuilder, contract.getLoc(), targetCall.getResult(0), contract.getSymName(), "target"
  );

  for (auto [index, includeOp] : llvm::enumerate(includes)) {
    auto loweredOperands = lowerIncludeHelperOperands(
        state, contract, includeOp.getArgOperands(), bodyBuilder, valueMap, selfMemberMap
    );
    if (failed(loweredOperands)) {
      return failure();
    }
    bodyBuilder.create<func::CallOp>(
        includeOp.getLoc(), helperInfo.includeHelpers[index], TypeRange {}, *loweredOperands
    );
  }

  auto postCall = bodyBuilder.create<func::CallOp>(
      contract.getLoc(), helperInfo.post, TypeRange {llzk::smt::BoolType::get(state.context)},
      entry->getArguments()
  );
  proveByUnsatAndAssert(
      bodyBuilder, contract.getLoc(), postCall.getResult(0), contract.getSymName(), "post"
  );

  bodyBuilder.create<func::ReturnOp>(contract.getLoc());
  return success();
}

struct VerifToSmtPass : public llzk::impl::VerifToSmtPassBase<VerifToSmtPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        arith::ArithDialect, func::FuncDialect, llzk::boolean::BoolDialect, llzk::smt::SMTDialect>(
    );
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    LoweringContext state {.context = &getContext()};

    SmallVector<ContractOp> contracts;
    module.walk([&](ContractOp contract) { contracts.push_back(contract); });

    for (ContractOp contract : contracts) {
      if (failed(ensureScalarContractSupported(state, contract))) {
        signalPassFailure();
        return;
      }
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

      auto contractHelpers = getOrCreateContractHelpers(state, module, contract);
      if (failed(contractHelpers)) {
        signalPassFailure();
        return;
      }

      SmallVector<IncludeOp> includes;
      contract.walk([&](IncludeOp includeOp) { includes.push_back(includeOp); });
      for (auto [index, includeOp] : llvm::enumerate(includes)) {
        if (failed(createIncludeWrapperHelper(
                state, module, includeOp, contractHelpers->includeHelpers[index]
            ))) {
          signalPassFailure();
          return;
        }
      }
      if (failed(createContractEntryHelper(state, module, contract, includes))) {
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
