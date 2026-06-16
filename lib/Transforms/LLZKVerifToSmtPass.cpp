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
/// The pass expects IR that has already been scalarized so that no `array.type` or `pod.type`
/// appears in contract signatures, target signatures, or struct members. The aggregate-aware
/// `-llzk-verif-to-smt` pipeline is responsible for running aggregate scalarization before this
/// pass.
///
/// The lowering introduces several kinds of SMT-oriented helper functions:
///
/// 0. `@smt_<function>` helpers for free-function targets.
///
/// 1. `@smt_<Struct>_compute` and `@smt_<Struct>_constrain` helpers for struct targets.
///
/// 2. `@smt_verif_<Contract>_pre`, `_target`, and `_post` helpers that lower the direct contract
///    conditions into SMT booleans.
///
/// 3. `@smt_verif_<Contract>_include_<N>` wrappers, one per `verif.include`, that call the
///    callee contract entry helper transitively.
///
/// 4. `@smt_verif_<Contract>_entry`, which proves each direct contract stage by checking that the
///    negated condition is unsat before asserting the condition into the caller solver context.
///
/// The pass runs as a sequential module transformation. For each contract in source order it
/// validates the contract and its direct target, creates any missing target helpers, materializes
/// the contract-local helpers, creates any per-include wrappers, and finally emits the full entry
/// helper. Once every contract has been lowered successfully, the original `verif.contract`
/// operations are erased from the module.
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
#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>

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

/// Stores the helper symbol names created for a struct target.
struct TargetHelperNames {
  /// Symbol name of the lowered `@compute` helper.
  std::string compute;

  /// Symbol name of the lowered `@constrain` helper.
  std::string constrain;
};

/// Stores the helper symbol names created for one contract.
struct ContractHelperNames {
  /// Symbol name of the direct precondition helper.
  std::string pre;

  /// Symbol name of the target-equality helper.
  std::string target;

  /// Symbol name of the direct postcondition helper.
  std::string post;

  /// Symbol name of the full contract entry helper.
  std::string entry;

  /// Symbol names of the per-include wrapper helpers in source order.
  SmallVector<std::string> includeHelpers;
};

/// Represents the lowered SMT-facing argument and result types for a callable.
struct LoweredSignature {
  /// Lowered argument types.
  SmallVector<Type> argTypes;

  /// Lowered result types.
  SmallVector<Type> resultTypes;
};

/// Shared lowering state for one execution of the pass.
struct LoweringContext {
  /// Build and type context.
  MLIRContext *context = nullptr;

  /// Symbol tables used during target and contract resolution.
  SymbolTableCollection tables {};

  /// Cached free-function helpers keyed by the original `function.def`.
  DenseMap<Operation *, func::FuncOp> functionHelpers {};

  /// Cached struct target helpers keyed by the original `struct.def`.
  DenseMap<Operation *, TargetHelperNames> structHelpers {};

  /// Cached contract helper bundles keyed by the original `verif.contract`.
  DenseMap<Operation *, ContractHelperNames> contractHelpers {};
};

/// Return the struct members for a resolved struct type.
static FailureOr<SmallVector<MemberDefOp>>
getStructMembers(LoweringContext &state, StructType type, Operation *origin);

/// Set the insertion point to the end of the module body.
static void setModuleInsertionPointToEnd(OpBuilder &builder, ModuleOp module) {
  builder.setInsertionPointToEnd(module.getBody());
}

/// Lower one LLZK scalar type to the corresponding SMT-facing type.
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

/// Return `true` iff the given type still contains an unsupported aggregate.
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

/// Return `true` iff the given type still contains an unsupported aggregate.
static bool typeContainsUnsupportedAggregate(LoweringContext &state, Type type, Operation *origin) {
  llvm::DenseSet<Type> visited;
  return typeContainsUnsupportedAggregate(state, type, origin, visited);
}

/// Emit an error if the given scalar-only type requirement is violated.
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

/// Emit an error if the given callable signature still contains aggregates.
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

/// Emit an error if the given contract or its target still contains aggregates.
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

/// Resolve the `struct.def` for the given struct type.
static FailureOr<StructDefOp>
resolveStructDef(LoweringContext &state, StructType type, Operation *origin) {
  return llzk::verifyStructTypeResolution(state.tables, type, origin);
}

/// Return the members of the given struct type in declaration order.
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

/// Lower the boundary signature of a callable to SMT-facing scalar types.
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

/// Build a left-associated conjunction of the given SMT boolean values.
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

/// Build element-wise equality conditions between two equally-sized value ranges.
static SmallVector<Value>
buildEqualityConditions(OpBuilder &builder, Location loc, ValueRange lhs, ValueRange rhs) {
  SmallVector<Value> conditions;
  for (auto [lhsValue, rhsValue] : llvm::zip(lhs, rhs)) {
    conditions.push_back(builder.create<llzk::smt::EqOp>(loc, lhsValue, rhsValue).getResult());
  }
  return conditions;
}

/// Populate one `smt.check` branch region and terminate it with `smt.yield`.
static void populateVoidCheckRegion(
    Region &region, Location loc, llvm::function_ref<void(OpBuilder &)> buildBody
) {
  Block *block = new Block();
  region.push_back(block);
  OpBuilder builder = OpBuilder::atBlockBegin(block);
  buildBody(builder);
  builder.create<llzk::smt::YieldOp>(loc);
}

/// Prove a condition by checking that its negation is unsat, then assert the condition on success.
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

/// Lower contract-local expressions into SMT operations inside one helper body.
class ExprLowerer {
public:
  /// Construct an expression lowerer that reuses previously lowered values.
  ExprLowerer(
      LoweringContext &state, OpBuilder &builder, DenseMap<Value, Value> &valueMap,
      DenseMap<StringRef, Value> &selfMemberMap
  )
      : state(state), builder(builder), valueMap(valueMap), selfMemberMap(selfMemberMap) {}

  /// Lower the given LLZK SSA value to an SMT-facing SSA value.
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
      return loweredCall.getResult(cast<OpResult>(value).getResultNumber());
    }

    definingOp->emitError("unsupported expression in verif-to-smt lowering");
    return failure();
  }

private:
  /// Shared pass state.
  LoweringContext &state;

  /// Rewriter used to create lowered operations.
  OpBuilder &builder;

  /// Lowered SSA value cache for ordinary arguments and derived values.
  DenseMap<Value, Value> &valueMap;

  /// Flattened `%self` member bindings keyed by member name.
  DenseMap<StringRef, Value> &selfMemberMap;
};

/// Lower a free function target to an SMT helper, reusing an existing helper if present.
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

  OpBuilder moduleBuilder(state.context);
  setModuleInsertionPointToEnd(moduleBuilder, module);
  std::string helperName = ("smt_" + func.getSymName()).str();
  auto helper = moduleBuilder.create<func::FuncOp>(
      func.getLoc(), helperName,
      FunctionType::get(state.context, loweredSig->argTypes, loweredSig->resultTypes)
  );

  Block *entry = helper.addEntryBlock();
  OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);
  DenseMap<Value, Value> valueMap;
  DenseMap<StringRef, Value> selfMemberMap;
  for (auto [original, lowered] : llvm::zip(func.getArguments(), entry->getArguments())) {
    valueMap[original] = lowered;
  }

  ExprLowerer lowerer(state, entryBuilder, valueMap, selfMemberMap);
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
  entryBuilder.create<func::ReturnOp>(returnOp.getLoc(), returnedValues);

  state.functionHelpers[func.getOperation()] = helper;
  return helper;
}

/// Lower a struct target to its SMT compute and constrain helpers.
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

  OpBuilder moduleBuilder(state.context);
  setModuleInsertionPointToEnd(moduleBuilder, module);

  std::string computeName = ("smt_" + structDef.getSymName() + "_compute").str();
  auto computeHelper = moduleBuilder.create<func::FuncOp>(
      computeFunc.getLoc(), computeName,
      FunctionType::get(state.context, computeSig->argTypes, memberTypes)
  );
  {
    Block *entry = computeHelper.addEntryBlock();
    OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);
    DenseMap<Value, Value> valueMap;
    DenseMap<StringRef, Value> selfMemberMap;
    for (auto [original, lowered] : llvm::zip(computeFunc.getArguments(), entry->getArguments())) {
      valueMap[original] = lowered;
    }

    ExprLowerer lowerer(state, entryBuilder, valueMap, selfMemberMap);
    DenseMap<StringRef, Value> writtenMembers;
    bool failedToLower = false;
    computeFunc.walk([&](MemberWriteOp writeOp) {
      if (failedToLower) {
        return;
      }
      auto loweredValue = lowerer.lower(writeOp.getVal());
      if (failed(loweredValue)) {
        failedToLower = true;
        return;
      }
      writtenMembers[writeOp.getMemberName()] = *loweredValue;
    });
    if (failedToLower) {
      return failure();
    }

    SmallVector<Value> results;
    for (MemberDefOp member : members) {
      auto it = writtenMembers.find(member.getSymName());
      if (it == writtenMembers.end()) {
        computeFunc.emitError().append("missing write for member @", member.getSymName());
        return failure();
      }
      results.push_back(it->second);
    }
    entryBuilder.create<func::ReturnOp>(computeFunc.getLoc(), results);
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
    OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);
    entryBuilder.create<func::ReturnOp>(
        constrainFunc.getLoc(), ValueRange(entry->getArguments()).take_front(memberTypes.size())
    );
  }

  TargetHelperNames helperNames {computeName, constrainName};
  state.structHelpers[structDef.getOperation()] = helperNames;
  return helperNames;
}

/// Lower the contract boundary signature to SMT-facing scalar types.
static FailureOr<LoweredSignature>
lowerContractSignature(LoweringContext &state, ContractOp contract) {
  return lowerCallableSignature(state, contract.getFunctionType(), contract);
}

/// Return `true` iff the contract contains compute-mode conditions.
static bool contractUsesComputeTarget(ContractOp contract) {
  bool usesCompute = false;
  contract.walk([&](Operation *op) {
    if (isa<RequireComputeOp, EnsureComputeOp>(op)) {
      usesCompute = true;
    }
  });
  return usesCompute;
}

/// Return `true` iff the contract contains constrain-mode conditions.
static bool contractUsesConstrainTarget(ContractOp contract) {
  bool usesConstrain = false;
  contract.walk([&](Operation *op) {
    if (isa<RequireConstrainOp, EnsureConstrainOp>(op)) {
      usesConstrain = true;
    }
  });
  return usesConstrain;
}

/// Seed lowered contract argument maps for ordinary args and flattened `%self` members.
static FailureOr<std::pair<DenseMap<Value, Value>, DenseMap<StringRef, Value>>>
seedContractArgumentMaps(LoweringContext &state, ContractOp contract, Block *entry) {
  DenseMap<Value, Value> valueMap;
  DenseMap<StringRef, Value> selfMemberMap;
  unsigned nextArg = 0;
  for (BlockArgument originalArg : contract.getArguments()) {
    if (auto structType = dyn_cast<StructType>(originalArg.getType())) {
      auto members = getStructMembers(state, structType, contract);
      if (failed(members)) {
        return failure();
      }
      for (MemberDefOp member : *members) {
        selfMemberMap[member.getSymName()] = entry->getArgument(nextArg++);
      }
      continue;
    }
    valueMap[originalArg] = entry->getArgument(nextArg++);
  }
  return std::make_pair(std::move(valueMap), std::move(selfMemberMap));
}

/// Create a direct contract condition helper, such as `_pre` or `_post`.
static FailureOr<func::FuncOp> createContractConditionHelper(
    LoweringContext &state, ModuleOp module, ContractOp contract, StringRef helperName,
    llvm::function_ref<bool(Operation *)> predicate
) {
  auto loweredSig = lowerContractSignature(state, contract);
  if (failed(loweredSig)) {
    return failure();
  }

  OpBuilder moduleBuilder(state.context);
  setModuleInsertionPointToEnd(moduleBuilder, module);
  auto helper = moduleBuilder.create<func::FuncOp>(
      contract.getLoc(), helperName,
      FunctionType::get(
          state.context, loweredSig->argTypes, TypeRange {llzk::smt::BoolType::get(state.context)}
      )
  );

  Block *entry = helper.addEntryBlock();
  OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);
  auto seededMaps = seedContractArgumentMaps(state, contract, entry);
  if (failed(seededMaps)) {
    return failure();
  }
  auto &[valueMap, selfMemberMap] = *seededMaps;
  ExprLowerer lowerer(state, entryBuilder, valueMap, selfMemberMap);

  SmallVector<Value> conditions;
  bool failedToLower = false;
  contract.walk([&](Operation *op) {
    if (failedToLower || !predicate(op)) {
      return;
    }

    auto conditionOp = dyn_cast<ConditionOpInterface>(op);
    auto lowered = lowerer.lower(conditionOp.getCondition());
    if (failed(lowered)) {
      failedToLower = true;
      return;
    }
    conditions.push_back(*lowered);
  });
  if (failedToLower) {
    return failure();
  }

  Value combined = buildConjunction(entryBuilder, contract.getLoc(), conditions);
  entryBuilder.create<func::ReturnOp>(contract.getLoc(), combined);
  return helper;
}

/// Create the `_target` helper for a free-function contract target.
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

  OpBuilder moduleBuilder(state.context);
  auto rawTargetHelper = getOrCreateFunctionHelper(state, module, targetFunc->get());
  if (failed(rawTargetHelper)) {
    return failure();
  }

  auto targetLoweredSig =
      lowerCallableSignature(state, targetFunc->get().getFunctionType(), targetFunc->get());
  if (failed(targetLoweredSig)) {
    return failure();
  }
  setModuleInsertionPointToEnd(moduleBuilder, module);
  std::string helperName = ("smt_verif_" + contract.getSymName() + "_target").str();
  auto helper = moduleBuilder.create<func::FuncOp>(
      contract.getLoc(), helperName,
      FunctionType::get(
          state.context, loweredSig->argTypes, TypeRange {llzk::smt::BoolType::get(state.context)}
      )
  );

  Block *entry = helper.addEntryBlock();
  OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);

  ValueRange args = entry->getArguments();
  ValueRange targetInputs = args.take_front(targetLoweredSig->argTypes.size());
  auto targetCall = entryBuilder.create<func::CallOp>(
      contract.getLoc(), rawTargetHelper->getSymName(),
      rawTargetHelper->getFunctionType().getResults(), targetInputs
  );

  ValueRange expectedResults = args.drop_front(targetLoweredSig->argTypes.size())
                                   .take_front(targetLoweredSig->resultTypes.size());
  SmallVector<Value> conditions = buildEqualityConditions(
      entryBuilder, contract.getLoc(), targetCall.getResults(), expectedResults
  );
  Value combined = buildConjunction(entryBuilder, contract.getLoc(), conditions);
  entryBuilder.create<func::ReturnOp>(contract.getLoc(), combined);
  return helper;
}

/// Create the `_target` helper for a struct contract target.
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
  OpBuilder moduleBuilder(state.context);
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
  setModuleInsertionPointToEnd(moduleBuilder, module);
  std::string helperName = ("smt_verif_" + contract.getSymName() + "_target").str();
  auto helper = moduleBuilder.create<func::FuncOp>(
      contract.getLoc(), helperName,
      FunctionType::get(
          state.context, loweredSig->argTypes, TypeRange {llzk::smt::BoolType::get(state.context)}
      )
  );

  Block *entry = helper.addEntryBlock();
  OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);
  ValueRange args = entry->getArguments();
  ValueRange selfArgs = args.take_front(numFlattenedSelfMembers);
  ValueRange nonSelfArgs = args.drop_front(numFlattenedSelfMembers);

  SmallVector<Value> conditions;
  if (useCompute) {
    auto computeCall = entryBuilder.create<func::CallOp>(
        contract.getLoc(), targetHelpers->compute, TypeRange(selfArgs.getTypes()), nonSelfArgs
    );
    SmallVector<Value> computeConditions = buildEqualityConditions(
        entryBuilder, contract.getLoc(), computeCall.getResults(), selfArgs
    );
    llvm::append_range(conditions, computeConditions);
  }
  if (useConstrain) {
    auto constrainCall = entryBuilder.create<func::CallOp>(
        contract.getLoc(), targetHelpers->constrain, TypeRange(selfArgs.getTypes()), args
    );
    SmallVector<Value> constrainConditions = buildEqualityConditions(
        entryBuilder, contract.getLoc(), constrainCall.getResults(), selfArgs
    );
    llvm::append_range(conditions, constrainConditions);
  }

  Value combined = buildConjunction(entryBuilder, contract.getLoc(), conditions);
  entryBuilder.create<func::ReturnOp>(contract.getLoc(), combined);
  return helper;
}

/// Create or look up the direct helper symbols for one contract.
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

/// Lower one include operand list using the contract-local lowering environment.
static FailureOr<SmallVector<Value>> lowerIncludeHelperOperands(
    LoweringContext &state, ValueRange operands, OpBuilder &builder,
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

/// Create a per-include helper that forwards to the callee contract entry helper.
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

  OpBuilder moduleBuilder(state.context);
  setModuleInsertionPointToEnd(moduleBuilder, module);
  auto helper = moduleBuilder.create<func::FuncOp>(
      includeOp.getLoc(), helperName,
      FunctionType::get(state.context, loweredSig->argTypes, TypeRange {})
  );

  Block *entry = helper.addEntryBlock();
  OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);
  entryBuilder.create<func::CallOp>(
      includeOp.getLoc(), calleeHelpers->entry, TypeRange {}, entry->getArguments()
  );
  entryBuilder.create<func::ReturnOp>(includeOp.getLoc());
  return helper;
}

/// Create the full `_entry` helper for one contract.
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

  OpBuilder moduleBuilder(state.context);
  setModuleInsertionPointToEnd(moduleBuilder, module);
  auto helper = moduleBuilder.create<func::FuncOp>(
      contract.getLoc(), helperInfo.entry,
      FunctionType::get(state.context, loweredSig->argTypes, TypeRange {})
  );

  Block *entry = helper.addEntryBlock();
  OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);
  auto seededMaps = seedContractArgumentMaps(state, contract, entry);
  if (failed(seededMaps)) {
    return failure();
  }
  auto &[valueMap, selfMemberMap] = *seededMaps;

  auto preCall = entryBuilder.create<func::CallOp>(
      contract.getLoc(), helperInfo.pre, TypeRange {llzk::smt::BoolType::get(state.context)},
      entry->getArguments()
  );
  proveByUnsatAndAssert(
      entryBuilder, contract.getLoc(), preCall.getResult(0), contract.getSymName(), "pre"
  );

  auto targetCall = entryBuilder.create<func::CallOp>(
      contract.getLoc(), helperInfo.target, TypeRange {llzk::smt::BoolType::get(state.context)},
      entry->getArguments()
  );
  proveByUnsatAndAssert(
      entryBuilder, contract.getLoc(), targetCall.getResult(0), contract.getSymName(), "target"
  );

  for (auto [index, includeOp] : llvm::enumerate(includes)) {
    auto loweredOperands = lowerIncludeHelperOperands(
        state, includeOp.getArgOperands(), entryBuilder, valueMap, selfMemberMap
    );
    if (failed(loweredOperands)) {
      return failure();
    }
    entryBuilder.create<func::CallOp>(
        includeOp.getLoc(), helperInfo.includeHelpers[index], TypeRange {}, *loweredOperands
    );
  }

  auto postCall = entryBuilder.create<func::CallOp>(
      contract.getLoc(), helperInfo.post, TypeRange {llzk::smt::BoolType::get(state.context)},
      entry->getArguments()
  );
  proveByUnsatAndAssert(
      entryBuilder, contract.getLoc(), postCall.getResult(0), contract.getSymName(), "post"
  );

  entryBuilder.create<func::ReturnOp>(contract.getLoc());
  return success();
}

/// Return the direct target definition for the given contract.
static FailureOr<Operation *>
getDirectTargetDefinition(LoweringContext &state, ContractOp contract) {
  if (contract.hasStructTarget()) {
    auto structTarget = contract.getStructTarget(state.tables);
    if (failed(structTarget)) {
      return failure();
    }
    return structTarget->get().getOperation();
  }

  auto targetFunc = llzk::lookupSymbolIn<FuncDefOp>(
      state.tables, contract.getTarget(), llzk::Within(contract.getOperation()->getParentOp()),
      contract, true
  );
  if (failed(targetFunc)) {
    return failure();
  }
  return targetFunc->get().getOperation();
}

/// Module pass that lowers scalar-only `verif` contracts into SMT helper functions.
struct VerifToSmtPass : public llzk::impl::VerifToSmtPassBase<VerifToSmtPass> {
  /// Register the dialects required by the generated SMT helper IR.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        arith::ArithDialect, func::FuncDialect, llzk::boolean::BoolDialect, llzk::smt::SMTDialect>(
    );
  }

  /// Run the contract lowering sequentially over the module body.
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

      auto target = getDirectTargetDefinition(state, contract);
      if (failed(target)) {
        signalPassFailure();
        return;
      }
      if (auto func = dyn_cast<FuncDefOp>(*target)) {
        if (failed(getOrCreateFunctionHelper(state, module, func))) {
          signalPassFailure();
          return;
        }
      } else {
        auto structDef = cast<StructDefOp>(*target);
        if (failed(getOrCreateStructHelpers(state, module, structDef))) {
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
