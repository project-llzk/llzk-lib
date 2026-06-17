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
#include "llzk/Util/Walk.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>

#include <string>

namespace llzk {
#define GEN_PASS_DECL_VERIFTOSMTPASS
#define GEN_PASS_DEF_VERIFTOSMTPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;
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
class LoweringContext {
public:
  explicit LoweringContext(MLIRContext *ctx) : context(ctx) {}

  /// Get the SMT-converted function for the given LLZK source function op.
  FailureOr<func::FuncOp> getFuncHelper(FuncDefOp sourceOp) {
    if (auto it = functionHelpers.find(sourceOp); it != functionHelpers.end()) {
      return it->second;
    }
    return failure();
  }

  /// Return the members of the given struct type in declaration order.
  FailureOr<SmallVector<MemberDefOp>> getStructMembers(StructType type, Operation *origin);

  /// Lookup callee target using the current context's SymbolTableColelction.
  FailureOr<SymbolLookupResult<FuncDefOp>> getCalleeTarget(CallOp call) {
    return call.getCalleeTarget(tables);
  }

  /// Emit an error if the given scalar-only type requirement is violated.
  LogicalResult ensureScalarTypeSupported(Type type, Operation *origin, StringRef description);

  /// Emit an error if the given callable signature still contains aggregates.
  LogicalResult
  ensureScalarSignatureSupported(FunctionType type, Operation *origin, StringRef description);

  /// Return the direct target definition for the given contract.
  FailureOr<Operation *> getDirectTargetDefinition(ContractOp contract);

  /// Emit an error if the given contract or its target still contains aggregates.
  LogicalResult ensureScalarContractSupported(ContractOp contract);

  /// Lower a free function target to an SMT helper, reusing an existing helper if present.
  FailureOr<func::FuncOp> getOrCreateFunctionHelper(ModuleOp module, FuncDefOp func);

  /// Lower a struct target to its SMT compute and constrain helpers.
  FailureOr<TargetHelperNames> getOrCreateStructHelpers(ModuleOp module, StructDefOp structDef);

  /// Create or look up the direct helper symbols for one contract.
  FailureOr<ContractHelperNames> getOrCreateContractHelpers(ModuleOp module, ContractOp contract);

  /// Create a per-include helper that forwards to the callee contract entry helper.
  FailureOr<func::FuncOp>
  createIncludeWrapperHelper(ModuleOp module, IncludeOp includeOp, StringRef helperName);

  /// Create the full `_entry` helper for one contract.
  LogicalResult createContractEntryHelper(
      ModuleOp module, ContractOp contract, SmallVectorImpl<IncludeOp> &includes
  );

  /// Return the struct definitions that were lowered to SMT helpers.
  ArrayRef<StructDefOp> getLoweredStructDefs() const { return loweredStructDefs; }

private:
  /// Resolve the `struct.def` for the given struct type.
  FailureOr<StructDefOp> resolveStructDef(StructType type, Operation *origin);

  /// Return `true` iff the given type still contains an unsupported aggregate.
  bool typeContainsUnsupportedAggregate(Type type, Operation *origin);

  /// Lower the boundary signature of a callable to SMT-facing scalar types.
  FailureOr<LoweredSignature> lowerCallableSignature(FunctionType type, Operation *origin);

  /// Lower the contract boundary signature to SMT-facing scalar types.
  FailureOr<LoweredSignature> lowerContractSignature(ContractOp contract);

  /// Seed lowered contract argument maps for ordinary args and flattened `%self` members.
  FailureOr<std::pair<DenseMap<Value, Value>, DenseMap<StringRef, Value>>>
  seedContractArgumentMaps(ContractOp contract, Block *entry);

  /// Create a direct contract condition helper, such as `_pre` or `_post`.
  FailureOr<func::FuncOp> createContractConditionHelper(
      ModuleOp module, ContractOp contract, StringRef helperName,
      llvm::function_ref<bool(Operation *)> predicate
  );

  /// Create the `_target` helper for a free-function contract target.
  FailureOr<func::FuncOp> createFreeFunctionTargetHelper(ModuleOp module, ContractOp contract);

  /// Create the `_target` helper for a struct contract target.
  FailureOr<func::FuncOp> createStructTargetHelper(ModuleOp module, ContractOp contract);

  /// Lower one include operand list using the contract-local lowering environment.
  FailureOr<SmallVector<Value>> lowerIncludeHelperOperands(
      ValueRange operands, OpBuilder &builder, DenseMap<Value, Value> &valueMap,
      DenseMap<StringRef, Value> &selfMemberMap
  );

  /// Lower one LLZK scalar type to the corresponding SMT-facing type.
  Type lowerType(Type type) {
    if (isa<FeltType>(type)) {
      return smt::IntType::get(context);
    }
    if (auto intType = dyn_cast<IntegerType>(type);
        intType && intType.isSignless() && intType.getWidth() == 1) {
      return smt::BoolType::get(context);
    }
    return type;
  }

  /// Create an OpBuilder in the current context and set the insertion point to
  /// the end of the module
  OpBuilder createModuleBuilder(ModuleOp module) {
    OpBuilder builder(context);
    builder.setInsertionPointToEnd(module.getBody());
    return builder;
  }

  bool typeContainsUnsupportedAggregateImpl(
      Type type, Operation *origin, llvm::DenseSet<Type> &visited
  ) {
    if (!visited.insert(type).second) {
      return false;
    }
    if (isa<array::ArrayType, pod::PodType>(type)) {
      return true;
    }
    if (auto structType = dyn_cast<StructType>(type)) {
      auto members = getStructMembers(structType, origin);
      if (failed(members)) {
        return true;
      }
      for (MemberDefOp member : *members) {
        if (typeContainsUnsupportedAggregateImpl(member.getType(), origin, visited)) {
          return true;
        }
      }
    }
    return false;
  }

  MLIRContext *context = nullptr;
  SymbolTableCollection tables {};

  /// Cached free-function helpers keyed by the original `function.def`.
  DenseMap<Operation *, func::FuncOp> functionHelpers {};

  /// Cached struct target helpers keyed by the original `struct.def`.
  DenseMap<Operation *, TargetHelperNames> structHelpers {};

  /// Struct targets that were lowered to SMT helpers and can be removed after lowering.
  SmallVector<StructDefOp> loweredStructDefs {};

  /// Set used to deduplicate lowered struct target cleanup.
  DenseSet<Operation *> loweredStructDefSet {};

  /// Cached contract helper bundles keyed by the original `verif.contract`.
  DenseMap<Operation *, ContractHelperNames> contractHelpers {};
};

/// Resolve the `struct.def` for the given struct type.
FailureOr<StructDefOp> LoweringContext::resolveStructDef(StructType type, Operation *origin) {
  return verifyStructTypeResolution(tables, type, origin);
}

/// Return the members of the given struct type in declaration order.
FailureOr<SmallVector<MemberDefOp>>
LoweringContext::getStructMembers(StructType type, Operation *origin) {
  auto structDef = resolveStructDef(type, origin);
  if (failed(structDef)) {
    return failure();
  }
  SmallVector<MemberDefOp> members;
  for (MemberDefOp member : structDef->getMemberDefs()) {
    members.push_back(member);
  }
  return members;
}

/// Return `true` iff the given type still contains an unsupported aggregate.
bool LoweringContext::typeContainsUnsupportedAggregate(Type type, Operation *origin) {
  llvm::DenseSet<Type> visited;
  return typeContainsUnsupportedAggregateImpl(type, origin, visited);
}

/// Emit an error if the given scalar-only type requirement is violated.
LogicalResult
LoweringContext::ensureScalarTypeSupported(Type type, Operation *origin, StringRef description) {
  if (!typeContainsUnsupportedAggregate(type, origin)) {
    return success();
  }
  origin->emitError() << "llzk-scalar-verif-to-smt requires array/pod-free IR for " << description
                      << "; run -llzk-verif-to-smt instead";
  return failure();
}

/// Emit an error if the given callable signature still contains aggregates.
LogicalResult LoweringContext::ensureScalarSignatureSupported(
    FunctionType type, Operation *origin, StringRef description
) {
  for (Type input : type.getInputs()) {
    if (failed(ensureScalarTypeSupported(input, origin, description))) {
      return failure();
    }
  }
  for (Type result : type.getResults()) {
    if (failed(ensureScalarTypeSupported(result, origin, description))) {
      return failure();
    }
  }
  return success();
}

/// Emit an error if the given contract or its target still contains aggregates.
LogicalResult LoweringContext::ensureScalarContractSupported(ContractOp contract) {
  if (failed(
          ensureScalarSignatureSupported(contract.getFunctionType(), contract, "contract signature")
      )) {
    return failure();
  }

  if (contract.hasStructTarget()) {
    auto structTarget = contract.getStructTarget(tables);
    if (failed(structTarget)) {
      return failure();
    }
    for (MemberDefOp member : structTarget->get().getMemberDefs()) {
      if (failed(
              ensureScalarTypeSupported(member.getType(), contract, "struct target member types")
          )) {
        return failure();
      }
    }
    return success();
  }

  auto targetFunc = lookupSymbolIn<FuncDefOp>(
      tables, contract.getTarget(), Within(contract->getParentOp()), contract, true
  );
  if (failed(targetFunc)) {
    return failure();
  }
  return ensureScalarSignatureSupported(
      targetFunc->get().getFunctionType(), contract, "target function signature"
  );
}

/// Lower the boundary signature of a callable to SMT-facing scalar types.
FailureOr<LoweredSignature>
LoweringContext::lowerCallableSignature(FunctionType type, Operation *origin) {
  LoweredSignature lowered;
  for (Type input : type.getInputs()) {
    if (auto structType = dyn_cast<StructType>(input)) {
      auto members = getStructMembers(structType, origin);
      if (failed(members)) {
        return failure();
      }
      for (MemberDefOp member : *members) {
        lowered.argTypes.push_back(lowerType(member.getType()));
      }
      continue;
    }
    lowered.argTypes.push_back(lowerType(input));
  }
  for (Type result : type.getResults()) {
    lowered.resultTypes.push_back(lowerType(result));
  }
  return lowered;
}

/// Build a left-associated conjunction of the given SMT boolean values.
static Value buildConjunction(OpBuilder &builder, Location loc, ArrayRef<Value> conditions) {
  if (conditions.empty()) {
    return builder.create<smt::BoolConstantOp>(loc, builder.getBoolAttr(true)).getResult();
  }

  Value current = conditions.front();
  for (Value condition : conditions.drop_front()) {
    current = builder.create<smt::AndOp>(loc, current, condition).getResult();
  }
  return current;
}

/// Build element-wise equality conditions between two equally-sized value ranges.
static SmallVector<Value>
buildEqualityConditions(OpBuilder &builder, Location loc, ValueRange lhs, ValueRange rhs) {
  SmallVector<Value> conditions;
  for (auto [lhsValue, rhsValue] : llvm::zip(lhs, rhs)) {
    conditions.push_back(builder.create<smt::EqOp>(loc, lhsValue, rhsValue).getResult());
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
  builder.create<smt::YieldOp>(loc);
}

/// Prove a condition by checking that its negation is unsat, then assert the condition on success.
static void proveByUnsatAndAssert(
    OpBuilder &builder, Location loc, Value condition, StringRef contractName, StringRef stageName
) {
  builder.create<smt::PushOp>(loc, 1);
  Value negated = builder.create<smt::NotOp>(loc, condition).getResult();
  builder.create<smt::AssertOp>(loc, negated);
  auto check = builder.create<smt::CheckOp>(loc, TypeRange {});

  auto makeFailureMessage = [&](StringRef outcome) {
    return builder.getStringAttr(
        (Twine("verification failed in ") + contractName + " " + stageName + ": " + outcome).str()
    );
  };

  populateVoidCheckRegion(check.getSatRegion(), loc, [&](OpBuilder &regionBuilder) {
    regionBuilder.create<smt::PopOp>(loc, 1);
    Value failed = regionBuilder.create<arith::ConstantOp>(loc, regionBuilder.getBoolAttr(false));
    regionBuilder.create<boolean::AssertOp>(
        loc, failed, makeFailureMessage("counterexample found")
    );
  });
  populateVoidCheckRegion(check.getUnknownRegion(), loc, [&](OpBuilder &regionBuilder) {
    regionBuilder.create<smt::PopOp>(loc, 1);
    Value failed = regionBuilder.create<arith::ConstantOp>(loc, regionBuilder.getBoolAttr(false));
    regionBuilder.create<boolean::AssertOp>(
        loc, failed, makeFailureMessage("condition unprovable")
    );
  });
  populateVoidCheckRegion(check.getUnsatRegion(), loc, [&](OpBuilder &regionBuilder) {
    regionBuilder.create<smt::PopOp>(loc, 1);
    regionBuilder.create<smt::AssertOp>(loc, condition);
  });
}

/// Lower contract-local expressions into SMT operations inside one helper body.
class ExprLowerer {
public:
  /// Construct an expression lowerer that reuses previously lowered values.
  ExprLowerer(
      LoweringContext &l, OpBuilder &b, DenseMap<Value, Value> &v, DenseMap<StringRef, Value> &s
  )
      : state(l), builder(b), valueMap(v), selfMemberMap(s) {}

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
      auto lowered = builder.create<smt::IntConstantOp>(
          feltConst.getLoc(),
          IntegerAttr::get(builder.getContext(), APSInt(feltConst.getValue().getValue()))
      );
      valueMap[value] = lowered.getResult();
      return lowered.getResult();
    }

    if (auto constOp = dyn_cast<arith::ConstantOp>(definingOp)) {
      if (auto boolAttr = dyn_cast<BoolAttr>(constOp.getValue())) {
        auto lowered = builder.create<smt::BoolConstantOp>(constOp.getLoc(), boolAttr).getResult();
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

    if (auto boolCmp = dyn_cast<boolean::CmpOp>(definingOp)) {
      auto lhs = lower(boolCmp.getLhs());
      auto rhs = lower(boolCmp.getRhs());
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }

      Value cmp = [&]() -> Value {
        using boolean::FeltCmpPredicate;
        switch (boolCmp.getPredicate()) {
        case FeltCmpPredicate::EQ:
          return builder.create<smt::EqOp>(boolCmp.getLoc(), *lhs, *rhs).getResult();
        case FeltCmpPredicate::NE: {
          Value eq = builder.create<smt::EqOp>(boolCmp.getLoc(), *lhs, *rhs).getResult();
          return builder.create<smt::NotOp>(boolCmp.getLoc(), eq).getResult();
        }
        case FeltCmpPredicate::LT:
          return builder.create<smt::IntCmpOp>(boolCmp.getLoc(), smt::IntPredicate::lt, *lhs, *rhs)
              .getResult();
        case FeltCmpPredicate::LE:
          return builder.create<smt::IntCmpOp>(boolCmp.getLoc(), smt::IntPredicate::le, *lhs, *rhs)
              .getResult();
        case FeltCmpPredicate::GT:
          return builder.create<smt::IntCmpOp>(boolCmp.getLoc(), smt::IntPredicate::gt, *lhs, *rhs)
              .getResult();
        case FeltCmpPredicate::GE:
          return builder.create<smt::IntCmpOp>(boolCmp.getLoc(), smt::IntPredicate::ge, *lhs, *rhs)
              .getResult();
        }
        llvm_unreachable("unknown bool.cmp predicate");
      }();

      valueMap[value] = cmp;
      return cmp;
    }

    if (auto call = dyn_cast<CallOp>(definingOp)) {
      auto calleeTarget = state.getCalleeTarget(call);
      if (failed(calleeTarget)) {
        return failure();
      }

      auto funcHelperRes = state.getFuncHelper(calleeTarget->get());
      if (failed(funcHelperRes)) {
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
          call.getLoc(), funcHelperRes->getSymName(), funcHelperRes->getFunctionType().getResults(),
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
FailureOr<func::FuncOp>
LoweringContext::getOrCreateFunctionHelper(ModuleOp module, FuncDefOp func) {
  if (auto it = functionHelpers.find(func.getOperation()); it != functionHelpers.end()) {
    return it->second;
  }

  auto loweredSig = lowerCallableSignature(func.getFunctionType(), func);
  if (failed(loweredSig)) {
    return failure();
  }

  OpBuilder moduleBuilder = createModuleBuilder(module);
  std::string helperName = ("smt_" + func.getSymName()).str();
  auto helper = moduleBuilder.create<func::FuncOp>(
      func.getLoc(), helperName,
      FunctionType::get(context, loweredSig->argTypes, loweredSig->resultTypes)
  );

  Block *entry = helper.addEntryBlock();
  OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);
  DenseMap<Value, Value> valueMap;
  DenseMap<StringRef, Value> selfMemberMap;
  for (auto [original, lowered] : llvm::zip(func.getArguments(), entry->getArguments())) {
    valueMap[original] = lowered;
  }

  ExprLowerer lowerer(*this, entryBuilder, valueMap, selfMemberMap);
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

  functionHelpers[func.getOperation()] = helper;
  return helper;
}

/// Lower a struct target to its SMT compute and constrain helpers.
FailureOr<TargetHelperNames>
LoweringContext::getOrCreateStructHelpers(ModuleOp module, StructDefOp structDef) {
  if (auto it = structHelpers.find(structDef.getOperation()); it != structHelpers.end()) {
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
    memberTypes.push_back(lowerType(member.getType()));
  }

  auto computeSig = lowerCallableSignature(computeFunc.getFunctionType(), computeFunc);
  if (failed(computeSig)) {
    return failure();
  }

  OpBuilder moduleBuilder = createModuleBuilder(module);

  std::string computeName = ("smt_" + structDef.getSymName() + "_compute").str();
  auto computeHelper = moduleBuilder.create<func::FuncOp>(
      computeFunc.getLoc(), computeName,
      FunctionType::get(context, computeSig->argTypes, memberTypes)
  );
  {
    Block *entry = computeHelper.addEntryBlock();
    OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);
    DenseMap<Value, Value> valueMap;
    DenseMap<StringRef, Value> selfMemberMap;
    for (auto [original, lowered] : llvm::zip(computeFunc.getArguments(), entry->getArguments())) {
      valueMap[original] = lowered;
    }

    ExprLowerer lowerer(*this, entryBuilder, valueMap, selfMemberMap);
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
    constrainArgs.push_back(lowerType(input));
  }
  auto constrainHelper = moduleBuilder.create<func::FuncOp>(
      constrainFunc.getLoc(), constrainName, FunctionType::get(context, constrainArgs, memberTypes)
  );
  {
    Block *entry = constrainHelper.addEntryBlock();
    OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);
    entryBuilder.create<func::ReturnOp>(
        constrainFunc.getLoc(), ValueRange(entry->getArguments()).take_front(memberTypes.size())
    );
  }

  TargetHelperNames helperNames {computeName, constrainName};
  structHelpers[structDef.getOperation()] = helperNames;
  if (loweredStructDefSet.insert(structDef.getOperation()).second) {
    loweredStructDefs.push_back(structDef);
  }
  return helperNames;
}

/// Lower the contract boundary signature to SMT-facing scalar types.
FailureOr<LoweredSignature> LoweringContext::lowerContractSignature(ContractOp contract) {
  return lowerCallableSignature(contract.getFunctionType(), contract);
}

/// Seed lowered contract argument maps for ordinary args and flattened `%self` members.
FailureOr<std::pair<DenseMap<Value, Value>, DenseMap<StringRef, Value>>>
LoweringContext::seedContractArgumentMaps(ContractOp contract, Block *entry) {
  DenseMap<Value, Value> valueMap;
  DenseMap<StringRef, Value> selfMemberMap;
  unsigned nextArg = 0;
  for (BlockArgument originalArg : contract.getArguments()) {
    if (auto structType = dyn_cast<StructType>(originalArg.getType())) {
      auto members = getStructMembers(structType, contract);
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
FailureOr<func::FuncOp> LoweringContext::createContractConditionHelper(
    ModuleOp module, ContractOp contract, StringRef helperName,
    llvm::function_ref<bool(Operation *)> predicate
) {
  auto loweredSig = lowerContractSignature(contract);
  if (failed(loweredSig)) {
    return failure();
  }

  OpBuilder moduleBuilder = createModuleBuilder(module);
  auto helper = moduleBuilder.create<func::FuncOp>(
      contract.getLoc(), helperName,
      FunctionType::get(context, loweredSig->argTypes, TypeRange {smt::BoolType::get(context)})
  );

  Block *entry = helper.addEntryBlock();
  OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);
  auto seededMaps = seedContractArgumentMaps(contract, entry);
  if (failed(seededMaps)) {
    return failure();
  }
  auto &[valueMap, selfMemberMap] = *seededMaps;
  ExprLowerer lowerer(*this, entryBuilder, valueMap, selfMemberMap);

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
FailureOr<func::FuncOp>
LoweringContext::createFreeFunctionTargetHelper(ModuleOp module, ContractOp contract) {
  auto loweredSig = lowerContractSignature(contract);
  if (failed(loweredSig)) {
    return failure();
  }

  auto targetFunc = lookupSymbolIn<FuncDefOp>(
      tables, contract.getTarget(), Within(module.getOperation()), contract,
      /*reportMissing=*/true
  );
  if (failed(targetFunc)) {
    return failure();
  }

  auto rawTargetHelper = getOrCreateFunctionHelper(module, targetFunc->get());
  if (failed(rawTargetHelper)) {
    return failure();
  }

  auto targetLoweredSig =
      lowerCallableSignature(targetFunc->get().getFunctionType(), targetFunc->get());
  if (failed(targetLoweredSig)) {
    return failure();
  }
  OpBuilder moduleBuilder = createModuleBuilder(module);
  std::string helperName = ("smt_verif_" + contract.getSymName() + "_target").str();
  auto helper = moduleBuilder.create<func::FuncOp>(
      contract.getLoc(), helperName,
      FunctionType::get(context, loweredSig->argTypes, TypeRange {smt::BoolType::get(context)})
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
FailureOr<func::FuncOp>
LoweringContext::createStructTargetHelper(ModuleOp module, ContractOp contract) {
  auto loweredSig = lowerContractSignature(contract);
  if (failed(loweredSig)) {
    return failure();
  }

  auto structTarget = contract.getStructTarget(tables);
  if (failed(structTarget)) {
    return failure();
  }
  auto targetHelpers = getOrCreateStructHelpers(module, structTarget->get());
  if (failed(targetHelpers)) {
    return failure();
  }

  auto members = getStructMembers(structTarget->get().getType(), contract);
  if (failed(members)) {
    return failure();
  }
  unsigned numFlattenedSelfMembers = members->size();

  OpBuilder moduleBuilder = createModuleBuilder(module);
  std::string helperName = ("smt_verif_" + contract.getSymName() + "_target").str();
  auto helper = moduleBuilder.create<func::FuncOp>(
      contract.getLoc(), helperName,
      FunctionType::get(context, loweredSig->argTypes, TypeRange {smt::BoolType::get(context)})
  );

  Block *entry = helper.addEntryBlock();
  OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);
  ValueRange args = entry->getArguments();
  ValueRange selfArgs = args.take_front(numFlattenedSelfMembers);
  ValueRange nonSelfArgs = args.drop_front(numFlattenedSelfMembers);

  SmallVector<Value> conditions;
  auto computeCall = entryBuilder.create<func::CallOp>(
      contract.getLoc(), targetHelpers->compute, TypeRange(selfArgs.getTypes()), nonSelfArgs
  );
  SmallVector<Value> computeConditions =
      buildEqualityConditions(entryBuilder, contract.getLoc(), computeCall.getResults(), selfArgs);
  llvm::append_range(conditions, computeConditions);

  auto constrainCall = entryBuilder.create<func::CallOp>(
      contract.getLoc(), targetHelpers->constrain, TypeRange(selfArgs.getTypes()), args
  );
  SmallVector<Value> constrainConditions = buildEqualityConditions(
      entryBuilder, contract.getLoc(), constrainCall.getResults(), selfArgs
  );
  llvm::append_range(conditions, constrainConditions);

  Value combined = buildConjunction(entryBuilder, contract.getLoc(), conditions);
  entryBuilder.create<func::ReturnOp>(contract.getLoc(), combined);
  return helper;
}

/// Create or look up the direct helper symbols for one contract.
FailureOr<ContractHelperNames>
LoweringContext::getOrCreateContractHelpers(ModuleOp module, ContractOp contract) {
  if (auto it = contractHelpers.find(contract.getOperation()); it != contractHelpers.end()) {
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
  contractHelpers[contract.getOperation()] = names;

  auto preHelper = createContractConditionHelper(module, contract, preName, [](Operation *op) {
    return isa<RequireComputeOp, RequireConstrainOp>(op);
  });
  if (failed(preHelper)) {
    return failure();
  }

  auto postHelper = createContractConditionHelper(module, contract, postName, [](Operation *op) {
    return isa<EnsureComputeOp, EnsureConstrainOp>(op);
  });
  if (failed(postHelper)) {
    return failure();
  }

  FailureOr<func::FuncOp> targetHelper = contract.hasStructTarget()
                                             ? createStructTargetHelper(module, contract)
                                             : createFreeFunctionTargetHelper(module, contract);
  if (failed(targetHelper)) {
    return failure();
  }

  return names;
}

/// Lower one include operand list using the contract-local lowering environment.
FailureOr<SmallVector<Value>> LoweringContext::lowerIncludeHelperOperands(
    ValueRange operands, OpBuilder &builder, DenseMap<Value, Value> &valueMap,
    DenseMap<StringRef, Value> &selfMemberMap
) {
  ExprLowerer lowerer(*this, builder, valueMap, selfMemberMap);
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
FailureOr<func::FuncOp> LoweringContext::createIncludeWrapperHelper(
    ModuleOp module, IncludeOp includeOp, StringRef helperName
) {
  auto calleeTarget = includeOp.getCalleeTarget(tables);
  if (failed(calleeTarget)) {
    return failure();
  }
  ContractOp callee = calleeTarget->get();
  auto calleeHelpers = getOrCreateContractHelpers(module, callee);
  if (failed(calleeHelpers)) {
    return failure();
  }

  auto loweredSig = lowerContractSignature(callee);
  if (failed(loweredSig)) {
    return failure();
  }

  OpBuilder moduleBuilder = createModuleBuilder(module);
  auto helper = moduleBuilder.create<func::FuncOp>(
      includeOp.getLoc(), helperName, FunctionType::get(context, loweredSig->argTypes, TypeRange {})
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
LogicalResult LoweringContext::createContractEntryHelper(
    ModuleOp module, ContractOp contract, SmallVectorImpl<IncludeOp> &includes
) {
  auto loweredSig = lowerContractSignature(contract);
  if (failed(loweredSig)) {
    return failure();
  }

  auto helperInfoIt = contractHelpers.find(contract.getOperation());
  if (helperInfoIt == contractHelpers.end()) {
    contract.emitError("missing contract helper names for entry helper generation");
    return failure();
  }
  const ContractHelperNames &helperInfo = helperInfoIt->second;

  OpBuilder moduleBuilder = createModuleBuilder(module);
  auto helper = moduleBuilder.create<func::FuncOp>(
      contract.getLoc(), helperInfo.entry,
      FunctionType::get(context, loweredSig->argTypes, TypeRange {})
  );

  Block *entry = helper.addEntryBlock();
  OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);
  auto seededMaps = seedContractArgumentMaps(contract, entry);
  if (failed(seededMaps)) {
    return failure();
  }
  auto &[valueMap, selfMemberMap] = *seededMaps;

  auto preCall = entryBuilder.create<func::CallOp>(
      contract.getLoc(), helperInfo.pre, TypeRange {smt::BoolType::get(context)},
      entry->getArguments()
  );
  proveByUnsatAndAssert(
      entryBuilder, contract.getLoc(), preCall.getResult(0), contract.getSymName(), "pre"
  );

  auto targetCall = entryBuilder.create<func::CallOp>(
      contract.getLoc(), helperInfo.target, TypeRange {smt::BoolType::get(context)},
      entry->getArguments()
  );
  proveByUnsatAndAssert(
      entryBuilder, contract.getLoc(), targetCall.getResult(0), contract.getSymName(), "target"
  );

  for (auto [index, includeOp] : llvm::enumerate(includes)) {
    auto loweredOperands = lowerIncludeHelperOperands(
        includeOp.getArgOperands(), entryBuilder, valueMap, selfMemberMap
    );
    if (failed(loweredOperands)) {
      return failure();
    }
    entryBuilder.create<func::CallOp>(
        includeOp.getLoc(), helperInfo.includeHelpers[index], TypeRange {}, *loweredOperands
    );
  }

  auto postCall = entryBuilder.create<func::CallOp>(
      contract.getLoc(), helperInfo.post, TypeRange {smt::BoolType::get(context)},
      entry->getArguments()
  );
  proveByUnsatAndAssert(
      entryBuilder, contract.getLoc(), postCall.getResult(0), contract.getSymName(), "post"
  );

  entryBuilder.create<func::ReturnOp>(contract.getLoc());
  return success();
}

/// Return the direct target definition for the given contract.
FailureOr<Operation *> LoweringContext::getDirectTargetDefinition(ContractOp contract) {
  if (contract.hasStructTarget()) {
    auto structTarget = contract.getStructTarget(tables);
    if (failed(structTarget)) {
      return failure();
    }
    return structTarget->get().getOperation();
  }

  auto targetFunc = lookupSymbolIn<FuncDefOp>(
      tables, contract.getTarget(), Within(contract.getOperation()->getParentOp()), contract, true
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
    registry
        .insert<arith::ArithDialect, func::FuncDialect, boolean::BoolDialect, smt::SMTDialect>();
  }

  /// Run the contract lowering sequentially over the module body.
  void runOnOperation() override {
    ModuleOp module = getOperation();
    LoweringContext state {&getContext()};

    SmallVector<ContractOp> contracts;
    module.walk([&](ContractOp contract) { contracts.push_back(contract); });

    for (ContractOp contract : contracts) {
      if (failed(state.ensureScalarContractSupported(contract))) {
        signalPassFailure();
        return;
      }

      auto target = state.getDirectTargetDefinition(contract);
      if (failed(target)) {
        signalPassFailure();
        return;
      }
      if (auto func = dyn_cast<FuncDefOp>(*target)) {
        if (failed(state.getOrCreateFunctionHelper(module, func))) {
          signalPassFailure();
          return;
        }
      } else {
        auto structDef = cast<StructDefOp>(*target);
        if (failed(state.getOrCreateStructHelpers(module, structDef))) {
          signalPassFailure();
          return;
        }
      }

      auto contractHelpers = state.getOrCreateContractHelpers(module, contract);
      if (failed(contractHelpers)) {
        signalPassFailure();
        return;
      }

      SmallVector<IncludeOp> includes;
      contract.walk([&](IncludeOp includeOp) { includes.push_back(includeOp); });
      for (auto [index, includeOp] : llvm::enumerate(includes)) {
        if (failed(state.createIncludeWrapperHelper(
                module, includeOp, contractHelpers->includeHelpers[index]
            ))) {
          signalPassFailure();
          return;
        }
      }

      if (failed(state.createContractEntryHelper(module, contract, includes))) {
        signalPassFailure();
        return;
      }
    }

    for (ContractOp contract : contracts) {
      contract.erase();
    }
    for (StructDefOp structDef : state.getLoweredStructDefs()) {
      structDef.erase();
    }
  }
};

} // namespace
