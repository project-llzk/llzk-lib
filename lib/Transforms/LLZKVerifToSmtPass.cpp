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
/// 2. `@smt_verif_<Contract>_compute_{pre,target,post}` and
///    `@smt_verif_<Contract>_constrain_{pre,target,post}` helpers that lower the direct contract
///    conditions into SMT booleans for the two verification stages independently.
///
/// 3. `@smt_verif_<Contract>_{compute,constrain}_include_<N>` wrappers, one per `verif.include`
///    and verification stage, that call the corresponding callee contract entry helper
///    transitively.
///
/// 4. `@smt_verif_<Contract>_{compute,constrain}_entry`, which prove each direct contract stage
///    by checking that the negated condition is unsat before asserting the condition into the
///    caller solver context.
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
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/SMT/IR/SMTDialect.h"
#include "llzk/Dialect/SMT/IR/SMTOps.h"
#include "llzk/Dialect/SMT/IR/SMTTypes.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "llzk/Dialect/Verif/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/Field.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/Walk.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/raw_ostream.h>

#include <optional>
#include <string>
#include <tuple>

namespace llzk {
#define GEN_PASS_DEF_VERIFTOSMTPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;
using namespace llzk::component;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::polymorphic;
using namespace llzk::verif;

namespace {

/// Verify that one `scf.if` branch region is a single block ending in `scf.yield`.
static LogicalResult
validateSingleBlockRegion(scf::IfOp ifOp, Region &region, StringRef branchName, StringRef context) {
  if (!llvm::hasSingleElement(region)) {
    return ifOp.emitOpError() << "expects a single-block " << branchName
                              << " region while lowering " << context << " to SMT";
  }

  auto &block = region.front();
  if (!isa<scf::YieldOp>(block.getTerminator())) {
    return ifOp.emitOpError() << "expects " << branchName
                              << " region to terminate with scf.yield while lowering " << context
                              << " to SMT";
  }

  return success();
}

/// Verify that an `scf.if` has the structural shape required by this lowering.
static LogicalResult validateIfShape(scf::IfOp ifOp, bool requireElseRegion, StringRef context) {
  if (failed(validateSingleBlockRegion(ifOp, ifOp.getThenRegion(), "then", context))) {
    return failure();
  }
  if (!requireElseRegion && ifOp.getElseRegion().empty()) {
    return success();
  }
  if (failed(validateSingleBlockRegion(ifOp, ifOp.getElseRegion(), "else", context))) {
    return failure();
  }
  return success();
}

/// Return whether `descendant` is nested under the then-region of `ifOp`.
static FailureOr<bool> isInThenRegion(scf::IfOp ifOp, Operation *descendant) {
  for (Operation *cursor = descendant; cursor != nullptr; cursor = cursor->getParentOp()) {
    Region *parentRegion = cursor->getParentRegion();
    if (parentRegion == &ifOp.getThenRegion()) {
      return true;
    }
    if (parentRegion == &ifOp.getElseRegion()) {
      return false;
    }
  }

  ifOp.emitOpError("failed to determine descendant branch while lowering to SMT");
  return failure();
}

static bool templateHasLowerableDefs(polymorphic::TemplateOp templateOp) {
  return llvm::any_of(templateOp.getBodyRegion().front().getOperations(), [](Operation &op) {
    return isa<StructDefOp, FuncDefOp>(op);
  });
}

/// Stores the helper symbol names created for a struct target.
struct TargetHelperNames {
  /// Symbol name of the lowered `@compute` helper.
  std::string compute;

  /// Symbol name of the lowered `@constrain` helper.
  std::string constrain;
};

/// Stores the helper symbol names created for one contract.
struct ContractStageHelperNames {
  /// Symbol name of the direct precondition helper.
  std::string pre;

  /// Symbol name of the stage target helper.
  std::string target;

  /// Symbol name of the direct postcondition helper.
  std::string post;

  /// Symbol name of the full contract entry helper.
  std::string entry;

  /// Symbol names of the per-include wrapper helpers in source order.
  SmallVector<std::string> includeHelpers;
};

/// Stores the compute/constrain helper-name bundles created for one contract.
struct ContractHelperNames {
  ContractStageHelperNames compute;
  ContractStageHelperNames constrain;
  ContractStageHelperNames combined;
};

/// Distinguishes compute-side and constrain-side contract lowering.
enum class VerificationStage { Compute, Constrain };

/// Return the naming suffix used for stage-specific helpers.
static StringRef stageSuffix(VerificationStage stage) {
  switch (stage) {
  case VerificationStage::Compute:
    return "compute";
  case VerificationStage::Constrain:
    return "constrain";
  }
  llvm_unreachable("unknown verification stage");
}

/// Return a human-readable stage prefix for solver failure messages.
static std::string stageMessagePrefix(VerificationStage stage) {
  return (Twine(stageSuffix(stage)) + " ").str();
}

/// Lowered `%self` members keyed by source aggregate SSA value and member name.
using FlattenedMemberValues = SmallVector<Value>;
using SelfMemberMap = DenseMap<Value, DenseMap<StringRef, FlattenedMemberValues>>;

/// Represents the lowered SMT-facing argument and result types for a callable.
struct LoweredSignature {
  /// Lowered argument types.
  SmallVector<Type> argTypes;

  /// Lowered result types.
  SmallVector<Type> resultTypes;

  /// Enclosing non-type template params threaded through the helper.
  SmallVector<polymorphic::TemplateParamOp> templateParams;
};

/// Shared lowering state for one execution of the pass.
class LoweringContext {
public:
  explicit LoweringContext(MLIRContext *ctx, bool emitLegacyEntries)
      : context(ctx), emitLegacyEntries(emitLegacyEntries) {}
  friend class ExprLowerer;

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
  FailureOr<func::FuncOp> createIncludeWrapperHelper(
      ModuleOp module, IncludeOp includeOp, StringRef helperName, VerificationStage stage
  );

  /// Create a compatibility include helper that forwards to the legacy combined contract entry.
  FailureOr<func::FuncOp>
  createCombinedIncludeWrapperHelper(ModuleOp module, IncludeOp includeOp, StringRef helperName);

  /// Create the full `_entry` helper for one contract.
  LogicalResult createContractEntryHelper(
      ModuleOp module, ContractOp contract, SmallVectorImpl<IncludeOp> &includes,
      VerificationStage stage
  );

  /// Create the compatibility `_entry` helper consumed by `-smt-to-smtlib`.
  LogicalResult createCombinedContractEntryHelper(
      ModuleOp module, ContractOp contract, SmallVectorImpl<IncludeOp> &includes
  );

  /// Return the struct definitions that were lowered to SMT helpers.
  ArrayRef<StructDefOp> getLoweredStructDefs() const { return loweredStructDefs; }

private:
  /// Resolve the `struct.def` for the given struct type.
  FailureOr<StructDefOp> resolveStructDef(StructType type, Operation *origin);

  /// Return `true` iff the given type still contains an unsupported aggregate.
  bool typeContainsUnsupportedAggregate(Type type, Operation *origin);

  /// Append the recursively flattened SMT-facing boundary types for `type`.
  LogicalResult
  appendLoweredBoundaryTypes(Type type, Operation *origin, SmallVectorImpl<Type> &loweredTypes);

  /// Return the number of recursively flattened SMT-facing boundary values for `type`.
  FailureOr<unsigned> getNumLoweredBoundaryValues(Type type, Operation *origin);

  /// Seed one aggregate SSA value with the flattened values of its direct members.
  LogicalResult seedFlattenedStructValue(
      Value aggregate, StructType structType, ArrayRef<Value> loweredValues, Operation *origin,
      SelfMemberMap &selfMemberMap
  );

  /// Materialize a flattened mapping for `aggregate` if it is derived from a struct member read.
  LogicalResult
  ensureFlattenedStructMapping(Value aggregate, Operation *origin, SelfMemberMap &selfMemberMap);

  /// Lower one source SSA value to its recursively flattened SMT-facing values.
  FailureOr<SmallVector<Value>> lowerFlattenedValue(
      Value value, Operation *origin, OpBuilder &builder, DenseMap<Value, Value> &valueMap,
      SelfMemberMap &selfMemberMap, DenseMap<StringRef, Value> &constParamMap
  );

  /// Lower the boundary signature of a callable to SMT-facing scalar types.
  FailureOr<LoweredSignature> lowerCallableSignature(FunctionType type, Operation *origin);

  /// Seed lowered callable argument maps for ordinary args and flattened struct members.
  FailureOr<unsigned> seedCallableArgumentMaps(
      ValueRange originalArgs, Block *entry, Operation *origin, DenseMap<Value, Value> &valueMap,
      SelfMemberMap &selfMemberMap
  );

  /// Lower the contract boundary signature to SMT-facing scalar types.
  FailureOr<LoweredSignature> lowerContractSignature(ContractOp contract);

  /// Seed lowered contract argument maps for ordinary args and flattened `%self` members.
  FailureOr<std::tuple<DenseMap<Value, Value>, SelfMemberMap, DenseMap<StringRef, Value>>>
  seedContractArgumentMaps(ContractOp contract, Block *entry);

  /// Create a direct contract condition helper, such as `_pre` or `_post`.
  FailureOr<func::FuncOp> createContractConditionHelper(
      ModuleOp module, ContractOp contract, StringRef helperName,
      llvm::function_ref<bool(Operation *)> predicate
  );

  /// Create the shared `_target` helper for a free-function contract target.
  FailureOr<func::FuncOp> createFreeFunctionTargetHelper(ModuleOp module, ContractOp contract);

  /// Create the `_target` helper for a struct contract target.
  FailureOr<func::FuncOp>
  createStructTargetHelper(ModuleOp module, ContractOp contract, VerificationStage stage);

  /// Create a helper that combines compute- and constrain-stage target checks.
  FailureOr<func::FuncOp>
  createCombinedTargetHelper(ModuleOp module, ContractOp contract, StringRef helperName);

  /// Create a helper that conjoins the results of two boolean helpers.
  FailureOr<func::FuncOp> createCombinedBoolHelper(
      ModuleOp module, ContractOp contract, StringRef helperName, StringRef lhsHelperName,
      StringRef rhsHelperName
  );

  /// Lower one include operand list using the contract-local lowering environment.
  FailureOr<SmallVector<Value>> lowerIncludeHelperOperands(
      ValueRange operands, Operation *origin, OpBuilder &builder, DenseMap<Value, Value> &valueMap,
      SelfMemberMap &selfMemberMap, DenseMap<StringRef, Value> &constParamMap
  );

  /// Return enclosing value-like template params in declaration order.
  SmallVector<polymorphic::TemplateParamOp> getValueTemplateParams(Operation *origin);

  /// Lower template parameter attributes to SMT ints for a templated callee.
  FailureOr<SmallVector<Value>> lowerTemplateParamOperands(
      Operation *origin, ArrayAttr templateParams, OpBuilder &builder,
      DenseMap<StringRef, Value> &constParamMap
  );

  /// Seed hidden const-parameter bindings from the signature suffix.
  DenseMap<StringRef, Value>
  seedConstParamMap(ArrayRef<polymorphic::TemplateParamOp> params, ValueRange values);

  /// Create a fresh SMT symbol base name.
  std::string getFreshName(StringRef prefix) {
    return (Twine(prefix) + "_" + Twine(nextFreshId++)).str();
  }

  /// Build a stable helper-name stem from a symbol's fully qualified path.
  std::string getHelperStem(SymbolRefAttr path) {
    std::string stem;
    llvm::raw_string_ostream os(stem);
    bool first = true;
    for (StringRef piece : getNames(path)) {
      if (!first) {
        os << "__";
      }
      first = false;
      os << piece;
    }
    return stem;
  }

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

  /// Whether to emit legacy combined helpers for SMTLIB export compatibility.
  bool emitLegacyEntries = false;

  /// Monotonic suffix for fresh SMT witness names.
  unsigned nextFreshId = 0;
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

/// Return enclosing non-type template parameters that are materialized as SMT ints.
SmallVector<polymorphic::TemplateParamOp>
LoweringContext::getValueTemplateParams(Operation *origin) {
  SmallVector<polymorphic::TemplateParamOp> params;
  auto parentTemplate = origin->getParentOfType<polymorphic::TemplateOp>();
  if (!parentTemplate) {
    return params;
  }

  for (polymorphic::TemplateParamOp param :
       parentTemplate.getConstOps<polymorphic::TemplateParamOp>()) {
    auto typeOpt = param.getTypeOpt();
    if (typeOpt && isa<polymorphic::TypeVarType>(*typeOpt)) {
      continue;
    }
    params.push_back(param);
  }
  return params;
}

LogicalResult LoweringContext::appendLoweredBoundaryTypes(
    Type type, Operation *origin, SmallVectorImpl<Type> &loweredTypes
) {
  if (auto structType = dyn_cast<StructType>(type)) {
    auto members = getStructMembers(structType, origin);
    if (failed(members)) {
      return failure();
    }
    for (MemberDefOp member : *members) {
      if (failed(appendLoweredBoundaryTypes(member.getType(), origin, loweredTypes))) {
        return failure();
      }
    }
    return success();
  }

  loweredTypes.push_back(lowerType(type));
  return success();
}

FailureOr<unsigned> LoweringContext::getNumLoweredBoundaryValues(Type type, Operation *origin) {
  SmallVector<Type> loweredTypes;
  if (failed(appendLoweredBoundaryTypes(type, origin, loweredTypes))) {
    return failure();
  }
  return loweredTypes.size();
}

LogicalResult LoweringContext::seedFlattenedStructValue(
    Value aggregate, StructType structType, ArrayRef<Value> loweredValues, Operation *origin,
    SelfMemberMap &selfMemberMap
) {
  auto members = getStructMembers(structType, origin);
  if (failed(members)) {
    return failure();
  }

  DenseMap<StringRef, FlattenedMemberValues> loweredMembers;
  unsigned nextValue = 0;
  for (MemberDefOp member : *members) {
    auto numLoweredValues = getNumLoweredBoundaryValues(member.getType(), origin);
    if (failed(numLoweredValues)) {
      return failure();
    }
    if (nextValue + *numLoweredValues > loweredValues.size()) {
      emitError(
          aggregate.getLoc(), "insufficient flattened values while seeding struct lowering map"
      );
      return failure();
    }

    FlattenedMemberValues memberValues;
    memberValues.reserve(*numLoweredValues);
    llvm::append_range(memberValues, loweredValues.slice(nextValue, *numLoweredValues));
    loweredMembers[member.getSymName()] = std::move(memberValues);
    nextValue += *numLoweredValues;
  }

  if (nextValue != loweredValues.size()) {
    emitError(aggregate.getLoc(), "unused flattened values while seeding struct lowering map");
    return failure();
  }

  selfMemberMap[aggregate] = std::move(loweredMembers);
  return success();
}

LogicalResult LoweringContext::ensureFlattenedStructMapping(
    Value aggregate, Operation *origin, SelfMemberMap &selfMemberMap
) {
  if (selfMemberMap.contains(aggregate)) {
    return success();
  }

  auto structType = dyn_cast<StructType>(aggregate.getType());
  if (!structType) {
    emitError(aggregate.getLoc(), "expected struct-typed aggregate while lowering to SMT");
    return failure();
  }

  auto memberRead = dyn_cast_or_null<MemberReadOp>(aggregate.getDefiningOp());
  if (!memberRead) {
    emitError(
        aggregate.getLoc(), "missing flattened struct mapping while lowering aggregate value"
    );
    return failure();
  }

  if (failed(ensureFlattenedStructMapping(memberRead.getComponent(), origin, selfMemberMap))) {
    return failure();
  }
  auto parentIt = selfMemberMap.find(memberRead.getComponent());
  if (parentIt == selfMemberMap.end()) {
    emitError(aggregate.getLoc(), "missing flattened parent struct mapping while lowering to SMT");
    return failure();
  }

  auto loweredMemberIt = parentIt->second.find(memberRead.getMemberName());
  if (loweredMemberIt == parentIt->second.end()) {
    emitError(
        aggregate.getLoc(), (Twine("missing lowered member @") + memberRead.getMemberName() +
                             " while materializing nested struct mapping")
                                .str()
    );
    return failure();
  }

  return seedFlattenedStructValue(
      aggregate, structType, loweredMemberIt->second, origin, selfMemberMap
  );
}

/// Bind lowered hidden template arguments to their source parameter names.
DenseMap<StringRef, Value> LoweringContext::seedConstParamMap(
    ArrayRef<polymorphic::TemplateParamOp> params, ValueRange values
) {
  DenseMap<StringRef, Value> bindings;
  if (values.size() < params.size()) {
    return bindings;
  }
  for (auto [index, value] : llvm::enumerate(values)) {
    if (index >= params.size()) {
      break;
    }
    auto param = params[index];
    bindings[param.getNameAttr().getValue()] = value;
  }
  return bindings;
}

/// Lower concrete or inherited template arguments for one call-like operation.
FailureOr<SmallVector<Value>> LoweringContext::lowerTemplateParamOperands(
    Operation *origin, ArrayAttr templateParams, OpBuilder &builder,
    DenseMap<StringRef, Value> &constParamMap
) {
  SmallVector<polymorphic::TemplateParamOp> params = getValueTemplateParams(origin);
  if (params.empty()) {
    return SmallVector<Value> {};
  }

  if (!templateParams) {
    SmallVector<Value> inherited;
    inherited.reserve(params.size());
    for (polymorphic::TemplateParamOp param : params) {
      auto it = constParamMap.find(param.getName());
      if (it == constParamMap.end()) {
        origin->emitError().append(
            "missing hidden template parameter binding for @", param.getName(),
            " in verif-to-smt lowering"
        );
        return failure();
      }
      inherited.push_back(it->second);
    }
    return inherited;
  }

  auto parentTemplate = origin->getParentOfType<polymorphic::TemplateOp>();
  if (!parentTemplate) {
    origin->emitError("unexpected template parameter list on non-templated callee");
    return failure();
  }
  auto allParams = llvm::to_vector(parentTemplate.getConstOps<polymorphic::TemplateParamOp>());
  if (templateParams.size() != allParams.size()) {
    origin->emitError("template parameter list size does not match callee template");
    return failure();
  }

  SmallVector<Value> lowered;
  lowered.reserve(params.size());
  for (auto [param, attr] : llvm::zip_equal(allParams, templateParams)) {
    auto typeOpt = param.getTypeOpt();
    if (typeOpt && isa<polymorphic::TypeVarType>(*typeOpt)) {
      continue;
    }

    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      lowered.push_back(builder.create<smt::IntConstantOp>(origin->getLoc(), intAttr).getResult());
      continue;
    }
    if (auto feltAttr = dyn_cast<FeltConstAttr>(attr)) {
      lowered.push_back(builder
                            .create<smt::IntConstantOp>(
                                origin->getLoc(), IntegerAttr::get(
                                                      builder.getContext(),
                                                      toAPSInt(toDynamicAPInt(feltAttr.getValue()))
                                                  )
                            )
                            .getResult());
      continue;
    }
    if (auto refAttr = dyn_cast<FlatSymbolRefAttr>(attr)) {
      auto it = constParamMap.find(refAttr.getRootReference().strref());
      if (it == constParamMap.end()) {
        origin->emitError().append(
            "missing template parameter binding for @", refAttr.getRootReference(),
            " in verif-to-smt lowering"
        );
        return failure();
      }
      lowered.push_back(it->second);
      continue;
    }

    origin->emitError("unsupported non-type template parameter in verif-to-smt lowering");
    return failure();
  }
  return lowered;
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
  origin->emitError() << "'llzk-scalar-verif-to-smt' pass requires array/pod-free IR for "
                      << description << "; run 'llzk-verif-to-smt' instead";
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
    if (failed(appendLoweredBoundaryTypes(input, origin, lowered.argTypes))) {
      return failure();
    }
  }
  for (Type result : type.getResults()) {
    if (failed(appendLoweredBoundaryTypes(result, origin, lowered.resultTypes))) {
      return failure();
    }
  }
  lowered.templateParams = getValueTemplateParams(origin);
  lowered.argTypes.reserve(lowered.argTypes.size() + lowered.templateParams.size());
  for (polymorphic::TemplateParamOp _ : lowered.templateParams) {
    lowered.argTypes.push_back(smt::IntType::get(context));
  }
  return lowered;
}

/// Seed lowered argument maps for callables whose SMT helper signatures flatten struct arguments.
FailureOr<unsigned> LoweringContext::seedCallableArgumentMaps(
    ValueRange originalArgs, Block *entry, Operation *origin, DenseMap<Value, Value> &valueMap,
    SelfMemberMap &selfMemberMap
) {
  unsigned nextArg = 0;
  for (Value originalArg : originalArgs) {
    if (auto structType = dyn_cast<StructType>(originalArg.getType())) {
      auto numLoweredValues = getNumLoweredBoundaryValues(structType, origin);
      if (failed(numLoweredValues)) {
        return failure();
      }
      SmallVector<Value> loweredValues;
      loweredValues.reserve(*numLoweredValues);
      for (unsigned index = 0; index < *numLoweredValues; ++index) {
        loweredValues.push_back(entry->getArgument(nextArg++));
      }
      if (failed(seedFlattenedStructValue(
              originalArg, structType, loweredValues, origin, selfMemberMap
          ))) {
        return failure();
      }
      continue;
    }
    valueMap[originalArg] = entry->getArgument(nextArg++);
  }
  return nextArg;
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

/// Return `true` if `op` is nested under an `scf.if` before reaching `scope`.
static bool hasIfAncestor(Operation *op, Operation *scope) {
  for (Operation *ancestor = op->getParentOp(); ancestor && ancestor != scope;
       ancestor = ancestor->getParentOp()) {
    if (isa<scf::IfOp>(ancestor)) {
      return true;
    }
  }
  return false;
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
      LoweringContext &l, OpBuilder &b, DenseMap<Value, Value> &v, SelfMemberMap &s,
      DenseMap<StringRef, Value> &c
  )
      : state(l), builder(b), valueMap(v), selfMemberMap(s), constParamMap(c) {}

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
      auto lowered = builder.create<smt::IntConstantOp>(feltConst.getLoc(), toIntAttr(feltConst));
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
          succeeded(state.ensureFlattenedStructMapping(
              memberRead.getComponent(), memberRead.getOperation(), selfMemberMap
          ))) {
        auto membersIt = selfMemberMap.find(memberRead.getComponent());
        if (membersIt != selfMemberMap.end()) {
          auto it = membersIt->second.find(memberRead.getMemberName());
          if (it != membersIt->second.end()) {
            if (it->second.size() != 1) {
              if (isa<StructType>(memberRead.getType())) {
                (void)state.seedFlattenedStructValue(
                    value, cast<StructType>(memberRead.getType()), it->second,
                    memberRead.getOperation(), selfMemberMap
                );
              }
              emitError(
                  value.getLoc(), "expected scalar lowered member but found nested aggregate"
              );
              return failure();
            }
            valueMap[value] = it->second.front();
            return it->second.front();
          }
        }
      }
    }

    if (auto constRead = dyn_cast<polymorphic::ConstReadOp>(definingOp)) {
      if (!isa<FeltType>(constRead.getType())) {
        constRead.emitError(
            "only felt-typed poly.read_const is supported in verif-to-smt lowering"
        );
        return failure();
      }
      auto it = constParamMap.find(constRead.getConstName());
      if (it == constParamMap.end()) {
        constRead.emitError().append(
            "missing hidden binding for template parameter @", constRead.getConstName()
        );
        return failure();
      }
      valueMap[value] = it->second;
      return it->second;
    }

    if (auto unary = dyn_cast<NegFeltOp>(definingOp)) {
      auto operand = lower(unary.getOperand());
      if (failed(operand)) {
        return failure();
      }
      auto lowered = builder.create<smt::IntNegOp>(unary.getLoc(), *operand).getResult();
      valueMap[value] = lowered;
      return lowered;
    }

    if (auto unary = dyn_cast<NotFeltOp>(definingOp)) {
      auto operand = lower(unary.getOperand());
      if (failed(operand)) {
        return failure();
      }
      FeltType type = cast<FeltType>(unary.getType());
      Value bv = emitCanonicalIntToBV(unary.getLoc(), *operand, type);
      Value notValue = builder.create<smt::BVNotOp>(unary.getLoc(), bv).getResult();
      Value lowered = emitBVToCanonicalInt(unary.getLoc(), notValue, type);
      valueMap[value] = lowered;
      return lowered;
    }

    if (auto inv = dyn_cast<InvFeltOp>(definingOp)) {
      auto operand = lower(inv.getOperand());
      if (failed(operand)) {
        return failure();
      }
      Value lowered = emitInverseValue(inv.getLoc(), *operand, cast<FeltType>(inv.getType()));
      valueMap[value] = lowered;
      return lowered;
    }

    if (auto add = dyn_cast<AddFeltOp>(definingOp)) {
      auto lowered = lowerCanonicalBinaryExpr(
          add.getLoc(), cast<FeltType>(add.getType()), add.getLhs(), add.getRhs(),
          [&](Value lhs, Value rhs) {
        Value sum = builder.create<smt::IntAddOp>(add.getLoc(), ValueRange {lhs, rhs}).getResult();
        return emitCanonical(add.getLoc(), sum, cast<FeltType>(add.getType()));
      }
      );
      if (failed(lowered)) {
        return failure();
      }
      valueMap[value] = *lowered;
      return *lowered;
    }

    if (auto sub = dyn_cast<SubFeltOp>(definingOp)) {
      auto lowered = lowerCanonicalBinaryExpr(
          sub.getLoc(), cast<FeltType>(sub.getType()), sub.getLhs(), sub.getRhs(),
          [&](Value lhs, Value rhs) {
        Value diff = builder.create<smt::IntSubOp>(sub.getLoc(), lhs, rhs).getResult();
        return emitCanonical(sub.getLoc(), diff, cast<FeltType>(sub.getType()));
      }
      );
      if (failed(lowered)) {
        return failure();
      }
      valueMap[value] = *lowered;
      return *lowered;
    }

    if (auto mul = dyn_cast<MulFeltOp>(definingOp)) {
      auto lowered = lowerCanonicalBinaryExpr(
          mul.getLoc(), cast<FeltType>(mul.getType()), mul.getLhs(), mul.getRhs(),
          [&](Value lhs, Value rhs) {
        Value product =
            builder.create<smt::IntMulOp>(mul.getLoc(), ValueRange {lhs, rhs}).getResult();
        return emitCanonical(mul.getLoc(), product, cast<FeltType>(mul.getType()));
      }
      );
      if (failed(lowered)) {
        return failure();
      }
      valueMap[value] = *lowered;
      return *lowered;
    }

    if (auto udiv = dyn_cast<UnsignedIntDivFeltOp>(definingOp)) {
      auto lowered = lowerCanonicalBinaryExpr(
          udiv.getLoc(), cast<FeltType>(udiv.getType()), udiv.getLhs(), udiv.getRhs(),
          [&](Value lhs, Value rhs) {
        return builder.create<smt::IntDivOp>(udiv.getLoc(), lhs, rhs).getResult();
      }
      );
      if (failed(lowered)) {
        return failure();
      }
      valueMap[value] = *lowered;
      return *lowered;
    }

    if (auto umod = dyn_cast<UnsignedModFeltOp>(definingOp)) {
      auto lowered = lowerCanonicalBinaryExpr(
          umod.getLoc(), cast<FeltType>(umod.getType()), umod.getLhs(), umod.getRhs(),
          [&](Value lhs, Value rhs) {
        return builder.create<smt::IntModOp>(umod.getLoc(), lhs, rhs).getResult();
      }
      );
      if (failed(lowered)) {
        return failure();
      }
      valueMap[value] = *lowered;
      return *lowered;
    }

    if (auto sdiv = dyn_cast<SignedIntDivFeltOp>(definingOp)) {
      auto lhs = lower(sdiv.getLhs());
      auto rhs = lower(sdiv.getRhs());
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      Value lowered =
          emitSignedIntDivValue(sdiv.getLoc(), *lhs, *rhs, cast<FeltType>(sdiv.getType()));
      valueMap[value] = lowered;
      return lowered;
    }

    if (auto smod = dyn_cast<SignedModFeltOp>(definingOp)) {
      auto lhs = lower(smod.getLhs());
      auto rhs = lower(smod.getRhs());
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      Value lowered = emitSignedModValue(smod.getLoc(), *lhs, *rhs, cast<FeltType>(smod.getType()));
      valueMap[value] = lowered;
      return lowered;
    }

    if (auto shl = dyn_cast<ShlFeltOp>(definingOp)) {
      auto lowered = lowerBitvectorBinaryExpr(
          shl.getLoc(), cast<FeltType>(shl.getType()), shl.getLhs(), shl.getRhs(),
          [&](Value lhs, Value rhs) {
        return builder.create<smt::BVShlOp>(shl.getLoc(), lhs, rhs).getResult();
      }
      );
      if (failed(lowered)) {
        return failure();
      }
      valueMap[value] = *lowered;
      return *lowered;
    }

    if (auto shr = dyn_cast<ShrFeltOp>(definingOp)) {
      auto lowered = lowerBitvectorBinaryExpr(
          shr.getLoc(), cast<FeltType>(shr.getType()), shr.getLhs(), shr.getRhs(),
          [&](Value lhs, Value rhs) {
        return builder.create<smt::BVLShrOp>(shr.getLoc(), lhs, rhs).getResult();
      }
      );
      if (failed(lowered)) {
        return failure();
      }
      valueMap[value] = *lowered;
      return *lowered;
    }

    if (auto andOp = dyn_cast<AndFeltOp>(definingOp)) {
      auto lowered = lowerBitvectorBinaryExpr(
          andOp.getLoc(), cast<FeltType>(andOp.getType()), andOp.getLhs(), andOp.getRhs(),
          [&](Value lhs, Value rhs) {
        return builder.create<smt::BVAndOp>(andOp.getLoc(), lhs, rhs).getResult();
      }
      );
      if (failed(lowered)) {
        return failure();
      }
      valueMap[value] = *lowered;
      return *lowered;
    }

    if (auto orOp = dyn_cast<OrFeltOp>(definingOp)) {
      auto lowered = lowerBitvectorBinaryExpr(
          orOp.getLoc(), cast<FeltType>(orOp.getType()), orOp.getLhs(), orOp.getRhs(),
          [&](Value lhs, Value rhs) {
        return builder.create<smt::BVOrOp>(orOp.getLoc(), lhs, rhs).getResult();
      }
      );
      if (failed(lowered)) {
        return failure();
      }
      valueMap[value] = *lowered;
      return *lowered;
    }

    if (auto xorOp = dyn_cast<XorFeltOp>(definingOp)) {
      auto lowered = lowerBitvectorBinaryExpr(
          xorOp.getLoc(), cast<FeltType>(xorOp.getType()), xorOp.getLhs(), xorOp.getRhs(),
          [&](Value lhs, Value rhs) {
        return builder.create<smt::BVXOrOp>(xorOp.getLoc(), lhs, rhs).getResult();
      }
      );
      if (failed(lowered)) {
        return failure();
      }
      valueMap[value] = *lowered;
      return *lowered;
    }

    if (auto powOp = dyn_cast<PowFeltOp>(definingOp)) {
      auto lhs = lower(powOp.getLhs());
      auto rhs = lower(powOp.getRhs());
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      auto lowered = emitPowValue(powOp, *lhs, *rhs, cast<FeltType>(powOp.getType()));
      if (failed(lowered)) {
        return failure();
      }
      valueMap[value] = *lowered;
      return *lowered;
    }

    if (auto div = dyn_cast<DivFeltOp>(definingOp)) {
      auto lhs = lower(div.getLhs());
      auto rhs = lower(div.getRhs());
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      Value lowered = emitDivisionValue(div.getLoc(), *lhs, *rhs, cast<FeltType>(div.getType()));
      valueMap[value] = lowered;
      return lowered;
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

    if (auto ifOp = dyn_cast<scf::IfOp>(definingOp)) {
      if (failed(validateIfShape(ifOp, /*requireElseRegion=*/true, "scf.if result"))) {
        return failure();
      }

      auto loweredCond = lower(ifOp.getCondition());
      if (failed(loweredCond)) {
        return failure();
      }

      auto resultNumber = cast<OpResult>(value).getResultNumber();
      auto thenYield = cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
      auto elseYield = cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());

      auto loweredThen = lower(thenYield.getOperand(resultNumber));
      auto loweredElse = lower(elseYield.getOperand(resultNumber));
      if (failed(loweredThen) || failed(loweredElse)) {
        return failure();
      }

      auto ite = builder.create<smt::IteOp>(ifOp.getLoc(), *loweredCond, *loweredThen, *loweredElse)
                     .getResult();
      valueMap[value] = ite;
      return ite;
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
      loweredOperands.reserve(call.getArgOperands().size());
      for (Value operand : call.getArgOperands()) {
        auto loweredOperand = lower(operand);
        if (failed(loweredOperand)) {
          return failure();
        }
        loweredOperands.push_back(*loweredOperand);
      }
      auto loweredTemplateParams = state.lowerTemplateParamOperands(
          calleeTarget->get().getOperation(), call.getTemplateParamsAttr(), builder, constParamMap
      );
      if (failed(loweredTemplateParams)) {
        return failure();
      }
      llvm::append_range(loweredOperands, *loweredTemplateParams);

      auto loweredCall = builder.create<func::CallOp>(
          call.getLoc(), funcHelperRes->getSymName(), funcHelperRes->getFunctionType().getResults(),
          loweredOperands
      );
      for (auto [orig, lowered] : llvm::zip(call.getResults(), loweredCall.getResults())) {
        valueMap[orig] = lowered;
      }
      return loweredCall.getResult(cast<OpResult>(value).getResultNumber());
    }

    definingOp->emitError().append(
        "unsupported expression in verif-to-smt lowering: ", definingOp->getName().getStringRef()
    );
    return failure();
  }

private:
  /// Build a stable cache key for an SMT integer literal.
  static std::string getIntConstantKey(const llvm::DynamicAPInt &value) {
    llvm::SmallString<64> repr;
    llvm::raw_svector_ostream(repr) << value;
    return std::string(repr);
  }

  /// Convert a felt constant attribute into the equivalent SMT integer literal.
  IntegerAttr toIntAttr(FeltConstantOp op) {
    return IntegerAttr::get(
        builder.getContext(), toAPSInt(toDynamicAPInt(op.getValue().getValue()))
    );
  }

  /// Materialize an SMT integer constant from a field element-sized APInt value.
  Value createIntConstant(Location loc, const llvm::DynamicAPInt &value) {
    std::string key = getIntConstantKey(value);
    if (auto it = intConstantCache.find(key); it != intConstantCache.end()) {
      return it->second;
    }

    Value constant = builder
                         .create<smt::IntConstantOp>(
                             loc, IntegerAttr::get(builder.getContext(), toAPSInt(value))
                         )
                         .getResult();
    intConstantCache.try_emplace(std::move(key), constant);
    return constant;
  }

  /// Materialize the prime modulus of `field` as an SMT integer constant.
  Value emitPrimeConstant(Location loc, const Field &field) {
    return createIntConstant(loc, field.prime());
  }

  /// Return the constant integer value carried by an SMT integer literal, if known.
  std::optional<llvm::DynamicAPInt> getKnownIntegerValue(Value value) {
    if (auto intConst = value.getDefiningOp<smt::IntConstantOp>()) {
      return toDynamicAPInt(intConst.getValue());
    }
    return std::nullopt;
  }

  /// Canonicalize an integer-valued felt expression into the field range.
  Value emitCanonical(Location loc, Value value, FeltType type) {
    return builder.create<smt::IntModOp>(loc, value, emitPrimeConstant(loc, type.getField()))
        .getResult();
  }

  /// Convert a canonical field element into its signed representative.
  Value emitSignedRepresentative(Location loc, Value value, FeltType type) {
    const Field &field = type.getField();
    Value canonical = emitCanonical(loc, value, type);
    Value threshold = createIntConstant(loc, field.half());
    Value isNonNegative =
        builder.create<smt::IntCmpOp>(loc, smt::IntPredicate::lt, canonical, threshold).getResult();
    Value negative =
        builder.create<smt::IntSubOp>(loc, canonical, emitPrimeConstant(loc, field)).getResult();
    return builder.create<smt::IteOp>(loc, isNonNegative, canonical, negative).getResult();
  }

  /// Convert a canonicalized felt integer to a fixed-width SMT bitvector.
  Value emitCanonicalIntToBV(Location loc, Value value, FeltType type) {
    Value canonical = emitCanonical(loc, value, type);
    return builder
        .create<smt::Int2BVOp>(
            loc, smt::BitVectorType::get(builder.getContext(), type.getField().bitWidth()),
            canonical
        )
        .getResult();
  }

  /// Convert an SMT bitvector result back to a canonical felt integer.
  Value emitBVToCanonicalInt(Location loc, Value value, FeltType type) {
    Value intValue = builder.create<smt::BV2IntOp>(loc, value, UnitAttr()).getResult();
    return emitCanonical(loc, intValue, type);
  }

  /// Lower both operands of an integer-valued binary expression and invoke `fn`.
  template <typename Fn>
  FailureOr<Value> lowerBinaryIntExpr(Location loc, Value lhsValue, Value rhsValue, Fn &&fn) {
    auto lhs = lower(lhsValue);
    auto rhs = lower(rhsValue);
    if (failed(lhs) || failed(rhs)) {
      return failure();
    }
    return fn(*lhs, *rhs);
  }

  /// Lower and canonicalize both operands of a field arithmetic binary expression.
  template <typename Fn>
  FailureOr<Value>
  lowerCanonicalBinaryExpr(Location loc, FeltType type, Value lhsValue, Value rhsValue, Fn &&fn) {
    auto lhs = lower(lhsValue);
    auto rhs = lower(rhsValue);
    if (failed(lhs) || failed(rhs)) {
      return failure();
    }
    return fn(emitCanonical(loc, *lhs, type), emitCanonical(loc, *rhs, type));
  }

  /// Lower both operands through bitvector conversion and translate the result back.
  template <typename Fn>
  FailureOr<Value>
  lowerBitvectorBinaryExpr(Location loc, FeltType type, Value lhsValue, Value rhsValue, Fn &&fn) {
    auto lhs = lower(lhsValue);
    auto rhs = lower(rhsValue);
    if (failed(lhs) || failed(rhs)) {
      return failure();
    }
    Value result = fn(emitCanonicalIntToBV(loc, *lhs, type), emitCanonicalIntToBV(loc, *rhs, type));
    return emitBVToCanonicalInt(loc, result, type);
  }

  /// Return the absolute value of a signed SMT integer.
  Value emitAbsValue(Location loc, Value value) {
    Value zero = builder.create<smt::IntConstantOp>(loc, builder.getI64IntegerAttr(0)).getResult();
    Value isNegative =
        builder.create<smt::IntCmpOp>(loc, smt::IntPredicate::lt, value, zero).getResult();
    Value negated = builder.create<smt::IntNegOp>(loc, value).getResult();
    return builder.create<smt::IteOp>(loc, isNegative, negated, value).getResult();
  }

  /// Lower signed division over SMT integers with truncation toward zero.
  Value emitTruncatingSignedDivision(Location loc, Value lhs, Value rhs) {
    Value zero = builder.create<smt::IntConstantOp>(loc, builder.getI64IntegerAttr(0)).getResult();
    Value lhsNeg = builder.create<smt::IntCmpOp>(loc, smt::IntPredicate::lt, lhs, zero).getResult();
    Value rhsNeg = builder.create<smt::IntCmpOp>(loc, smt::IntPredicate::lt, rhs, zero).getResult();
    Value lhsAbs = emitAbsValue(loc, lhs);
    Value rhsAbs = emitAbsValue(loc, rhs);
    Value absQuotient = builder.create<smt::IntDivOp>(loc, lhsAbs, rhsAbs).getResult();
    Value signsDiffer = builder.create<smt::XOrOp>(loc, ValueRange {lhsNeg, rhsNeg}).getResult();
    Value negatedQuotient = builder.create<smt::IntNegOp>(loc, absQuotient).getResult();
    return builder.create<smt::IteOp>(loc, signsDiffer, negatedQuotient, absQuotient).getResult();
  }

  /// Lower signed division or remainder over field elements via signed representatives.
  Value emitSignedDivOrRem(Location loc, Value lhs, Value rhs, FeltType type, bool isDiv) {
    Value signedLhs = emitSignedRepresentative(loc, lhs, type);
    Value signedRhs = emitSignedRepresentative(loc, rhs, type);
    Value quotient = emitTruncatingSignedDivision(loc, signedLhs, signedRhs);
    if (isDiv) {
      return emitCanonical(loc, quotient, type);
    }
    Value product =
        builder.create<smt::IntMulOp>(loc, ValueRange {signedRhs, quotient}).getResult();
    Value remainder = builder.create<smt::IntSubOp>(loc, signedLhs, product).getResult();
    return emitCanonical(loc, remainder, type);
  }

  /// Lower signed integer division over field elements.
  Value emitSignedIntDivValue(Location loc, Value lhs, Value rhs, FeltType type) {
    return emitSignedDivOrRem(loc, lhs, rhs, type, /*isDiv=*/true);
  }

  /// Lower signed remainder over field elements.
  Value emitSignedModValue(Location loc, Value lhs, Value rhs, FeltType type) {
    return emitSignedDivOrRem(loc, lhs, rhs, type, /*isDiv=*/false);
  }

  /// Lower exponentiation by repeated squaring over canonical field elements.
  FailureOr<Value> emitPowValue(PowFeltOp powOp, Value base, Value exponent, FeltType type) {
    Location loc = powOp.getLoc();
    const Field &field = type.getField();
    Value canonicalBase = emitCanonical(loc, base, type);
    Value canonicalExponent = emitCanonical(loc, exponent, type);

    auto buildBitSet = [&](unsigned bit) -> Value {
      Value divisor = createIntConstant(loc, llvm::DynamicAPInt(1) << llvm::DynamicAPInt(bit));
      Value shifted = builder.create<smt::IntDivOp>(loc, canonicalExponent, divisor).getResult();
      Value lsb =
          builder.create<smt::IntModOp>(loc, shifted, createIntConstant(loc, llvm::DynamicAPInt(2)))
              .getResult();
      return builder.create<smt::EqOp>(loc, lsb, createIntConstant(loc, llvm::DynamicAPInt(1)))
          .getResult();
    };

    auto knownExponent = getKnownIntegerValue(exponent);
    if (knownExponent) {
      llvm::DynamicAPInt reducedExponent = field.reduce(*knownExponent);
      Value acc = createIntConstant(loc, llvm::DynamicAPInt(1));
      Value curPow = canonicalBase;
      llvm::APInt exponentBits = toAPInt(reducedExponent, field.bitWidth());
      for (unsigned bit = 0; bit < exponentBits.getActiveBits(); ++bit) {
        if (!exponentBits[bit]) {
          curPow = emitCanonical(
              loc, builder.create<smt::IntMulOp>(loc, ValueRange {curPow, curPow}).getResult(), type
          );
          continue;
        }
        acc = emitCanonical(
            loc, builder.create<smt::IntMulOp>(loc, ValueRange {acc, curPow}).getResult(), type
        );
        curPow = emitCanonical(
            loc, builder.create<smt::IntMulOp>(loc, ValueRange {curPow, curPow}).getResult(), type
        );
      }
      return acc;
    }

    auto knownBase = getKnownIntegerValue(base);
    if (knownBase) {
      llvm::DynamicAPInt reducedBase = field.reduce(*knownBase);
      llvm::DynamicAPInt zero(0);
      llvm::DynamicAPInt one(1);
      llvm::DynamicAPInt minusOne = field.reduce(-llvm::DynamicAPInt(1));

      if (reducedBase == zero) {
        Value exponentIsZero =
            builder.create<smt::EqOp>(loc, canonicalExponent, createIntConstant(loc, zero))
                .getResult();
        return builder
            .create<smt::IteOp>(
                loc, exponentIsZero, createIntConstant(loc, one), createIntConstant(loc, zero)
            )
            .getResult();
      }
      if (reducedBase == one) {
        return createIntConstant(loc, one);
      }
      if (reducedBase == minusOne) {
        Value parity = builder
                           .create<smt::IntModOp>(
                               loc, canonicalExponent, createIntConstant(loc, llvm::DynamicAPInt(2))
                           )
                           .getResult();
        Value isOdd =
            builder.create<smt::EqOp>(loc, parity, createIntConstant(loc, one)).getResult();
        return builder
            .create<smt::IteOp>(
                loc, isOdd, createIntConstant(loc, minusOne), createIntConstant(loc, one)
            )
            .getResult();
      }

      Value acc = createIntConstant(loc, one);
      llvm::DynamicAPInt curPow = reducedBase;
      for (unsigned bit = 0; bit < field.bitWidth(); ++bit) {
        Value bitSet = buildBitSet(bit);
        Value curPowConst = createIntConstant(loc, curPow);
        Value multiplied = emitCanonical(
            loc, builder.create<smt::IntMulOp>(loc, ValueRange {acc, curPowConst}).getResult(), type
        );
        acc = builder.create<smt::IteOp>(loc, bitSet, multiplied, acc).getResult();
        curPow = field.reduce(curPow * curPow);
      }
      return acc;
    }

    powOp.emitError(
        "verif-to-smt does not support fully symbolic felt.pow; require a constant base or "
        "constant exponent"
    );
    return failure();
  }

  /// Lower division by introducing a fresh witness constrained as the quotient.
  Value emitDivisionValue(Location loc, Value lhs, Value rhs, FeltType type) {
    Value numerator = emitCanonical(loc, lhs, type);
    Value denominator = emitCanonical(loc, rhs, type);
    auto fresh = builder.create<smt::DeclareFunOp>(
        loc, smt::IntType::get(builder.getContext()),
        StringAttr::get(builder.getContext(), state.getFreshName("felt_div"))
    );
    Value zero = createIntConstant(loc, llvm::DynamicAPInt(0));
    Value denominatorIsZero = builder.create<smt::EqOp>(loc, denominator, zero).getResult();
    Value divIsZero = builder.create<smt::EqOp>(loc, fresh.getResult(), zero).getResult();
    Value product =
        builder.create<smt::IntMulOp>(loc, ValueRange {denominator, fresh.getResult()}).getResult();
    Value productMod = emitCanonical(loc, product, type);
    Value productEqualsNumerator =
        builder.create<smt::EqOp>(loc, productMod, numerator).getResult();
    Value constraint =
        builder.create<smt::IteOp>(loc, denominatorIsZero, divIsZero, productEqualsNumerator)
            .getResult();
    builder.create<smt::AssertOp>(loc, constraint);
    return fresh.getResult();
  }

  /// Lower inversion by introducing a fresh witness constrained as the inverse.
  Value emitInverseValue(Location loc, Value operand, FeltType type) {
    Value canonicalOperand = emitCanonical(loc, operand, type);
    auto fresh = builder.create<smt::DeclareFunOp>(
        loc, smt::IntType::get(builder.getContext()),
        StringAttr::get(builder.getContext(), state.getFreshName("felt_inv"))
    );
    Value zero = createIntConstant(loc, llvm::DynamicAPInt(0));
    Value one = createIntConstant(loc, llvm::DynamicAPInt(1));
    Value operandIsZero = builder.create<smt::EqOp>(loc, canonicalOperand, zero).getResult();
    Value invIsZero = builder.create<smt::EqOp>(loc, fresh.getResult(), zero).getResult();
    Value product =
        builder.create<smt::IntMulOp>(loc, ValueRange {canonicalOperand, fresh.getResult()})
            .getResult();
    Value productEqualsOne =
        builder.create<smt::EqOp>(loc, emitCanonical(loc, product, type), one).getResult();
    Value constraint =
        builder.create<smt::IteOp>(loc, operandIsZero, invIsZero, productEqualsOne).getResult();
    builder.create<smt::AssertOp>(loc, constraint);
    return fresh.getResult();
  }

  /// Shared pass state.
  LoweringContext &state;

  /// Rewriter used to create lowered operations.
  OpBuilder &builder;

  /// Lowered SSA value cache for ordinary arguments and derived values.
  DenseMap<Value, Value> &valueMap;

  /// Flattened `%self` member bindings keyed by member name.
  SelfMemberMap &selfMemberMap;

  /// Hidden template-parameter bindings keyed by parameter name.
  DenseMap<StringRef, Value> &constParamMap;

  /// Per-helper cache of emitted SMT integer literals.
  llvm::StringMap<Value> intConstantCache;
};

/// Guard a contract-local condition with enclosing `scf.if` control flow.
static FailureOr<Value>
lowerGuardedCondition(Operation *conditionOp, ExprLowerer &lowerer, OpBuilder &builder) {
  auto contractCond = cast<ConditionOpInterface>(conditionOp);
  auto lowered = lowerer.lower(contractCond.getCondition());
  if (failed(lowered)) {
    return failure();
  }

  Value guarded = *lowered;
  for (Operation *ancestor = conditionOp->getParentOp(); ancestor;
       ancestor = ancestor->getParentOp()) {
    auto ifOp = dyn_cast<scf::IfOp>(ancestor);
    if (!ifOp) {
      continue;
    }
    if (failed(validateIfShape(ifOp, /*requireElseRegion=*/false, "contract condition"))) {
      return failure();
    }

    auto loweredCond = lowerer.lower(ifOp.getCondition());
    if (failed(loweredCond)) {
      return failure();
    }

    auto inThen = isInThenRegion(ifOp, conditionOp);
    if (failed(inThen)) {
      return failure();
    }
    Value antecedent = *loweredCond;
    if (!*inThen) {
      antecedent = builder.create<smt::NotOp>(ifOp.getLoc(), antecedent).getResult();
    }
    guarded =
        builder.create<smt::ImpliesOp>(conditionOp->getLoc(), antecedent, guarded).getResult();
  }

  return guarded;
}

/// Guard an already-lowered boolean with enclosing `scf.if` conditions.
static FailureOr<Value> lowerGuardedBoolean(
    Operation *op, Value condition, ExprLowerer &lowerer, OpBuilder &builder, StringRef context
) {
  Value guarded = condition;
  for (Operation *ancestor = op->getParentOp(); ancestor; ancestor = ancestor->getParentOp()) {
    auto ifOp = dyn_cast<scf::IfOp>(ancestor);
    if (!ifOp) {
      continue;
    }
    if (failed(validateIfShape(ifOp, /*requireElseRegion=*/false, context))) {
      return failure();
    }

    auto loweredCond = lowerer.lower(ifOp.getCondition());
    if (failed(loweredCond)) {
      return failure();
    }

    auto inThen = isInThenRegion(ifOp, op);
    if (failed(inThen)) {
      return failure();
    }
    Value antecedent = *loweredCond;
    if (!*inThen) {
      antecedent = builder.create<smt::NotOp>(ifOp.getLoc(), antecedent).getResult();
    }
    guarded = builder.create<smt::ImpliesOp>(op->getLoc(), antecedent, guarded).getResult();
  }
  return guarded;
}

/// Reject struct constrain bodies that cannot be summarized as one boolean formula.
static LogicalResult checkStructConstrainBodyIsBooleanSummarizable(FuncDefOp constrainFunc) {
  bool failed = false;
  constrainFunc.walk([&](Operation *op) {
    if (op == constrainFunc.getOperation() || isa<ReturnOp, scf::YieldOp>(op) ||
        isa<constrain::EmitEqualityOp>(op) || isa<CallOp>(op)) {
      return WalkResult::advance();
    }
    if (isa<scf::IfOp>(op)) {
      return WalkResult::advance();
    }
    if (op->getNumRegions() != 0 || op->getNumSuccessors() != 0) {
      op->emitError(
      ) << "verif-to-smt requires struct constrain bodies without loops or successor-based "
           "control flow; run `llzk-flatten` or another control-flow lowering pass first";
      failed = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(failed);
}

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
  std::string helperName = "smt_" + getHelperStem(func.getFullyQualifiedName());
  auto helper = moduleBuilder.create<func::FuncOp>(
      func.getLoc(), helperName,
      FunctionType::get(context, loweredSig->argTypes, loweredSig->resultTypes)
  );

  Block *entry = helper.addEntryBlock();
  OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);
  DenseMap<Value, Value> valueMap;
  SelfMemberMap selfMemberMap;
  auto numLoweredInputs =
      seedCallableArgumentMaps(func.getArguments(), entry, func, valueMap, selfMemberMap);
  if (failed(numLoweredInputs)) {
    return failure();
  }
  DenseMap<StringRef, Value> constParamMap = seedConstParamMap(
      loweredSig->templateParams, ValueRange(entry->getArguments()).drop_front(*numLoweredInputs)
  );

  ExprLowerer lowerer(*this, entryBuilder, valueMap, selfMemberMap, constParamMap);
  auto returnOp = dyn_cast<ReturnOp>(func.getBody().front().getTerminator());
  if (!returnOp) {
    func.emitError("expected function.return terminator");
    return failure();
  }

  SmallVector<Value> returnedValues;
  for (Value operand : returnOp.getOperands()) {
    auto loweredOperand = lowerFlattenedValue(
        operand, returnOp.getOperation(), entryBuilder, valueMap, selfMemberMap, constParamMap
    );
    if (failed(loweredOperand)) {
      return failure();
    }
    llvm::append_range(returnedValues, *loweredOperand);
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
  for (MemberDefOp member : members) {
    if (failed(
            appendLoweredBoundaryTypes(member.getType(), structDef.getOperation(), memberTypes)
        )) {
      return failure();
    }
  }

  auto computeSig = lowerCallableSignature(computeFunc.getFunctionType(), computeFunc);
  if (failed(computeSig)) {
    return failure();
  }
  auto constrainSig = lowerCallableSignature(constrainFunc.getFunctionType(), constrainFunc);
  if (failed(constrainSig)) {
    return failure();
  }

  OpBuilder moduleBuilder = createModuleBuilder(module);

  std::string computeName = "smt_" + getHelperStem(structDef.getFullyQualifiedName()) + "_compute";
  auto computeHelper = moduleBuilder.create<func::FuncOp>(
      computeFunc.getLoc(), computeName,
      FunctionType::get(context, computeSig->argTypes, memberTypes)
  );
  {
    Block *entry = computeHelper.addEntryBlock();
    OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);
    DenseMap<Value, Value> valueMap;
    SelfMemberMap selfMemberMap;
    auto numLoweredInputs = seedCallableArgumentMaps(
        computeFunc.getArguments(), entry, computeFunc, valueMap, selfMemberMap
    );
    if (failed(numLoweredInputs)) {
      return failure();
    }
    DenseMap<StringRef, Value> constParamMap = seedConstParamMap(
        computeSig->templateParams, ValueRange(entry->getArguments()).drop_front(*numLoweredInputs)
    );

    ExprLowerer lowerer(*this, entryBuilder, valueMap, selfMemberMap, constParamMap);
    DenseMap<StringRef, SmallVector<Value>> writtenMembers;
    Value returnedSelf = computeFunc.getSelfValueFromCompute();
    bool failedToLower = false;
    computeFunc.walk([&](MemberWriteOp writeOp) {
      if (failedToLower) {
        return;
      }
      if (writeOp.getComponent() != returnedSelf) {
        return;
      }
      if (hasIfAncestor(writeOp.getOperation(), computeFunc.getOperation())) {
        computeFunc.emitError(
            "verif-to-smt requires compute writes to the returned component to be unguarded"
        );
        failedToLower = true;
        return;
      }
      auto loweredValue = lowerFlattenedValue(
          writeOp.getVal(), writeOp.getOperation(), entryBuilder, valueMap, selfMemberMap,
          constParamMap
      );
      if (failed(loweredValue)) {
        failedToLower = true;
        return;
      }
      if (writtenMembers.contains(writeOp.getMemberName())) {
        computeFunc.emitError().append(
            "verif-to-smt requires a single write to returned member @", writeOp.getMemberName()
        );
        failedToLower = true;
        return;
      }
      writtenMembers[writeOp.getMemberName()] = std::move(*loweredValue);
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
      llvm::append_range(results, it->second);
    }
    entryBuilder.create<func::ReturnOp>(computeFunc.getLoc(), results);
  }

  if (failed(checkStructConstrainBodyIsBooleanSummarizable(constrainFunc))) {
    return failure();
  }

  std::string constrainName =
      "smt_" + getHelperStem(structDef.getFullyQualifiedName()) + "_constrain";
  auto constrainHelper = moduleBuilder.create<func::FuncOp>(
      constrainFunc.getLoc(), constrainName,
      FunctionType::get(context, constrainSig->argTypes, TypeRange {smt::BoolType::get(context)})
  );
  {
    Block *entry = constrainHelper.addEntryBlock();
    OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);
    DenseMap<Value, Value> valueMap;
    SelfMemberMap selfMemberMap;
    auto numLoweredInputs = seedCallableArgumentMaps(
        constrainFunc.getArguments(), entry, constrainFunc, valueMap, selfMemberMap
    );
    if (failed(numLoweredInputs)) {
      return failure();
    }
    DenseMap<StringRef, Value> constParamMap = seedConstParamMap(
        constrainSig->templateParams,
        ValueRange(entry->getArguments()).drop_front(*numLoweredInputs)
    );
    ExprLowerer lowerer(*this, entryBuilder, valueMap, selfMemberMap, constParamMap);

    SmallVector<Value> conditions;
    bool failedToLower = false;
    constrainFunc.walk([&](Operation *op) {
      if (failedToLower) {
        return WalkResult::interrupt();
      }
      if (isa<ReturnOp, scf::YieldOp, scf::IfOp>(op) || op == constrainFunc.getOperation()) {
        return WalkResult::advance();
      }
      if (auto eqOp = dyn_cast<constrain::EmitEqualityOp>(op)) {
        auto lhs = lowerer.lower(eqOp.getLhs());
        auto rhs = lowerer.lower(eqOp.getRhs());
        if (failed(lhs) || failed(rhs)) {
          failedToLower = true;
          return WalkResult::interrupt();
        }
        auto lowered = lowerGuardedBoolean(
            op, entryBuilder.create<smt::EqOp>(eqOp.getLoc(), *lhs, *rhs).getResult(), lowerer,
            entryBuilder, "struct constrain body"
        );
        if (failed(lowered)) {
          failedToLower = true;
          return WalkResult::interrupt();
        }
        conditions.push_back(*lowered);
        return WalkResult::advance();
      }
      if (isa<constrain::EmitContainmentOp>(op)) {
        op->emitError(
            "verif-to-smt does not yet support 'constrain.in' in struct constrain bodies"
        );
        failedToLower = true;
        return WalkResult::interrupt();
      }
      if (auto call = dyn_cast<CallOp>(op)) {
        if (call.getNumResults() != 0) {
          return WalkResult::advance();
        }
        auto calleeTarget = getCalleeTarget(call);
        if (failed(calleeTarget)) {
          failedToLower = true;
          return WalkResult::interrupt();
        }
        if (!calleeTarget->get().isStructConstrain()) {
          op->emitError(
              "verif-to-smt only supports standalone calls to struct constrain functions inside "
              "struct constrain bodies"
          );
          failedToLower = true;
          return WalkResult::interrupt();
        }
        auto calleeStruct = calleeTarget->get()->getParentOfType<StructDefOp>();
        auto calleeHelpers = getOrCreateStructHelpers(module, calleeStruct);
        if (failed(calleeHelpers)) {
          failedToLower = true;
          return WalkResult::interrupt();
        }
        auto loweredOperands = lowerIncludeHelperOperands(
            call.getArgOperands(), call, entryBuilder, valueMap, selfMemberMap, constParamMap
        );
        if (failed(loweredOperands)) {
          failedToLower = true;
          return WalkResult::interrupt();
        }
        auto loweredTemplateParams = lowerTemplateParamOperands(
            calleeTarget->get().getOperation(), call.getTemplateParamsAttr(), entryBuilder,
            constParamMap
        );
        if (failed(loweredTemplateParams)) {
          failedToLower = true;
          return WalkResult::interrupt();
        }
        llvm::append_range(*loweredOperands, *loweredTemplateParams);
        Value callCondition = entryBuilder
                                  .create<func::CallOp>(
                                      call.getLoc(), calleeHelpers->constrain,
                                      TypeRange {smt::BoolType::get(context)}, *loweredOperands
                                  )
                                  .getResult(0);
        auto guarded =
            lowerGuardedBoolean(op, callCondition, lowerer, entryBuilder, "struct constrain body");
        if (failed(guarded)) {
          failedToLower = true;
          return WalkResult::interrupt();
        }
        conditions.push_back(*guarded);
        return WalkResult::advance();
      }
      if (op->getNumResults() != 0) {
        return WalkResult::advance();
      }
      op->emitError().append(
          "unsupported op in struct constrain SMT lowering: ", op->getName().getStringRef()
      );
      failedToLower = true;
      return WalkResult::interrupt();
    });
    if (failedToLower) {
      return failure();
    }

    Value combined = buildConjunction(entryBuilder, constrainFunc.getLoc(), conditions);
    entryBuilder.create<func::ReturnOp>(constrainFunc.getLoc(), combined);
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
  auto target = getDirectTargetDefinition(contract);
  if (failed(target)) {
    return failure();
  }
  return lowerCallableSignature(contract.getFunctionType(), *target);
}

/// Seed lowered contract argument maps for ordinary args and flattened `%self` members.
FailureOr<std::tuple<DenseMap<Value, Value>, SelfMemberMap, DenseMap<StringRef, Value>>>
LoweringContext::seedContractArgumentMaps(ContractOp contract, Block *entry) {
  DenseMap<Value, Value> valueMap;
  SelfMemberMap selfMemberMap;
  auto nextArg =
      seedCallableArgumentMaps(contract.getArguments(), entry, contract, valueMap, selfMemberMap);
  if (failed(nextArg)) {
    return failure();
  }
  auto loweredSig = lowerContractSignature(contract);
  if (failed(loweredSig)) {
    return failure();
  }
  DenseMap<StringRef, Value> constParamMap = seedConstParamMap(
      loweredSig->templateParams, ValueRange(entry->getArguments()).drop_front(*nextArg)
  );
  return std::make_tuple(std::move(valueMap), std::move(selfMemberMap), std::move(constParamMap));
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
  auto &[valueMap, selfMemberMap, constParamMap] = *seededMaps;
  ExprLowerer lowerer(*this, entryBuilder, valueMap, selfMemberMap, constParamMap);

  SmallVector<Value> conditions;
  bool failedToLower = false;
  contract.walk([&](Operation *op) {
    if (failedToLower || !predicate(op)) {
      return;
    }

    auto lowered = lowerGuardedCondition(op, lowerer, entryBuilder);
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

/// Create the shared `_target` helper for a free-function contract target.
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
  if (targetFunc->get().hasAllowConstraintAttr()) {
    targetFunc->get().emitOpError(
        "verif-to-smt does not support free-function targets with "
        "`function.allow_constraint`; only struct `@constrain` lowering is supported"
    );
    return failure();
  }

  std::string helperName =
      "smt_verif_" + getHelperStem(contract.getFullyQualifiedName()) + "_target";
  OpBuilder moduleBuilder = createModuleBuilder(module);
  auto helper = moduleBuilder.create<func::FuncOp>(
      contract.getLoc(), helperName,
      FunctionType::get(context, loweredSig->argTypes, TypeRange {smt::BoolType::get(context)})
  );

  Block *entry = helper.addEntryBlock();
  OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);
  auto rawTargetHelper = getOrCreateFunctionHelper(module, targetFunc->get());
  if (failed(rawTargetHelper)) {
    return failure();
  }

  auto targetLoweredSig =
      lowerCallableSignature(targetFunc->get().getFunctionType(), targetFunc->get());
  if (failed(targetLoweredSig)) {
    return failure();
  }

  ValueRange args = entry->getArguments();
  unsigned numHiddenTemplateArgs = targetLoweredSig->templateParams.size();
  unsigned numLoweredTargetInputs = targetLoweredSig->argTypes.size() - numHiddenTemplateArgs;
  ValueRange targetInputs = args.take_front(numLoweredTargetInputs);
  SmallVector<Value> targetCallArgs(targetInputs.begin(), targetInputs.end());
  llvm::append_range(targetCallArgs, args.take_back(numHiddenTemplateArgs));
  auto targetCall = entryBuilder.create<func::CallOp>(
      contract.getLoc(), rawTargetHelper->getSymName(),
      rawTargetHelper->getFunctionType().getResults(), targetCallArgs
  );

  ValueRange expectedResults =
      args.drop_front(numLoweredTargetInputs).take_front(targetLoweredSig->resultTypes.size());
  SmallVector<Value> conditions = buildEqualityConditions(
      entryBuilder, contract.getLoc(), targetCall.getResults(), expectedResults
  );
  Value combined = buildConjunction(entryBuilder, contract.getLoc(), conditions);
  entryBuilder.create<func::ReturnOp>(contract.getLoc(), combined);
  return helper;
}

FailureOr<SmallVector<Value>> LoweringContext::lowerFlattenedValue(
    Value value, Operation *origin, OpBuilder &builder, DenseMap<Value, Value> &valueMap,
    SelfMemberMap &selfMemberMap, DenseMap<StringRef, Value> &constParamMap
) {
  ExprLowerer lowerer(*this, builder, valueMap, selfMemberMap, constParamMap);
  SmallVector<Value> loweredValues;

  if (auto structType = dyn_cast<StructType>(value.getType())) {
    if (failed(ensureFlattenedStructMapping(value, origin, selfMemberMap))) {
      return failure();
    }
    auto members = getStructMembers(structType, origin);
    if (failed(members)) {
      return failure();
    }
    auto loweredStructIt = selfMemberMap.find(value);
    if (loweredStructIt == selfMemberMap.end()) {
      emitError(value.getLoc(), "missing flattened struct mapping while lowering aggregate value");
      return failure();
    }
    for (MemberDefOp member : *members) {
      auto loweredMemberIt = loweredStructIt->second.find(member.getSymName());
      if (loweredMemberIt == loweredStructIt->second.end()) {
        emitError(
            value.getLoc(), (Twine("missing lowered member @") + member.getSymName() +
                             " while lowering aggregate value")
                                .str()
        );
        return failure();
      }
      llvm::append_range(loweredValues, loweredMemberIt->second);
    }
    return loweredValues;
  }

  auto loweredValue = lowerer.lower(value);
  if (failed(loweredValue)) {
    return failure();
  }
  loweredValues.push_back(*loweredValue);
  return loweredValues;
}

/// Create the `_target` helper for a struct contract target.
FailureOr<func::FuncOp> LoweringContext::createStructTargetHelper(
    ModuleOp module, ContractOp contract, VerificationStage stage
) {
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
  unsigned numFlattenedSelfMembers = 0;
  for (MemberDefOp member : *members) {
    auto numLoweredValues = getNumLoweredBoundaryValues(member.getType(), contract);
    if (failed(numLoweredValues)) {
      return failure();
    }
    numFlattenedSelfMembers += *numLoweredValues;
  }

  OpBuilder moduleBuilder = createModuleBuilder(module);
  std::string helperName = "smt_verif_" + getHelperStem(contract.getFullyQualifiedName()) + "_" +
                           stageSuffix(stage).str() + "_target";
  auto helper = moduleBuilder.create<func::FuncOp>(
      contract.getLoc(), helperName,
      FunctionType::get(context, loweredSig->argTypes, TypeRange {smt::BoolType::get(context)})
  );

  Block *entry = helper.addEntryBlock();
  OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);
  ValueRange args = entry->getArguments();
  ValueRange selfArgs = args.take_front(numFlattenedSelfMembers);
  ValueRange nonSelfArgs = args.drop_front(numFlattenedSelfMembers);

  if (stage == VerificationStage::Compute) {
    auto computeCall = entryBuilder.create<func::CallOp>(
        contract.getLoc(), targetHelpers->compute, TypeRange(selfArgs.getTypes()), nonSelfArgs
    );
    SmallVector<Value> computeConditions = buildEqualityConditions(
        entryBuilder, contract.getLoc(), computeCall.getResults(), selfArgs
    );
    Value combined = buildConjunction(entryBuilder, contract.getLoc(), computeConditions);
    entryBuilder.create<func::ReturnOp>(contract.getLoc(), combined);
  } else {
    auto constrainCall = entryBuilder.create<func::CallOp>(
        contract.getLoc(), targetHelpers->constrain, TypeRange {smt::BoolType::get(context)}, args
    );
    entryBuilder.create<func::ReturnOp>(contract.getLoc(), constrainCall.getResult(0));
  }
  return helper;
}

FailureOr<func::FuncOp> LoweringContext::createCombinedBoolHelper(
    ModuleOp module, ContractOp contract, StringRef helperName, StringRef lhsHelperName,
    StringRef rhsHelperName
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
  auto lhsCall = entryBuilder.create<func::CallOp>(
      contract.getLoc(), lhsHelperName, TypeRange {smt::BoolType::get(context)},
      entry->getArguments()
  );
  auto rhsCall = entryBuilder.create<func::CallOp>(
      contract.getLoc(), rhsHelperName, TypeRange {smt::BoolType::get(context)},
      entry->getArguments()
  );
  Value combined =
      entryBuilder.create<smt::AndOp>(contract.getLoc(), lhsCall.getResult(0), rhsCall.getResult(0))
          .getResult();
  entryBuilder.create<func::ReturnOp>(contract.getLoc(), combined);
  return helper;
}

FailureOr<func::FuncOp> LoweringContext::createCombinedTargetHelper(
    ModuleOp module, ContractOp contract, StringRef helperName
) {
  auto helperInfoIt = contractHelpers.find(contract.getOperation());
  if (helperInfoIt == contractHelpers.end()) {
    contract.emitError("missing contract helper names for combined target generation");
    return failure();
  }

  const ContractHelperNames &helperInfo = helperInfoIt->second;
  if (!contract.hasStructTarget()) {
    if (auto helper = module.lookupSymbol<func::FuncOp>(helperInfo.compute.target)) {
      return helper;
    }
    contract.emitError("missing shared target helper for combined target generation");
    return failure();
  }

  return createCombinedBoolHelper(
      module, contract, helperName, helperInfo.compute.target, helperInfo.constrain.target
  );
}

/// Create or look up the direct helper symbols for one contract.
FailureOr<ContractHelperNames>
LoweringContext::getOrCreateContractHelpers(ModuleOp module, ContractOp contract) {
  if (auto it = contractHelpers.find(contract.getOperation()); it != contractHelpers.end()) {
    return it->second;
  }

  std::string prefix = "smt_verif_" + getHelperStem(contract.getFullyQualifiedName());

  SmallVector<IncludeOp> includes;
  contract.walk([&](IncludeOp includeOp) { includes.push_back(includeOp); });

  auto makeStageNames = [&](VerificationStage stage) {
    std::string stagePrefix = prefix + "_" + stageSuffix(stage).str();
    SmallVector<std::string> includeHelperNames;
    includeHelperNames.reserve(includes.size());
    for (auto [index, _] : llvm::enumerate(includes)) {
      includeHelperNames.push_back(stagePrefix + "_include_" + std::to_string(index));
    }
    return ContractStageHelperNames {
        stagePrefix + "_pre", stagePrefix + "_target", stagePrefix + "_post",
        stagePrefix + "_entry", includeHelperNames
    };
  };

  ContractHelperNames names {
      makeStageNames(VerificationStage::Compute), makeStageNames(VerificationStage::Constrain),
      ContractStageHelperNames {
          prefix + "_pre", prefix + "_target", prefix + "_post", prefix + "_entry", {}
      }
  };
  names.combined.includeHelpers.reserve(includes.size());
  for (auto [index, _] : llvm::enumerate(includes)) {
    names.combined.includeHelpers.push_back(prefix + "_include_" + std::to_string(index));
  }
  if (!contract.hasStructTarget()) {
    std::string targetName = prefix + "_target";
    names.compute.target = targetName;
    names.constrain.target = targetName;
    names.combined.target = targetName;
  }
  contractHelpers[contract.getOperation()] = names;

  auto createStageHelpers = [&](VerificationStage stage,
                                const ContractStageHelperNames &stageNames) -> LogicalResult {
    auto preHelper =
        createContractConditionHelper(module, contract, stageNames.pre, [stage](Operation *op) {
      return stage == VerificationStage::Compute ? isa<RequireComputeOp>(op)
                                                 : isa<RequireConstrainOp>(op);
    });
    if (failed(preHelper)) {
      return failure();
    }

    auto postHelper =
        createContractConditionHelper(module, contract, stageNames.post, [stage](Operation *op) {
      return stage == VerificationStage::Compute ? isa<EnsureComputeOp>(op)
                                                 : isa<EnsureConstrainOp>(op);
    });
    if (failed(postHelper)) {
      return failure();
    }

    if (contract.hasStructTarget()) {
      auto targetHelper = createStructTargetHelper(module, contract, stage);
      return success(succeeded(targetHelper));
    }
    return success();
  };
  if (failed(createStageHelpers(VerificationStage::Compute, names.compute)) ||
      failed(createStageHelpers(VerificationStage::Constrain, names.constrain))) {
    return failure();
  }

  if (!contract.hasStructTarget()) {
    auto targetHelper = createFreeFunctionTargetHelper(module, contract);
    if (failed(targetHelper)) {
      return failure();
    }
  }

  if (emitLegacyEntries) {
    if (failed(createCombinedBoolHelper(
            module, contract, names.combined.pre, names.compute.pre, names.constrain.pre
        )) ||
        failed(createCombinedBoolHelper(
            module, contract, names.combined.post, names.compute.post, names.constrain.post
        ))) {
      return failure();
    }
    if (contract.hasStructTarget()) {
      auto targetHelper = createCombinedTargetHelper(module, contract, names.combined.target);
      if (failed(targetHelper)) {
        return failure();
      }
    }
  }

  return names;
}

/// Lower one include operand list using the contract-local lowering environment.
FailureOr<SmallVector<Value>> LoweringContext::lowerIncludeHelperOperands(
    ValueRange operands, Operation *origin, OpBuilder &builder, DenseMap<Value, Value> &valueMap,
    SelfMemberMap &selfMemberMap, DenseMap<StringRef, Value> &constParamMap
) {
  SmallVector<Value> loweredOperands;
  for (Value operand : operands) {
    auto loweredOperand =
        lowerFlattenedValue(operand, origin, builder, valueMap, selfMemberMap, constParamMap);
    if (failed(loweredOperand)) {
      return failure();
    }
    llvm::append_range(loweredOperands, *loweredOperand);
  }
  return loweredOperands;
}

/// Create a per-include helper that forwards to the callee contract entry helper.
FailureOr<func::FuncOp> LoweringContext::createIncludeWrapperHelper(
    ModuleOp module, IncludeOp includeOp, StringRef helperName, VerificationStage stage
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
      includeOp.getLoc(),
      stage == VerificationStage::Compute ? calleeHelpers->compute.entry
                                          : calleeHelpers->constrain.entry,
      TypeRange {}, entry->getArguments()
  );
  entryBuilder.create<func::ReturnOp>(includeOp.getLoc());
  return helper;
}

FailureOr<func::FuncOp> LoweringContext::createCombinedIncludeWrapperHelper(
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
      includeOp.getLoc(), calleeHelpers->combined.entry, TypeRange {}, entry->getArguments()
  );
  entryBuilder.create<func::ReturnOp>(includeOp.getLoc());
  return helper;
}

/// Create the full `_entry` helper for one contract.
LogicalResult LoweringContext::createContractEntryHelper(
    ModuleOp module, ContractOp contract, SmallVectorImpl<IncludeOp> &includes,
    VerificationStage stage
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
  const ContractStageHelperNames &helperInfo = stage == VerificationStage::Compute
                                                   ? helperInfoIt->second.compute
                                                   : helperInfoIt->second.constrain;

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
  auto &[valueMap, selfMemberMap, constParamMap] = *seededMaps;

  auto preCall = entryBuilder.create<func::CallOp>(
      contract.getLoc(), helperInfo.pre, TypeRange {smt::BoolType::get(context)},
      entry->getArguments()
  );
  proveByUnsatAndAssert(
      entryBuilder, contract.getLoc(), preCall.getResult(0), contract.getSymName(),
      stageMessagePrefix(stage) + "pre"
  );

  auto targetCall = entryBuilder.create<func::CallOp>(
      contract.getLoc(), helperInfo.target, TypeRange {smt::BoolType::get(context)},
      entry->getArguments()
  );
  proveByUnsatAndAssert(
      entryBuilder, contract.getLoc(), targetCall.getResult(0), contract.getSymName(),
      stageMessagePrefix(stage) + "target"
  );

  for (auto [index, includeOp] : llvm::enumerate(includes)) {
    if (hasIfAncestor(includeOp.getOperation(), contract.getOperation())) {
      includeOp.emitError("verif-to-smt does not yet support guarded verif.include operations");
      return failure();
    }
    auto calleeTarget = includeOp.getCalleeTarget(tables);
    if (failed(calleeTarget)) {
      return failure();
    }
    auto loweredOperands = lowerIncludeHelperOperands(
        includeOp.getArgOperands(), includeOp, entryBuilder, valueMap, selfMemberMap, constParamMap
    );
    if (failed(loweredOperands)) {
      return failure();
    }
    auto loweredTemplateParams = lowerTemplateParamOperands(
        calleeTarget->get().getOperation(), includeOp.getTemplateParamsAttr(), entryBuilder,
        constParamMap
    );
    if (failed(loweredTemplateParams)) {
      return failure();
    }
    llvm::append_range(*loweredOperands, *loweredTemplateParams);
    entryBuilder.create<func::CallOp>(
        includeOp.getLoc(), helperInfo.includeHelpers[index], TypeRange {}, *loweredOperands
    );
  }

  auto postCall = entryBuilder.create<func::CallOp>(
      contract.getLoc(), helperInfo.post, TypeRange {smt::BoolType::get(context)},
      entry->getArguments()
  );
  proveByUnsatAndAssert(
      entryBuilder, contract.getLoc(), postCall.getResult(0), contract.getSymName(),
      stageMessagePrefix(stage) + "post"
  );

  entryBuilder.create<func::ReturnOp>(contract.getLoc());
  return success();
}

LogicalResult LoweringContext::createCombinedContractEntryHelper(
    ModuleOp module, ContractOp contract, SmallVectorImpl<IncludeOp> &includes
) {
  auto loweredSig = lowerContractSignature(contract);
  if (failed(loweredSig)) {
    return failure();
  }

  auto helperInfoIt = contractHelpers.find(contract.getOperation());
  if (helperInfoIt == contractHelpers.end()) {
    contract.emitError("missing contract helper names for combined entry generation");
    return failure();
  }
  const ContractStageHelperNames &helperInfo = helperInfoIt->second.combined;

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
  auto &[valueMap, selfMemberMap, constParamMap] = *seededMaps;

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
    if (hasIfAncestor(includeOp.getOperation(), contract.getOperation())) {
      includeOp.emitError("verif-to-smt does not yet support guarded verif.include operations");
      return failure();
    }
    auto calleeTarget = includeOp.getCalleeTarget(tables);
    if (failed(calleeTarget)) {
      return failure();
    }
    auto loweredOperands = lowerIncludeHelperOperands(
        includeOp.getArgOperands(), includeOp, entryBuilder, valueMap, selfMemberMap, constParamMap
    );
    if (failed(loweredOperands)) {
      return failure();
    }
    auto loweredTemplateParams = lowerTemplateParamOperands(
        calleeTarget->get().getOperation(), includeOp.getTemplateParamsAttr(), entryBuilder,
        constParamMap
    );
    if (failed(loweredTemplateParams)) {
      return failure();
    }
    llvm::append_range(*loweredOperands, *loweredTemplateParams);
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
  using Base = llzk::impl::VerifToSmtPassBase<VerifToSmtPass>;
  using Base::Base;

  /// Register the dialects required by the generated SMT helper IR.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, func::FuncDialect, boolean::BoolDialect, smt::SMTDialect>();
  }

  /// Run the contract lowering sequentially over the module body.
  void runOnOperation() override {
    ModuleOp module = getOperation();
    LoweringContext state {&getContext(), cleanup};

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
        if (func.hasAllowConstraintAttr()) {
          func.emitOpError(
              "verif-to-smt does not support free-function targets with "
              "`function.allow_constraint`; only struct `@constrain` lowering is supported"
          );
          signalPassFailure();
          return;
        }
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
                module, includeOp, contractHelpers->compute.includeHelpers[index],
                VerificationStage::Compute
            )) ||
            failed(state.createIncludeWrapperHelper(
                module, includeOp, contractHelpers->constrain.includeHelpers[index],
                VerificationStage::Constrain
            ))) {
          signalPassFailure();
          return;
        }
        if (cleanup && failed(state.createCombinedIncludeWrapperHelper(
                           module, includeOp, contractHelpers->combined.includeHelpers[index]
                       ))) {
          signalPassFailure();
          return;
        }
      }

      if (failed(state.createContractEntryHelper(
              module, contract, includes, VerificationStage::Compute
          )) ||
          failed(state.createContractEntryHelper(
              module, contract, includes, VerificationStage::Constrain
          ))) {
        signalPassFailure();
        return;
      }
      if (cleanup && failed(state.createCombinedContractEntryHelper(module, contract, includes))) {
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

    if (!cleanup) {
      return;
    }

    SmallVector<Operation *> toErase;
    module.walk([&](Operation *op) {
      if (auto structDef = dyn_cast<StructDefOp>(op)) {
        toErase.push_back(structDef);
        return;
      }
      if (auto funcDef = dyn_cast<FuncDefOp>(op); funcDef && !funcDef.isInStruct()) {
        toErase.push_back(funcDef);
      }
    });
    for (Operation *op : llvm::reverse(toErase)) {
      op->erase();
    }

    SmallVector<TemplateOp> emptyTemplates;
    module.walk([&](TemplateOp templateOp) {
      if (!templateHasLowerableDefs(templateOp)) {
        emptyTemplates.push_back(templateOp);
      }
    });
    for (TemplateOp templateOp : llvm::reverse(emptyTemplates)) {
      templateOp.erase();
    }
  }
};

} // namespace
