//===-- FlatteningTypeRefinement.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Shared refinement and consumer-compatibility rules for flattening.
///
//===----------------------------------------------------------------------===//

#include "FlatteningTypeRefinement.h"

#include "FlatteningContext.h"
#include "TemplateInstantiation.h"

#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Transforms/ConversionUtils.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolLookup.h"
#include "llzk/Util/SymbolTableLLZK.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/AttrTypeSubElements.h>

using namespace mlir;
using namespace llzk;
using namespace llzk::array;
using namespace llzk::component;
using namespace llzk::constrain;
using namespace llzk::function;
using namespace llzk::polymorphic;
using namespace llzk::polymorphic::detail;

namespace llzk::polymorphic::detail::flattening {

/// Return the common refinement of two matching type parameters, if it can be represented directly.
FailureOr<Attribute> getCommonRefinedParamAttr(
    Attribute lhs, Attribute rhs, const ConversionTracker &tracker, bool unifyDynamicSize,
    Operation *scope
) {
  assertValidAttrForParamOfType(lhs);
  assertValidAttrForParamOfType(rhs);
  if (lhs == rhs) {
    return lhs;
  }

  if (auto lhsAffine = llvm::dyn_cast<AffineMapAttr>(lhs)) {
    if (auto rhsInt = llvm::dyn_cast<IntegerAttr>(rhs)) {
      if (!isDynamic(rhsInt)) {
        return rhs;
      }
    }
  }
  if (auto rhsAffine = llvm::dyn_cast<AffineMapAttr>(rhs)) {
    if (auto lhsInt = llvm::dyn_cast<IntegerAttr>(lhs)) {
      if (!isDynamic(lhsInt)) {
        return lhs;
      }
    }
  }

  if (unifyDynamicSize) {
    auto dynamicInt = [](Attribute attr) -> IntegerAttr {
      auto intAttr = llvm::dyn_cast<IntegerAttr>(attr);
      return intAttr && isDynamic(intAttr) ? intAttr : nullptr;
    };
    if (dynamicInt(lhs) && llvm::isa_and_present<IntegerAttr, SymbolRefAttr, AffineMapAttr>(rhs)) {
      return rhs;
    }
    if (dynamicInt(rhs) && llvm::isa_and_present<IntegerAttr, SymbolRefAttr, AffineMapAttr>(lhs)) {
      return lhs;
    }
  }

  auto lhsTy = llvm::dyn_cast<TypeAttr>(lhs);
  auto rhsTy = llvm::dyn_cast<TypeAttr>(rhs);
  if (lhsTy && rhsTy) {
    auto commonType = getCommonRefinedType(lhsTy.getValue(), rhsTy.getValue(), tracker, scope);
    if (succeeded(commonType)) {
      return TypeAttr::get(*commonType);
    }
  }
  return failure();
}

/// Return the common refinement of two unifying types, if this pass can name it safely.
FailureOr<Type>
getCommonRefinedType(Type lhs, Type rhs, const ConversionTracker &tracker, Operation *scope) {
  if (scope) {
    auto common =
        getCommonRefinedType(tracker.scopeType(lhs, scope), tracker.scopeType(rhs, scope), tracker);
    return failed(common) ? FailureOr<Type>(failure()) : tracker.scopeType(*common, scope, false);
  }
  if (lhs == rhs) {
    return lhs;
  }
  if (tracker.isLegalConversion(lhs, rhs, "getCommonRefinedType")) {
    return rhs;
  }
  if (tracker.isLegalConversion(rhs, lhs, "getCommonRefinedType")) {
    return lhs;
  }

  if (auto lhsStruct = llvm::dyn_cast<StructType>(lhs)) {
    auto rhsStruct = llvm::dyn_cast<StructType>(rhs);
    if (!rhsStruct) {
      return failure();
    }
    if (lhsStruct.getNameRef() != rhsStruct.getNameRef()) {
      std::optional<StructType> lhsPreimage = tracker.getPreimage(lhsStruct);
      std::optional<StructType> rhsPreimage = tracker.getPreimage(rhsStruct);
      if (!lhsPreimage && !rhsPreimage) {
        return failure();
      }
      auto commonPreimage = getCommonRefinedType(
          lhsPreimage.value_or(lhsStruct), rhsPreimage.value_or(rhsStruct), tracker
      );
      if (failed(commonPreimage)) {
        return failure();
      }
      if (auto commonStruct = llvm::dyn_cast<StructType>(*commonPreimage)) {
        if (std::optional<StructType> instantiatedCommon = tracker.getInstantiation(commonStruct)) {
          return *instantiatedCommon;
        }
      }
      return *commonPreimage;
    }

    ArrayAttr lhsParams = lhsStruct.getParams();
    ArrayAttr rhsParams = rhsStruct.getParams();
    ArrayRef<Attribute> emptyParams;
    ArrayRef<Attribute> lhsValues = lhsParams ? lhsParams.getValue() : emptyParams;
    ArrayRef<Attribute> rhsValues = rhsParams ? rhsParams.getValue() : emptyParams;
    if (lhsValues.size() != rhsValues.size()) {
      return failure();
    }

    SmallVector<Attribute> commonParams;
    for (auto [lhsParam, rhsParam] : llvm::zip_equal(lhsValues, rhsValues)) {
      auto commonParam = getCommonRefinedParamAttr(lhsParam, rhsParam, tracker);
      if (failed(commonParam)) {
        return failure();
      }
      commonParams.push_back(*commonParam);
    }
    StructType commonStruct =
        commonParams.empty()
            ? StructType::get(lhsStruct.getNameRef())
            : StructType::get(
                  lhsStruct.getNameRef(), ArrayAttr::get(lhs.getContext(), commonParams)
              );
    if (std::optional<StructType> instantiatedCommon = tracker.getInstantiation(commonStruct)) {
      return *instantiatedCommon;
    }
    return commonStruct;
  }

  if (auto lhsPod = llvm::dyn_cast<pod::PodType>(lhs)) {
    auto rhsPod = llvm::dyn_cast<pod::PodType>(rhs);
    if (!rhsPod || lhsPod.getRecords().size() != rhsPod.getRecords().size()) {
      return failure();
    }
    SmallVector<pod::RecordAttr> commonRecords;
    for (auto [lhsRecord, rhsRecord] : llvm::zip_equal(lhsPod.getRecords(), rhsPod.getRecords())) {
      if (lhsRecord.getName() != rhsRecord.getName()) {
        return failure();
      }
      auto commonRecord = getCommonRefinedType(lhsRecord.getType(), rhsRecord.getType(), tracker);
      if (failed(commonRecord)) {
        return failure();
      }
      commonRecords.push_back(
          pod::RecordAttr::get(lhs.getContext(), lhsRecord.getName(), *commonRecord)
      );
    }
    return pod::PodType::get(lhs.getContext(), commonRecords);
  }

  if (auto lhsArray = llvm::dyn_cast<ArrayType>(lhs)) {
    auto rhsArray = llvm::dyn_cast<ArrayType>(rhs);
    if (!rhsArray) {
      return failure();
    }

    auto commonElement =
        getCommonRefinedType(lhsArray.getElementType(), rhsArray.getElementType(), tracker);
    if (failed(commonElement)) {
      return failure();
    }
    ArrayRef<Attribute> lhsDims = lhsArray.getDimensionSizes();
    ArrayRef<Attribute> rhsDims = rhsArray.getDimensionSizes();
    if (lhsDims.size() != rhsDims.size()) {
      return failure();
    }

    SmallVector<Attribute> commonDims;
    for (auto [lhsDim, rhsDim] : llvm::zip_equal(lhsDims, rhsDims)) {
      auto commonDim =
          getCommonRefinedParamAttr(lhsDim, rhsDim, tracker, /*unifyDynamicSize=*/true);
      if (failed(commonDim)) {
        return failure();
      }
      commonDims.push_back(*commonDim);
    }
    return ArrayType::get(*commonElement, commonDims);
  }

  return failure();
}

/// Check every shared binding, accounting for struct instantiation in the current iteration.
bool callSignatureUnifies(
    FunctionType replacement, FunctionType callee, ArrayRef<StringRef> calleeNamespace,
    const ConversionTracker &tracker, bool allowUnresolvedStructs, Operation *scope
) {
  // Qualify declaration-local names before consulting the caller's instantiation history.
  AttrTypeReplacer qualify;
  qualify.addReplacement([&](StructType type) -> std::optional<Type> {
    SymbolRefAttr name = type.getNameRef();
    for (StringRef part : calleeNamespace) {
      name = asSymbolRefAttr(StringAttr::get(type.getContext(), part), name);
    }
    return StructType::get(name, type.getParams());
  });
  replacement = llvm::cast<FunctionType>(tracker.scopeType(replacement, scope));
  callee = llvm::cast<FunctionType>(tracker.scopeType(qualify.replace(callee), scope));
  calleeNamespace = {};

  // Partial instantiations can already share a family even when only one signature has
  // an exact recorded preimage. Preserve that agreement before restoring original types.
  if (allowUnresolvedStructs &&
      functionTypesUnifyConsistently(replacement, callee, calleeNamespace)) {
    return true;
  }
  if (allowUnresolvedStructs) {
    // Keep partial families aligned while restoring fully instantiated nested arguments.
    AttrTypeReplacer concretePreimages;
    concretePreimages.addReplacement([&](StructType type) -> std::optional<Type> {
      return isNullOrEmpty(type.getParams()) ? tracker.getPreimage(type) : std::nullopt;
    });
    if (functionTypesUnifyConsistently(
            llvm::cast<FunctionType>(concretePreimages.replace(replacement)),
            llvm::cast<FunctionType>(concretePreimages.replace(callee)), calleeNamespace
        )) {
      return true;
    }
  }
  // A proposed read refinement must agree with already-instantiated companions. Before function
  // instantiation, however, a generic companion can still refine later: compare parameterized
  // preimages to distinguish that case from two incompatible concrete specializations.
  AttrTypeReplacer instantiations;
  instantiations.addReplacement([&](StructType type) -> std::optional<Type> {
    auto mapped =
        allowUnresolvedStructs ? tracker.getPreimage(type) : tracker.getInstantiation(type);
    if (mapped) {
      return *mapped;
    }
    return std::nullopt;
  });
  replacement = llvm::cast<FunctionType>(instantiations.replace(replacement));
  callee = llvm::cast<FunctionType>(instantiations.replace(callee));
  return functionTypesUnifyConsistently(replacement, callee, calleeNamespace);
}

/// Build the hypothetical call signature without changing any operands or result types.
FunctionType getRefinedCallSignature(
    CallOp call, Value refinedValue, Type refinedType,
    const DenseMap<Value, Type> *pendingRefinements
) {
  SmallVector<Type> argumentTypes;
  for (Value argument : call.getArgOperands()) {
    Type type = argument.getType();
    if (pendingRefinements) {
      if (Type pending = pendingRefinements->lookup(argument)) {
        type = pending;
      }
    }
    if (argument == refinedValue) {
      type = refinedType;
    }
    argumentTypes.push_back(type);
  }
  return FunctionType::get(call.getContext(), argumentTypes, call.getResultTypes());
}

FailureOr<Type> getRefinedMemberReadType(
    MemberReadOp read, Type receiverType, SymbolTableCollection &tables,
    const ConversionTracker &tracker, Type *projectedMemberType
) {
  auto structure = dyn_cast<StructType>(receiverType);
  if (!structure) {
    return failure();
  }
  while (auto instantiated = tracker.getInstantiation(structure, read)) {
    structure = *instantiated;
  }
  auto receiver = structure.getDefinition(tables, read);
  if (failed(receiver)) {
    return failure();
  }
  StructType declaredReceiver = receiver->get().getType();
  SmallVector<StringRef> receiverNamespace(receiver->getNamespace());
  auto member = lookupSymbolIn<MemberDefOp>(
      tables, SymbolRefAttr::get(read.getContext(), read.getMemberName()), std::move(*receiver),
      read
  );
  if (failed(member)) {
    return failure();
  }
  member->prependNamespace(receiverNamespace);
  // Re-spell declaration-local struct names before inserting caller-local parameter bindings.
  AttrTypeReplacer qualify;
  qualify.addReplacement([&](StructType type) -> std::optional<Type> {
    SymbolRefAttr name = type.getNameRef();
    for (StringRef part : member->getNamespace()) {
      name = asSymbolRefAttr(StringAttr::get(read.getContext(), part), name);
    }
    return StructType::get(name, type.getParams());
  });
  Type memberType = qualify.replace(member->get().getType());
  if (ArrayAttr params = structure.getParams()) {
    ArrayAttr names = declaredReceiver.getParams();
    if (!names || names.size() != params.size()) {
      return failure();
    }
    DenseMap<Attribute, Attribute> bindings;
    for (auto [name, value] : llvm::zip_equal(names, params)) {
      bindings[name] = value;
    }
    memberType = TemplateTypeConverter(std::move(bindings)).convertType(memberType);
  }
  if (projectedMemberType) {
    *projectedMemberType = memberType;
  }
  return getCommonRefinedType(read.getVal().getType(), memberType, tracker, read);
}

/// Look up a proposed type, falling back to the unchanged SSA type.
static Type plannedType(Value value, const DenseMap<Value, Type> *pending) {
  if (pending) {
    if (Type type = pending->lookup(value)) {
      return type;
    }
  }
  return value.getType();
}

void extendPendingRefinements(
    ModuleOp module, DenseMap<Value, Type> &pending, const ConversionTracker &tracker
) {
  SymbolTableCollection tables;
  bool changed;
  do {
    changed = false;
    module.walk([&](Operation *op) {
      Value result;
      Type type;
      if (auto create = dyn_cast<CreateArrayOp>(op)) {
        if (create.getElements().empty() || !llvm::any_of(create.getElements(), [&](Value value) {
          return pending.contains(value);
        })) {
          return;
        }
        Type element = plannedType(create.getElements().front(), &pending);
        if (!llvm::all_of(create.getElements(), [&](Value value) {
          return plannedType(value, &pending) == element;
        })) {
          return;
        }
        result = create.getResult();
        type = create.getType().cloneWith(element);
      } else if (auto read = dyn_cast<ReadArrayOp>(op)) {
        auto array = dyn_cast_or_null<ArrayType>(pending.lookup(read.getArrRef()));
        if (!array) {
          return;
        }
        result = read.getResult();
        type = array.getElementType();
      } else if (auto extract = dyn_cast<ExtractArrayOp>(op)) {
        auto array = dyn_cast_or_null<ArrayType>(pending.lookup(extract.getArrRef()));
        if (!array) {
          return;
        }
        result = extract.getResult();
        type = array.getSelectionType(extract.getIndices().size());
      } else if (auto memberRead = dyn_cast<MemberReadOp>(op)) {
        Type receiver = pending.lookup(memberRead.getComponent());
        if (!receiver) {
          return;
        }
        auto refined = getRefinedMemberReadType(memberRead, receiver, tables, tracker);
        if (failed(refined)) {
          return;
        }
        result = memberRead.getVal();
        type = *refined;
      }
      if (result && type != plannedType(result, &pending) &&
          canUseScalarizedValueAsType(
              plannedType(result, &pending), type, tracker, "extendPendingRefinements", op
          )) {
        pending[result] = type;
        changed = true;
      }
    });
  } while (changed);
}

/// Return true iff a value of `sourceType` can be used where `targetType` is requested after
/// scalarization has retained the common refinement.
bool canUseScalarizedValueAsType(
    Type sourceType, Type targetType, const ConversionTracker &tracker, const char *patName,
    Operation *scope
) {
  if (sourceType == targetType ||
      tracker.isLegalConversion(sourceType, targetType, patName, scope)) {
    return true;
  }
  auto commonType = getCommonRefinedType(sourceType, targetType, tracker, scope);
  return succeeded(commonType) && *commonType == targetType;
}

/// Return true iff `type` has no symbolic array dimensions or element type.
static bool isFullyConcreteArrayConsumerType(Type type) {
  auto arrayType = llvm::dyn_cast<ArrayType>(type);
  if (!arrayType) {
    return isConcreteType(type, /*allowStructParams=*/false);
  }
  return llvm::all_of(arrayType.getDimensionSizes(), [](Attribute attr) {
    return isConcreteStructParamAttr(attr);
  }) && isFullyConcreteArrayConsumerType(arrayType.getElementType());
}

/// Check the independently declared requirements of a consumer using eventual operand types.
/// Producer-specific propagation (array/member reads, initializers and cast repair) stays with
/// the caller; this routine never changes IR or assumes a new consumer will be rewritten.
bool canRefineConsumerUse(
    OpOperand &use, Type refinedType, SymbolTableCollection &tables,
    const ConversionTracker &tracker, ConsumerTypeCheck typeCheck,
    const DenseMap<Value, Type> *pendingRefinements, Operation *diagnosticOp
) {
  Operation *user = use.getOwner();
  Value value = use.get();
  auto eventualType = [&](Value operand) {
    return operand == value ? refinedType : plannedType(operand, pendingRefinements);
  };
  auto accepts = [&](Type requiredType) {
    switch (typeCheck) {
    case ConsumerTypeCheck::Exact:
      return refinedType == requiredType;
    case ConsumerTypeCheck::ConcreteExact:
      return !isFullyConcreteArrayConsumerType(requiredType) || refinedType == requiredType;
    case ConsumerTypeCheck::Refinement:
      return canUseScalarizedValueAsType(
          refinedType, requiredType, tracker, "ConsumerRefinement", user
      );
    case ConsumerTypeCheck::Unification:
      return typesUnify(refinedType, requiredType);
    }
    llvm_unreachable("unknown consumer type check");
  };
  auto incompatible = [&](StringRef contract, Operation *observation,
                          function_ref<void(Diagnostic &)> explain) {
    if (diagnosticOp) {
      auto diag = diagnosticOp->emitError()
                  << "cannot scalarize heterogeneous array read because its index-specific value "
                     "type is incompatible with "
                  << contract;
      explain(diag.attachNote(observation->getLoc()));
    }
    return false;
  };
  auto requiresType = [&](Type requiredType, StringRef contract, StringRef description,
                          Operation *observation) {
    if (accepts(requiredType)) {
      return true;
    }
    return incompatible(contract, observation, [&](Diagnostic &note) {
      note << description << " requires " << requiredType << ", but this read is replaced with "
           << refinedType;
    });
  };

  // Independently generic requirements remain valid until their owning producer is refined.
  // Exact propagation paths deliberately require agreement even with a generic declaration.
  auto requiresDeclaredType = [&](Type requiredType, StringRef contract, StringRef description) {
    return (typeCheck != ConsumerTypeCheck::Exact &&
            !isFullyConcreteArrayConsumerType(requiredType)) ||
           requiresType(requiredType, contract, description, user);
  };

  if (auto equality = dyn_cast<EmitEqualityOp>(user)) {
    Type otherType =
        eventualType(use.getOperandNumber() == 0 ? equality.getRhs() : equality.getLhs());
    if (!isValidEmitEqType(refinedType) || !isValidEmitEqType(otherType)) {
      return incompatible("an equality constraint", user, [&](Diagnostic &note) {
        note << "equality operands must exclude struct and string types, but replacement operands "
                "are "
             << refinedType << " and " << otherType;
      });
    }
    if ((typeCheck == ConsumerTypeCheck::Refinement &&
         !isFullyConcreteArrayConsumerType(otherType)) ||
        accepts(otherType)) {
      return true;
    }
    return incompatible("an equality constraint", user, [&](Diagnostic &note) {
      note << "other equality operand requires " << otherType << ", but this read is replaced with "
           << refinedType;
    });
  }
  if (auto containment = dyn_cast<EmitContainmentOp>(user)) {
    Type lhsType = eventualType(containment.getLhs());
    Type rhsType = eventualType(containment.getRhs());
    if (areValidContainmentOperandTypes(lhsType, rhsType)) {
      return true;
    }
    return incompatible("a containment constraint", user, [&](Diagnostic &note) {
      note << "replacement containment operands " << lhsType << " and " << rhsType
           << " must form an array and a compatible element or subarray";
    });
  }
  if (auto call = dyn_cast<CallOp>(user)) {
    unsigned argIdx = use.getOperandNumber() - call.getArgOperands().getBeginOperandIndex();
    if (argIdx >= call.getArgOperands().size()) {
      return false;
    }
    auto callee = call.getCalleeTarget(tables);
    if (failed(callee)) {
      return false;
    }
    FunctionType signature = getRefinedCallSignature(call, value, refinedType, pendingRefinements);
    FunctionType declared = callee->get().getFunctionType();
    if (callSignatureUnifies(signature, declared, callee->getNamespace(), tracker, false, call)) {
      return true;
    }
    return incompatible("a function call", user, [&](Diagnostic &note) {
      note << "call argument requires " << declared.getInput(argIdx)
           << ", but this read is replaced with " << refinedType << "; replacement call signature "
           << signature << " must unify consistently with " << declared;
    });
  }
  if (auto returnOp = dyn_cast<ReturnOp>(user)) {
    FuncDefOp function = returnOp->getParentOfType<FuncDefOp>();
    TypeRange resultTypes = function.getFunctionType().getResults();
    if (use.getOperandNumber() >= resultTypes.size()) {
      return false;
    }
    SmallVector<Type> replacementTypes;
    for (Value operand : returnOp.getOperands()) {
      replacementTypes.push_back(eventualType(operand));
    }
    FunctionType signature = FunctionType::get(user->getContext(), {}, replacementTypes);
    FunctionType declared = FunctionType::get(user->getContext(), {}, resultTypes);
    if (callSignatureUnifies(signature, declared, {}, tracker, false, user) &&
        (typeCheck == ConsumerTypeCheck::Unification ||
         accepts(resultTypes[use.getOperandNumber()]))) {
      return true;
    }
    return incompatible("a function return", user, [&](Diagnostic &note) {
      note << "replacement return signature " << signature
           << " does not unify with declared return signature " << declared;
    });
  }
  if (auto memberWrite = dyn_cast<MemberWriteOp>(user)) {
    if (memberWrite.getVal() != value) {
      return false;
    }
    auto memberDef = memberWrite.getMemberDefOp(tables);
    if (failed(memberDef)) {
      return false;
    }
    Type memberType = memberDef->get().getType();
    if (isFullyConcreteArrayConsumerType(memberType) || typeCheck == ConsumerTypeCheck::Exact) {
      return requiresType(memberType, "a member write", "member write", user);
    }
    // A generic destination may be independently refined by its other writes. Check those
    // writes with their pending types before accepting any one source's refinement.
    auto parent = getParentOfType<StructDefOp>(memberDef->get());
    if (!parent) {
      return false;
    }
    auto uses = llzk::getSymbolUses(memberDef->get(), parent);
    if (!uses) {
      return false;
    }
    for (SymbolTable::SymbolUse memberUse : *uses) {
      auto otherWrite = dyn_cast<MemberWriteOp>(memberUse.getUser());
      if (otherWrite && otherWrite != memberWrite &&
          !requiresType(
              eventualType(otherWrite.getVal()), "another member write", "other member write",
              otherWrite
          )) {
        return false;
      }
    }
    return true;
  }
  if (auto write = dyn_cast<WriteArrayOp>(user)) {
    if (write.getRvalue() != value) {
      return false;
    }
    auto arrayType = dyn_cast<ArrayType>(eventualType(write.getArrRef()));
    return arrayType &&
           requiresDeclaredType(arrayType.getElementType(), "an array write", "array write");
  }
  if (auto insert = dyn_cast<InsertArrayOp>(user)) {
    if (insert.getRvalue() != value) {
      return false;
    }
    auto arrayType = dyn_cast<ArrayType>(eventualType(insert.getArrRef()));
    return arrayType && requiresDeclaredType(
                            arrayType.getSelectionType(insert.getIndices().size()),
                            "an array insert", "array insert"
                        );
  }
  if (auto castOp = dyn_cast<UnifiableCastOp>(user)) {
    return castOp.getInput() == value &&
           requiresDeclaredType(
               castOp.getResult().getType(), "a unifiable cast", "unifiable cast result"
           );
  }
  if (auto create = dyn_cast<CreateArrayOp>(user)) {
    if (use.getOperandNumber() >= create.getElements().size()) {
      return false;
    }
    // The caller handles arrays whose result is also changed or removed by its plan.
    Type elementType = create.getType().getElementType();
    if (refinedType == elementType) {
      return true;
    }
    return incompatible("an array initializer", user, [&](Diagnostic &note) {
      note << "array initializer requires exactly " << elementType
           << ", but this read is replaced with " << refinedType;
    });
  }
  if (diagnosticOp) {
    auto diag = diagnosticOp->emitError(
        "cannot scalarize heterogeneous array read because it has an unsupported result user"
    );
    diag.attachNote(user->getLoc()) << "unsupported read result user is here";
  }
  return false;
}

/// Validate inferred array-access results using the same contracts for every source array.
static bool canRefineArrayAccessUsers(
    Value result, Type type, SymbolTableCollection &tables, const ConversionTracker &tracker,
    ConsumerTypeCheck typeCheck, const DenseMap<Value, Type> *pending
) {
  for (OpOperand &use : result.getUses()) {
    if (auto read = dyn_cast<MemberReadOp>(use.getOwner())) {
      if (read.getComponent() != result) {
        return false;
      }
      auto refined = getRefinedMemberReadType(read, type, tables, tracker);
      if (failed(refined) ||
          (*refined != read.getVal().getType() &&
           !canRefineMemberReadUsersToType(read, *refined, tables, tracker, pending))) {
        return false;
      }
    } else if (!canRefineConsumerUse(use, type, tables, tracker, typeCheck, pending)) {
      return false;
    }
  }
  return true;
}

/// Return true iff typed users of `array.new` can accept a refined result type.
///
/// Retagging an array result updates its SSA value globally. Fully concrete consumers are not
/// retagged by propagation patterns, so they must already request the proposed array type.
bool canRefineCreateArrayUsersToType(
    CreateArrayOp createOp, Type refinedType, SymbolTableCollection &tables,
    const ConversionTracker &tracker, const DenseMap<Value, Type> *pendingRefinements
) {
  // The allocation's result and initializer operands have one exact element type. A write
  // can suggest a refinement, but cannot retag an initializer owned by another producer.
  Type elementType = mlir::cast<ArrayType>(refinedType).getElementType();
  for (Value initializer : createOp.getElements()) {
    if (plannedType(initializer, pendingRefinements) != elementType) {
      return false;
    }
  }
  Value result = createOp.getResult();
  for (OpOperand &use : result.getUses()) {
    Operation *user = use.getOwner();
    if (WriteArrayOp arrayWrite = llvm::dyn_cast<WriteArrayOp>(user)) {
      if (arrayWrite.getArrRef() != result) {
        return false;
      }
      // Retagging the allocation changes the element type observed by every direct write, but
      // does not change the write's rvalue. In particular, concrete affine specializations can
      // unify while still symbolic and become incompatible after instantiation, so require an
      // exact element-type match instead of the usual unification check.
      if (plannedType(arrayWrite.getRvalue(), pendingRefinements) !=
          mlir::cast<ArrayType>(refinedType).getElementType()) {
        return false;
      }
      continue;
    } else if (InsertArrayOp arrayInsert = llvm::dyn_cast<InsertArrayOp>(user)) {
      if (arrayInsert.getArrRef() != result) {
        return false;
      }
      // Retagging the allocation changes the subarray type required by every insert, but does
      // not change an insert's rvalue. As with array.write above, require an exact match rather
      // than relying on pre-instantiation unification of concrete affine specializations.
      ArrayType refinedArrayType = mlir::cast<ArrayType>(refinedType);
      Type refinedRvalueType = refinedArrayType.getSelectionType(arrayInsert.getIndices().size());
      if (plannedType(arrayInsert.getRvalue(), pendingRefinements) != refinedRvalueType) {
        return false;
      }
      continue;
    } else if (auto read = dyn_cast<ReadArrayOp>(user)) {
      if (read.getArrRef() != result || !canRefineArrayAccessUsers(
                                            read.getResult(), elementType, tables, tracker,
                                            ConsumerTypeCheck::ConcreteExact, pendingRefinements
                                        )) {
        return false;
      }
    } else if (auto extract = dyn_cast<ExtractArrayOp>(user)) {
      Type selected = cast<ArrayType>(refinedType).getSelectionType(extract.getIndices().size());
      if (extract.getArrRef() != result || !canRefineArrayAccessUsers(
                                               extract.getResult(), selected, tables, tracker,
                                               ConsumerTypeCheck::Exact, pendingRefinements
                                           )) {
        return false;
      }
    } else if (!canRefineConsumerUse(
                   use, refinedType, tables, tracker, ConsumerTypeCheck::ConcreteExact,
                   pendingRefinements
               )) {
      return false;
    }
  }
  return true;
}

/// Return true iff typed consumers of a member read can use its refined result type.
///
/// A generic read result may unify both with the proposed member type and with a different
/// concrete type required by a consumer. Updating the read in that situation would make the
/// consuming operation invalid, so the member definition must remain generic.
bool canRefineMemberReadUsersToType(
    MemberReadOp readOp, Type refinedType, SymbolTableCollection &tables,
    const ConversionTracker &tracker, const DenseMap<Value, Type> *pendingRefinements
) {
  Value result = readOp.getVal();
  for (OpOperand &use : result.getUses()) {
    Operation *user = use.getOwner();
    if (auto memberRead = dyn_cast<MemberReadOp>(user)) {
      if (memberRead.getComponent() != result) {
        return false;
      }
      auto type = getRefinedMemberReadType(memberRead, refinedType, tables, tracker);
      if (failed(type) || (*type != memberRead.getVal().getType() &&
                           !canRefineMemberReadUsersToType(
                               memberRead, *type, tables, tracker, pendingRefinements
                           ))) {
        return false;
      }
      continue;
    }
    if (CreateArrayOp createArray = llvm::dyn_cast<CreateArrayOp>(user)) {
      if (use.getOperandNumber() >= createArray.getElements().size()) {
        return false;
      }
      // array.new requires every initializer to exactly match its declared element type. Allow
      // this refinement only when the array can follow it: every other initializer already has
      // the refined type and the array element type can be refined accordingly.
      bool canUse = canUseScalarizedValueAsType(
          createArray.getType().getElementType(), refinedType, tracker,
          "UpdateMemberDefTypeFromWrite", readOp
      );
      if (!canUse) {
        return false;
      }
      for (OpOperand &initializer : createArray->getOpOperands()) {
        if (&initializer != &use &&
            plannedType(initializer.get(), pendingRefinements) != refinedType) {
          return false;
        }
      }
      ArrayType refinedArrayType = createArray.getType().cloneWith(refinedType);
      if (!canRefineCreateArrayUsersToType(
              createArray, refinedArrayType, tables, tracker, pendingRefinements
          )) {
        return false;
      }
      continue;
    }
    if (auto read = dyn_cast<ReadArrayOp>(user)) {
      auto arrayType = dyn_cast<ArrayType>(refinedType);
      if (read.getArrRef() != result || !arrayType ||
          !canRefineArrayAccessUsers(
              read.getResult(), arrayType.getElementType(), tables, tracker,
              ConsumerTypeCheck::ConcreteExact, pendingRefinements
          )) {
        return false;
      }
      continue;
    }
    if (auto extract = dyn_cast<ExtractArrayOp>(user)) {
      auto arrayType = dyn_cast<ArrayType>(refinedType);
      if (extract.getArrRef() != result || !arrayType ||
          !canRefineArrayAccessUsers(
              extract.getResult(), arrayType.getSelectionType(extract.getIndices().size()), tables,
              tracker, ConsumerTypeCheck::Exact, pendingRefinements
          )) {
        return false;
      }
      continue;
    }
    if (UnifiableCastOp castOp = llvm::dyn_cast<UnifiableCastOp>(user)) {
      if (castOp.getInput() != result) {
        return false;
      }
      Type castResultType = castOp.getResult().getType();
      if (refinedType == castResultType) {
        continue;
      }
      // Step 6 can repair a stale cast result only when every result use is a compatible member
      // write. Any other consumer observes the independently declared result type, so leave the
      // read generic rather than producing a cast whose input and output no longer unify.
      if (castOp.getResult().use_empty() || typesUnify(refinedType, castResultType)) {
        return false;
      }
      auto canUse = canUseScalarizedValueAsType(
          castResultType, refinedType, tracker, "UpdateUnifiableCastResultFromInput", readOp
      );
      if (!canUse) {
        return false;
      }
      for (OpOperand &castUse : castOp.getResult().getUses()) {
        auto memberWrite = llvm::dyn_cast<MemberWriteOp>(castUse.getOwner());
        if (!memberWrite || memberWrite.getVal() != castUse.get()) {
          return false;
        }
        auto memberDef = memberWrite.getMemberDefOp(tables);
        if (failed(memberDef) || !typesUnify(refinedType, memberDef->get().getType())) {
          return false;
        }
      }
      continue;
    }
    // Stores retain exact destination types; ordinary member consumers can also use a known
    // specialization. Cast repair above remains specific to this propagation path.
    ConsumerTypeCheck check = isa<WriteArrayOp, InsertArrayOp>(user)
                                  ? ConsumerTypeCheck::Exact
                                  : ConsumerTypeCheck::Refinement;
    if (!canRefineConsumerUse(use, refinedType, tables, tracker, check, pendingRefinements)) {
      return false;
    }
  }
  return true;
}

/// Rebuild a struct constraint call with its witness specialized to a concrete struct type.
static CallOp replaceStructConstraintCallWithSpecializedWitness(
    PatternRewriter &rewriter, CallOp call, Value specializedWitness
) {
  SmallVector<Value> args(call.getArgOperands());
  args.front() = specializedWitness;
  StructType structType = llvm::cast<StructType>(specializedWitness.getType());
  SymbolRefAttr callee =
      appendLeaf(structType.getNameRef(), call.getCalleeAttr().getLeafReference());
  // Argument and result dictionaries are inherent properties, so the replacement wrapper only
  // preserves the discardable attributes. The operand/result ordering does not change here.
  ArrayAttr argAttrs = call.getArgAttrsAttr(), resAttrs = call.getResAttrsAttr();
  CallOp replacement = replaceOpWithNewOp<CallOp>(
      rewriter, call, call.getResultTypes(), callee,
      CallOp::toVectorOfValueRange(call.getMapOperands()), call.getNumDimsPerMapAttr(), args
  );
  replacement.setArgAttrsAttr(argAttrs);
  replacement.setResAttrsAttr(resAttrs);
  return replacement;
}

/// Validate external observations of a specialized witness and return the calls to retarget.
/// The caller identifies uses handled by its own rewrite. Scalarization reports failures at
/// `diagnosticOp`; speculative propagation leaves it null and simply declines the rewrite.
FailureOr<SmallVector<CallOp>> collectSpecializedWitnessConstraints(
    Value source, Type type, SymbolTableCollection &tables,
    llvm::function_ref<bool(OpOperand &)> isInternalUse, Operation *diagnosticOp
) {
  SmallVector<CallOp> calls;
  for (OpOperand &use : source.getUses()) {
    if (isInternalUse(use)) {
      continue;
    }
    auto call = dyn_cast<CallOp>(use.getOwner());
    if (!call || !call.calleeIsStructConstrain() || use.getOperandNumber() != 0 ||
        !isa<StructType>(type)) {
      if (diagnosticOp) {
        auto diag = diagnosticOp->emitError(
            "cannot scalarize array because a generic nondeterministic initializer has an "
            "unsupported external use"
        );
        diag.attachNote(use.getOwner()->getLoc()) << "source witness is also used here";
      }
      return failure();
    }
    StructType structType = llvm::cast<StructType>(type);
    SymbolRefAttr callee =
        appendLeaf(structType.getNameRef(), call.getCalleeAttr().getLeafReference());
    auto target = lookupTopLevelSymbol<FuncDefOp>(tables, callee, call);
    if (failed(target)) {
      return failure();
    }
    SmallVector<Type> argTypes(call.getArgOperands().getTypes());
    argTypes.front() = type;
    // Use the namespace-aware compatibility rule from CallOp verification, including generic
    // companion arguments that need not exactly equal the specialized parameter types.
    if (!typeListsUnify(argTypes, target->get().getArgumentTypes(), target->getNamespace())) {
      if (diagnosticOp) {
        auto diag = diagnosticOp->emitError(
            "cannot scalarize array because specializing a generic nondeterministic initializer "
            "would make an external constraint call incompatible with its retargeted callee"
        );
        diag.attachNote(call.getLoc())
            << "retargeted constraint expects argument types " << target->get().getArgumentTypes();
      }
      return failure();
    }
    calls.push_back(call);
  }
  return calls;
}

/// Materialize a specialized nondeterministic witness after its source and retarget constraints.
///
/// The caller must have validated that every constraint in `externalConstraints` can observe the
/// specialized type.
Value materializeSpecializedNondet(
    PatternRewriter &rewriter, NonDetOp sourceNondet, Location loc, Type type,
    ArrayRef<CallOp> externalConstraints
) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(sourceNondet);
  NonDetOp specializedNondet = NonDetOp::create(rewriter, loc, type);
  specializedNondet->setDiscardableAttrs(sourceNondet->getDiscardableAttrDictionary());
  Value specializedValue = specializedNondet.getResult();
  for (CallOp call : externalConstraints) {
    replaceStructConstraintCallWithSpecializedWitness(rewriter, call, specializedValue);
  }
  return specializedValue;
}

} // namespace llzk::polymorphic::detail::flattening
