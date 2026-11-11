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
static FailureOr<Attribute> getCommonRefinedParamAttr(
    Attribute lhs, Attribute rhs, const ConversionTracker &tracker, bool unifyDynamicSize = false
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
    auto commonType = getCommonRefinedType(lhsTy.getValue(), rhsTy.getValue(), tracker);
    if (succeeded(commonType)) {
      return TypeAttr::get(*commonType);
    }
  }
  return failure();
}

/// Return the common refinement of two unifying types, if this pass can name it safely.
FailureOr<Type> getCommonRefinedType(Type lhs, Type rhs, const ConversionTracker &tracker) {
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
    const ConversionTracker &tracker, bool allowUnresolvedStructs
) {
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

/// Return true iff a value of `sourceType` can be used where `targetType` is requested after
/// scalarization has retained the common refinement.
bool canUseScalarizedValueAsType(
    Type sourceType, Type targetType, const ConversionTracker &tracker, const char *patName
) {
  if (sourceType == targetType || tracker.isLegalConversion(sourceType, targetType, patName)) {
    return true;
  }
  auto commonType = getCommonRefinedType(sourceType, targetType, tracker);
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

/// Return true iff `rhsType` remains a valid containment operand for `lhsType`.
///
/// This mirrors EmitContainmentOp::verify without emitting a diagnostic, so member-read
/// refinement can reject a type change that would invalidate constrain.in.
static bool areValidContainmentOperandTypes(Type lhsType, Type rhsType) {
  auto lhsArrayType = llvm::dyn_cast<ArrayType>(lhsType);
  if (!lhsArrayType) {
    return false;
  }
  if (auto rhsArrayType = llvm::dyn_cast<ArrayType>(rhsType)) {
    ArrayRef<Attribute> lhsDims = lhsArrayType.getDimensionSizes();
    ArrayRef<Attribute> rhsDims = rhsArrayType.getDimensionSizes();
    return lhsDims.size() >= rhsDims.size() &&
           typeParamsUnify(lhsDims.drop_front(lhsDims.size() - rhsDims.size()), rhsDims) &&
           typesUnify(lhsArrayType.getElementType(), rhsArrayType.getElementType());
  }
  return typesUnify(lhsArrayType.getElementType(), rhsType);
}

/// Array store operands supported by a consumer check. Reads and extracts have different contracts.
enum class ArrayStoreKind : std::uint8_t { None, Element, Subarray };

/// Look up an independently declared consumer type after validating the operand's role.
/// Callers choose which array stores they support and how to compare the returned type; this
/// lookup does not decide whether equality, unification, or downstream propagation is required.
static FailureOr<Type> getConsumerRequiredType(
    OpOperand &use, SymbolTableCollection &tables, ArrayStoreKind arrayStore = ArrayStoreKind::None
) {
  Operation *user = use.getOwner();
  Value value = use.get();
  if (auto memberWrite = dyn_cast<MemberWriteOp>(user)) {
    if (memberWrite.getVal() != value) {
      return failure();
    }
    auto memberDef = memberWrite.getMemberDefOp(tables);
    if (failed(memberDef)) {
      return failure();
    }
    return memberDef->get().getType();
  }
  if (auto call = dyn_cast<CallOp>(user)) {
    unsigned argIdx = use.getOperandNumber() - call.getArgOperands().getBeginOperandIndex();
    if (argIdx >= call.getArgOperands().size()) {
      return failure();
    }
    auto callee = call.getCalleeTarget(tables);
    if (failed(callee)) {
      return failure();
    }
    return callee->get().getFunctionType().getInput(argIdx);
  }
  if (auto returnOp = dyn_cast<ReturnOp>(user)) {
    FuncDefOp function = returnOp->getParentOfType<FuncDefOp>();
    TypeRange resultTypes = function.getFunctionType().getResults();
    unsigned resultIdx = use.getOperandNumber();
    if (resultIdx >= resultTypes.size()) {
      return failure();
    }
    return resultTypes[resultIdx];
  }
  if (auto castOp = dyn_cast<UnifiableCastOp>(user)) {
    if (castOp.getInput() != value) {
      return failure();
    }
    return castOp.getResult().getType();
  }
  if (auto write = dyn_cast<WriteArrayOp>(user);
      write && arrayStore == ArrayStoreKind::Element && write.getRvalue() == value) {
    return write.getArrRefType().getElementType();
  }
  if (auto insert = dyn_cast<InsertArrayOp>(user);
      insert && arrayStore == ArrayStoreKind::Subarray && insert.getRvalue() == value) {
    return insert.getArrRefType().getSelectionType(insert.getIndices().size());
  }
  return failure();
}

/// Return true iff typed users of `array.new` can accept a refined result type.
///
/// Retagging an array result updates its SSA value globally. Fully concrete consumers are not
/// retagged by propagation patterns, so they must already request the proposed array type.
bool canRefineCreateArrayUsersToType(
    CreateArrayOp createOp, Type refinedType, SymbolTableCollection &tables
) {
  Value result = createOp.getResult();
  for (OpOperand &use : result.getUses()) {
    Operation *user = use.getOwner();
    Type requiredType;
    if (isa<MemberWriteOp, CallOp, ReturnOp, UnifiableCastOp>(user)) {
      auto requirement = getConsumerRequiredType(use, tables);
      if (failed(requirement)) {
        return false;
      }
      requiredType = *requirement;
    } else if (WriteArrayOp arrayWrite = llvm::dyn_cast<WriteArrayOp>(user)) {
      if (arrayWrite.getArrRef() != result) {
        return false;
      }
      // Retagging the allocation changes the element type observed by every direct write, but
      // does not change the write's rvalue. In particular, concrete affine specializations can
      // unify while still symbolic and become incompatible after instantiation, so require an
      // exact element-type match instead of the usual unification check.
      if (arrayWrite.getRvalue().getType() != mlir::cast<ArrayType>(refinedType).getElementType()) {
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
      if (arrayInsert.getRvalue().getType() != refinedRvalueType) {
        return false;
      }
      continue;
    } else if (ReadArrayOp arrayRead = llvm::dyn_cast<ReadArrayOp>(user)) {
      if (arrayRead.getArrRef() != result) {
        return false;
      }
      // Retagging the allocation also changes this inferred read result. Check the result's
      // independently declared consumers before accepting the retag: unlike the allocation
      // itself, propagation does not retarget a concrete call, cast, return, member write, or
      // subsequent array write that observes the read.
      Type refinedElementType = mlir::cast<ArrayType>(refinedType).getElementType();
      for (OpOperand &readUse : arrayRead.getResult().getUses()) {
        auto requirement = getConsumerRequiredType(readUse, tables, ArrayStoreKind::Element);
        if (failed(requirement)) {
          return false;
        }
        Type readRequiredType = *requirement;
        if (readRequiredType != refinedElementType &&
            isFullyConcreteArrayConsumerType(readRequiredType)) {
          return false;
        }
      }
      continue;
    } else if (ExtractArrayOp arrayExtract = llvm::dyn_cast<ExtractArrayOp>(user)) {
      if (arrayExtract.getArrRef() != result) {
        return false;
      }
      // Retagging the allocation also changes this inferred extract result. As with array.read,
      // its typed consumers are not retargeted by propagation patterns, so they must already
      // request the refined subarray type. In particular, a parameterized struct type may still
      // have concrete arguments that are instantiated only after this propagation step.
      Type refinedExtractType =
          mlir::cast<ArrayType>(refinedType).getSelectionType(arrayExtract.getIndices().size());
      for (OpOperand &extractUse : arrayExtract.getResult().getUses()) {
        auto requirement = getConsumerRequiredType(extractUse, tables, ArrayStoreKind::Subarray);
        if (failed(requirement)) {
          return false;
        }
        Type extractRequiredType = *requirement;
        if (extractRequiredType != refinedExtractType) {
          return false;
        }
      }
      continue;
    } else if (EmitEqualityOp equalityOp = llvm::dyn_cast<EmitEqualityOp>(user)) {
      Value otherOperand;
      if (equalityOp.getLhs() == result) {
        otherOperand = equalityOp.getRhs();
      } else if (equalityOp.getRhs() == result) {
        otherOperand = equalityOp.getLhs();
      } else {
        return false;
      }
      // constrain.eq is not retagged when an array allocation is refined. Its other operand
      // must therefore already have the proposed type when concrete; otherwise two array types
      // that unify before template instantiation can become incompatible concrete types later.
      if (otherOperand.getType() != refinedType &&
          isFullyConcreteArrayConsumerType(otherOperand.getType())) {
        return false;
      }
      continue;
    } else {
      // Retagging changes this SSA value for every direct user. Be conservative until each
      // type-constraining use is explicitly checked above.
      return false;
    }
    // Array element specializations with distinct concrete affine arguments can still unify
    // before template instantiation. This array result is retagged in place, though, and no
    // propagation pattern retargets these users, so a fully concrete requirement must match
    // exactly.
    if (isFullyConcreteArrayConsumerType(requiredType) && refinedType != requiredType) {
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
    if (MemberWriteOp memberWrite = llvm::dyn_cast<MemberWriteOp>(user)) {
      if (memberWrite.getVal() != result) {
        return false;
      }
      auto memberDef = memberWrite.getMemberDefOp(tables);
      if (failed(memberDef)) {
        return false;
      }
      Type memberType = memberDef->get().getType();
      if (isConcreteType(memberType, /*allowStructParams=*/false)) {
        auto canUse = canUseScalarizedValueAsType(
            refinedType, memberType, tracker, "UpdateMemberDefTypeFromWrite"
        );
        if (!canUse) {
          return false;
        }
      }
      if (!isConcreteType(memberType, /*allowStructParams=*/false)) {
        // A generic destination may be independently refined by its other writes. Retagging this
        // read before checking them can make a previously valid generic write incompatible with
        // a concrete sibling write after template instantiation.
        StructDefOp parentStruct = getParentOfType<StructDefOp>(memberDef->get());
        if (!parentStruct) {
          return false;
        }
        auto memberUses = llzk::getSymbolUses(memberDef->get(), parentStruct);
        if (!memberUses) {
          return false;
        }
        for (SymbolTable::SymbolUse memberUse : memberUses.value()) {
          MemberWriteOp otherWrite = llvm::dyn_cast<MemberWriteOp>(memberUse.getUser());
          if (!otherWrite || otherWrite == memberWrite) {
            continue;
          }
          auto canUse = canUseScalarizedValueAsType(
              refinedType, otherWrite.getVal().getType(), tracker, "UpdateMemberDefTypeFromWrite"
          );
          if (!canUse) {
            return false;
          }
        }
      }
      continue;
    }
    if (CallOp call = llvm::dyn_cast<CallOp>(user)) {
      if (call.getArgOperands().empty()) {
        return false;
      }
      unsigned argIdx = use.getOperandNumber() - call.getArgOperands().getBeginOperandIndex();
      if (argIdx >= call.getArgOperands().size()) {
        return false;
      }
      auto callee = call.getCalleeTarget(tables);
      if (failed(callee) ||
          !callSignatureUnifies(
              getRefinedCallSignature(call, result, refinedType, pendingRefinements),
              callee->get().getFunctionType(), callee->getNamespace(), tracker
          )) {
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
          "UpdateMemberDefTypeFromWrite"
      );
      if (!canUse) {
        return false;
      }
      for (OpOperand &initializer : createArray->getOpOperands()) {
        if (&initializer != &use && initializer.get().getType() != refinedType) {
          return false;
        }
      }
      ArrayType refinedArrayType = createArray.getType().cloneWith(refinedType);
      if (!canRefineCreateArrayUsersToType(createArray, refinedArrayType, tables)) {
        return false;
      }
      continue;
    }
    if (WriteArrayOp arrayWrite = llvm::dyn_cast<WriteArrayOp>(user)) {
      if (arrayWrite.getRvalue() != result) {
        return false;
      }
      Type elementType = arrayWrite.getArrRefType().getElementType();
      // array.write's rvalue is not independently converted by the member refinement. Its
      // destination must therefore already require exactly the proposed read type. In
      // particular, do not use the normal unification-based conversion check here: symbolic
      // Cell<0> and Cell<1> can appear compatible before their template instantiations are
      // materialized, but become distinct concrete element types later in flattening.
      if (refinedType != elementType) {
        return false;
      }
      continue;
    }
    if (InsertArrayOp arrayInsert = llvm::dyn_cast<InsertArrayOp>(user)) {
      if (arrayInsert.getRvalue() != result) {
        return false;
      }
      // The refined read already has the complete inserted subarray type. Only select the
      // destination dimensions; selecting the read again would discard its leading dimensions.
      // Require exact agreement because affine specializations that unify before instantiation
      // can otherwise diverge later.
      if (refinedType !=
          arrayInsert.getArrRefType().getSelectionType(arrayInsert.getIndices().size())) {
        return false;
      }
      continue;
    }
    if (ReadArrayOp arrayRead = llvm::dyn_cast<ReadArrayOp>(user)) {
      if (arrayRead.getArrRef() != result) {
        return false;
      }
      auto refinedArrayType = llvm::dyn_cast<ArrayType>(refinedType);
      if (!refinedArrayType) {
        return false;
      }
      // A member-read refinement retags the array.read result to the array's refined element
      // type. Validate its independently declared users first: propagation does not retarget a
      // concrete consumer of that inferred result.
      for (OpOperand &arrayReadUse : arrayRead.getResult().getUses()) {
        if (auto memberRead = dyn_cast<MemberReadOp>(arrayReadUse.getOwner())) {
          if (memberRead.getComponent() != arrayRead.getResult()) {
            return false;
          }
          // Propagation retags this result after its receiver changes.
          continue;
        }
        auto requirement = getConsumerRequiredType(arrayReadUse, tables, ArrayStoreKind::Element);
        if (failed(requirement)) {
          return false;
        }
        Type requiredType = *requirement;
        if (requiredType != refinedArrayType.getElementType() &&
            isFullyConcreteArrayConsumerType(requiredType)) {
          return false;
        }
      }
      continue;
    }
    if (ExtractArrayOp arrayExtract = llvm::dyn_cast<ExtractArrayOp>(user)) {
      if (arrayExtract.getArrRef() != result) {
        return false;
      }
      auto refinedArrayType = llvm::dyn_cast<ArrayType>(refinedType);
      if (!refinedArrayType) {
        return false;
      }
      // Refining the member read retags this inferred extract result. Its direct consumers are
      // not retargeted by propagation, so they must already require the selected subarray type.
      Type refinedExtractType = refinedArrayType.getSelectionType(arrayExtract.getIndices().size());
      for (OpOperand &extractUse : arrayExtract.getResult().getUses()) {
        auto requirement = getConsumerRequiredType(extractUse, tables, ArrayStoreKind::Subarray);
        if (failed(requirement)) {
          return false;
        }
        Type requiredType = *requirement;
        if (requiredType != refinedExtractType) {
          return false;
        }
      }
      continue;
    }
    if (EmitEqualityOp equalityOp = llvm::dyn_cast<EmitEqualityOp>(user)) {
      Value otherOperand;
      if (equalityOp.getLhs() == result) {
        otherOperand = equalityOp.getRhs();
      } else if (equalityOp.getRhs() == result) {
        otherOperand = equalityOp.getLhs();
      } else {
        return false;
      }
      // constrain.eq is not retagged when a member read is refined. A concrete other operand
      // must therefore already accept the proposed read type; otherwise a generic read can be
      // changed to an incompatible concrete type after template instantiation.
      if (isConcreteType(otherOperand.getType(), /*allowStructParams=*/false) &&
          !canUseScalarizedValueAsType(
              refinedType, otherOperand.getType(), tracker, "UpdateMemberDefTypeFromWrite"
          )) {
        return false;
      }
      continue;
    }
    if (EmitContainmentOp containmentOp = llvm::dyn_cast<EmitContainmentOp>(user)) {
      // constrain.in is not retagged when a member read is refined. Validate both roles: a
      // refined containment LHS must still contain its RHS, and a refined RHS must still be
      // contained by its LHS. Otherwise dimensions that unify while generic can become
      // incompatible after refinement.
      if (containmentOp.getLhs() == result) {
        if (!areValidContainmentOperandTypes(refinedType, containmentOp.getRhs().getType())) {
          return false;
        }
      } else if (containmentOp.getRhs() == result) {
        if (!areValidContainmentOperandTypes(containmentOp.getLhs().getType(), refinedType)) {
          return false;
        }
      } else {
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
          castResultType, refinedType, tracker, "UpdateUnifiableCastResultFromInput"
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
    // Retagging a member read changes its SSA value but does not update arbitrary users. In
    // particular, pod.new verifies that every initializer exactly matches its declared record
    // type, so a read-derived refinement could otherwise leave the POD invalid. Keep the read
    // generic until each additional typed consumer has an explicit compatibility check here.
    return false;
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
