//===-- FlatteningScalarization.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Analysis and rewriting of heterogeneous arrays during flattening.
///
//===----------------------------------------------------------------------===//

#include "FlatteningScalarization.h"

#include "FlatteningContext.h"
#include "FlatteningTypeRefinement.h"

#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolLookup.h"
#include "llzk/Util/SymbolTableLLZK.h"
#include "llzk/Util/TypeHelper.h"
#include "llzk/Util/Walk.h"

#include <mlir/IR/Dominance.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>

using namespace mlir;
using namespace llzk;
using namespace llzk::array;
using namespace llzk::component;
using namespace llzk::function;
using namespace llzk::polymorphic;
using namespace llzk::polymorphic::detail;

namespace llzk::polymorphic::detail::flattening {

namespace {

/// Information about a local array allocation that can be replaced with the final values of its
/// statically-known elements.
///
/// This pass only scalarizes arrays after loop unrolling and affine-map instantiation have exposed
/// all element indices and value types. The candidate array must have either an initializer value
/// or exactly one write for every static element index, and all direct reads/member writes must
/// happen after every explicit element write. Those restrictions avoid imposing a new memory
/// semantics for partially initialized arrays, repeated writes, dynamic indices, or
/// branch-sensitive updates.
struct ScalarizedArrayInfo {
  /// The array allocation being removed.
  CreateArrayOp createOp;
  /// All static element indices in the array type, in the ArrayType's canonical order.
  SmallVector<ArrayAttr> indices;
  /// The final SSA value at each element index, from either the initializer or a later write.
  DenseMap<ArrayAttr, Value> valueByIndex;
  /// The type of the final value at each element index.
  DenseMap<ArrayAttr, Type> typeByIndex;
  /// The write operation that overwrites each element index, used for dominance-like ordering
  /// checks. Indices defined only by the array initializer have no entry.
  DenseMap<ArrayAttr, Operation *> writeOpByIndex;
  /// Discardable attributes from the write that supplies each element value. These are retained
  /// because the element writes are erased after their scalar member-write replacements are made.
  DenseMap<ArrayAttr, DictionaryAttr> writeDiscardableAttrsByIndex;
  /// Direct writes to the local allocation.
  SmallVector<WriteArrayOp> writes;
  /// Direct reads from the local allocation.
  SmallVector<ReadArrayOp> reads;
  /// Direct writes that store the whole local allocation into a struct member.
  SmallVector<MemberWriteOp> memberWrites;
};

/// Scalar member name and type.
using MemberInfo = std::pair<StringAttr, Type>;

/// Replacement scalar members for one array-typed member.
struct SplitMemberInfo {
  /// Static element indices that were split out of the original array-typed member.
  SmallVector<ArrayAttr> indices;
  /// Replacement scalar member for each static element index.
  DenseMap<ArrayAttr, MemberInfo> memberByIndex;
};

/// Return true iff `lhs` and `rhs` contain the same array indices.
static bool haveSameIndexSet(ArrayRef<ArrayAttr> lhs, ArrayRef<ArrayAttr> rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  DenseSet<ArrayAttr> rhsSet(rhs.begin(), rhs.end());
  return llvm::all_of(lhs, [&rhsSet](ArrayAttr idx) { return rhsSet.contains(idx); });
}

/// Return the first index in `lhs` that is not present in `rhs`.
static ArrayAttr findIndexMissingFrom(ArrayRef<ArrayAttr> lhs, ArrayRef<ArrayAttr> rhs) {
  DenseSet<ArrayAttr> rhsSet(rhs.begin(), rhs.end());
  const auto *it = llvm::find_if(lhs, [&rhsSet](ArrayAttr idx) { return !rhsSet.contains(idx); });
  return it == lhs.end() ? ArrayAttr() : *it;
}

/// Append indices from `source` that are not already present in `target`.
static void appendMissingIndices(SmallVector<ArrayAttr> &target, ArrayRef<ArrayAttr> source) {
  DenseSet<ArrayAttr> targetSet(target.begin(), target.end());
  for (ArrayAttr idx : source) {
    if (targetSet.insert(idx).second) {
      target.push_back(idx);
    }
  }
}

/// Replace all uses of `oldValue` without asking MLIR to enforce SSA type equality.
///
/// This is intentionally narrower than `replaceAllUsesWith()`: the whole purpose of this step is to
/// remove pseudo-homogeneous array values whose original element type no longer describes the
/// concrete value stored at each index. The caller must have already proven that every rewritten
/// use observes the value at a single static index, so replacing that use with the index-specific
/// value is type-correct for the consuming operation after the rewrite.
static inline void replaceAllUsesIgnoringType(Value oldValue, Value newValue) {
  for (OpOperand &use : llvm::make_early_inc_range(oldValue.getUses())) {
    use.set(newValue);
  }
}

/// Return true iff replacing `readOp` with a value of `replacementType` preserves the read result's
/// type refinement.
static inline bool canReplaceReadResultWithType(
    ReadArrayOp readOp, Type replacementType, const ConversionTracker &tracker, const char *patName
) {
  Type readResultType = readOp.getResult().getType();
  return readResultType == replacementType ||
         tracker.isLegalConversion(readResultType, replacementType, patName);
}

/// Return true iff direct typed consumers of `readOp` accept `replacementType`.
///
/// Scalarizing a pseudo-homogeneous array can replace a generic read result with the concrete
/// value stored at its static index. Checking the read result alone is insufficient: two
/// different concrete types can both unify with that generic result, while a consumer may require
/// only one of them. Typed consumers with separately declared requirements must be checked before
/// the type-ignoring replacement below.
static LogicalResult canReplaceReadUsersWithType(
    ReadArrayOp readOp, Type replacementType, SymbolTableCollection &tables,
    const DenseMap<Operation *, ScalarizedArrayInfo *> *candidateInfoByCreateOp = nullptr
) {
  Value result = readOp.getResult();
  for (OpOperand &use : result.getUses()) {
    Operation *user = use.getOwner();
    if (MemberWriteOp memberWrite = llvm::dyn_cast<MemberWriteOp>(user)) {
      if (memberWrite.getVal() != result) {
        return failure();
      }
      auto memberDef = memberWrite.getMemberDefOp(tables);
      if (failed(memberDef)) {
        return failure();
      }
      Type memberType = memberDef->get().getType();
      // A concrete member independently constrains which index-specific value is valid here.
      if (isConcreteType(memberType, /*allowStructParams=*/false) &&
          !typesUnify(replacementType, memberType)) {
        InFlightDiagnostic diag = readOp.emitError(
            "cannot scalarize heterogeneous array read because its index-specific value type is "
            "incompatible with a member write"
        );
        diag.attachNote(memberWrite.getLoc())
            << "member write requires " << memberType << ", but this read is replaced with "
            << replacementType;
        return diag;
      }

      if (!isConcreteType(memberType, /*allowStructParams=*/false)) {
        // A generic member may still be constrained by its other writes. Before scalarization,
        // both a generic read and another concrete write can unify with this member. Replacing
        // the read with an index-specific type must not turn those previously compatible writes
        // into incompatible refinement candidates after the source array has been erased.
        StructDefOp parentStruct = getParentOfType<StructDefOp>(memberDef->get());
        if (!parentStruct) {
          return failure();
        }
        auto memberUses = llzk::getSymbolUses(memberDef->get(), parentStruct);
        if (!memberUses) {
          return failure();
        }
        for (SymbolTable::SymbolUse memberUse : memberUses.value()) {
          MemberWriteOp otherWrite = llvm::dyn_cast<MemberWriteOp>(memberUse.getUser());
          if (!otherWrite || otherWrite == memberWrite) {
            continue;
          }
          Type otherWriteType = otherWrite.getVal().getType();
          if (!typesUnify(replacementType, otherWriteType)) {
            InFlightDiagnostic diag = readOp.emitError(
                "cannot scalarize heterogeneous array read because its index-specific value "
                "type is incompatible with another member write"
            );
            diag.attachNote(otherWrite.getLoc())
                << "other member write requires " << otherWriteType
                << ", but this read is replaced with " << replacementType;
            return diag;
          }
        }
      }
      continue;
    }
    if (CallOp call = llvm::dyn_cast<CallOp>(user)) {
      unsigned argIdx = use.getOperandNumber() - call.getArgOperands().getBeginOperandIndex();
      if (argIdx >= call.getArgOperands().size()) {
        return failure();
      }
      auto callee = call.getCalleeTarget(tables);
      if (failed(callee)) {
        return failure();
      }
      Type paramType = callee->get().getFunctionType().getInput(argIdx);
      if (isConcreteType(paramType, /*allowStructParams=*/false) &&
          !typesUnify(replacementType, paramType, callee->getNamespace())) {
        InFlightDiagnostic diag = readOp.emitError(
            "cannot scalarize heterogeneous array read because its index-specific value type is "
            "incompatible with a function call"
        );
        diag.attachNote(call.getLoc()) << "call argument requires " << paramType
                                       << ", but this read is replaced with " << replacementType;
        return diag;
      }
      continue;
    }
    if (WriteArrayOp arrayWrite = llvm::dyn_cast<WriteArrayOp>(user)) {
      if (arrayWrite.getRvalue() != result) {
        return failure();
      }
      Type elementType = arrayWrite.getArrRefType().getElementType();
      if (isConcreteType(elementType, /*allowStructParams=*/false) &&
          !typesUnify(replacementType, elementType)) {
        InFlightDiagnostic diag = readOp.emitError(
            "cannot scalarize heterogeneous array read because its index-specific value type is "
            "incompatible with an array write"
        );
        diag.attachNote(arrayWrite.getLoc())
            << "array write requires " << elementType << ", but this read is replaced with "
            << replacementType;
        return diag;
      }
      continue;
    }
    if (CreateArrayOp createArray = llvm::dyn_cast<CreateArrayOp>(user)) {
      if (use.getOperandNumber() >= createArray.getElements().size()) {
        return failure();
      }
      // A downstream scalarization candidate consumes the replacement while it is still valid,
      // then removes the initialized array. Its cached initializer values are refreshed from this
      // candidate before any rewrites run, so it is safe to let that supported path proceed.
      if (candidateInfoByCreateOp &&
          candidateInfoByCreateOp->contains(createArray.getOperation())) {
        continue;
      }
      Type elementType = createArray.getType().getElementType();
      // Unlike array.write, array.new requires each initializer to exactly match the result
      // array's element type. A generic read may be a valid initializer before scalarization,
      // even though its index-specific replacement is not.
      if (replacementType != elementType) {
        InFlightDiagnostic diag = readOp.emitError(
            "cannot scalarize heterogeneous array read because its index-specific value type is "
            "incompatible with an array initializer"
        );
        diag.attachNote(createArray.getLoc())
            << "array initializer requires exactly " << elementType
            << ", but this read is replaced with " << replacementType;
        return diag;
      }
      continue;
    }
    if (UnifiableCastOp castOp = llvm::dyn_cast<UnifiableCastOp>(user)) {
      if (castOp.getInput() != result) {
        return failure();
      }
      Type castResultType = castOp.getResult().getType();
      // The cast's result may independently require a concrete type. Replacing its generic input
      // with an index-specific value that cannot unify with that result would make the cast
      // invalid.
      if (isConcreteType(castResultType, /*allowStructParams=*/false) &&
          !typesUnify(replacementType, castResultType)) {
        InFlightDiagnostic diag = readOp.emitError(
            "cannot scalarize heterogeneous array read because its index-specific value type is "
            "incompatible with a unifiable cast"
        );
        diag.attachNote(castOp.getLoc()) << "unifiable cast result requires " << castResultType
                                         << ", but this read is replaced with " << replacementType;
        return diag;
      }
      continue;
    }
    if (MemberReadOp memberRead = llvm::dyn_cast<MemberReadOp>(user)) {
      if (memberRead.getComponent() != result) {
        return failure();
      }
      // The member-read propagation patterns will update the member result after this receiver
      // is retagged. Unlike region terminators, a member read does not retain an independent
      // result-type contract for the old generic receiver type.
      continue;
    }
    if (ReturnOp returnOp = llvm::dyn_cast<ReturnOp>(user)) {
      unsigned resultIdx = use.getOperandNumber();
      FuncDefOp function = returnOp->getParentOfType<FuncDefOp>();
      TypeRange resultTypes = function.getFunctionType().getResults();
      if (resultIdx >= resultTypes.size()) {
        return failure();
      }
      Type resultType = resultTypes[resultIdx];
      if (isConcreteType(resultType, /*allowStructParams=*/false) &&
          !typesUnify(replacementType, resultType)) {
        InFlightDiagnostic diag = readOp.emitError(
            "cannot scalarize heterogeneous array read because its index-specific value type is "
            "incompatible with a function return"
        );
        diag.attachNote(returnOp.getLoc())
            << "function return requires " << resultType << ", but this read is replaced with "
            << replacementType;
        return diag;
      }
      continue;
    }

    // `replaceAllUsesIgnoringType` deliberately bypasses MLIR's operand type check. Every
    // consumer therefore has to be accounted for here before scalarization can retarget a
    // generic read to its index-specific concrete value. In particular, region terminators
    // such as scf.yield impose a type relationship on their parent operation that this rewrite
    // does not update.
    InFlightDiagnostic diag = readOp.emitError(
        "cannot scalarize heterogeneous array read because it has an unsupported result user"
    );
    diag.attachNote(user->getLoc()) << "unsupported read result user is here";
    return diag;
  }
  return success();
}

/// Return true if `def` is in the same block as `user` and appears before it.
static inline bool strictlyBefore(Operation *def, Operation *user) {
  return def->getBlock() == user->getBlock() && def->isBeforeInBlock(user);
}

/// Return all direct users of `value`, sorted by their order in the containing block.
static FailureOr<SmallVector<Operation *>> getUsersInBlockOrder(Value value) {
  SmallVector<Operation *> users(value.getUsers().begin(), value.getUsers().end());
  if (users.empty()) {
    return users;
  }
  Block *block = users.front()->getBlock();
  if (!llvm::all_of(users, [block](Operation *user) { return user->getBlock() == block; })) {
    return failure();
  }
  llvm::sort(users, [](Operation *lhs, Operation *rhs) { return lhs->isBeforeInBlock(rhs); });
  return users;
}

/// Return true iff `createOp` has direct element writes whose order can update stored values.
static inline bool hasDirectArrayWrites(CreateArrayOp createOp) {
  Value arrayValue = createOp.getResult();
  return llvm::any_of(arrayValue.getUsers(), [arrayValue](Operation *user) {
    auto writeOp = llvm::dyn_cast<WriteArrayOp>(user);
    return writeOp && writeOp.getArrRef() == arrayValue;
  });
}

/// Return true iff `createOp` provides an initializer value for every static array element.
static inline bool hasFullInitializer(CreateArrayOp createOp, ArrayRef<ArrayAttr> indices) {
  Operation::operand_range elements = createOp.getElements();
  return !elements.empty() && elements.size() == indices.size();
}

/// Return true iff all element values are immutable initializer operands.
static inline bool
isInitializerOnlyLocalArray(CreateArrayOp createOp, ArrayRef<ArrayAttr> indices) {
  return hasFullInitializer(createOp, indices) && !hasDirectArrayWrites(createOp);
}

/// Return true iff the array allocation and every initializer value dominate `user`.
static inline bool initializerValuesDominateUse(
    CreateArrayOp createOp, Operation *user, const DominanceInfo &domInfo
) {
  if (!domInfo.dominates(createOp.getResult(), user)) {
    return false;
  }
  return llvm::all_of(createOp.getElements(), [user, &domInfo](Value element) {
    return domInfo.dominates(element, user);
  });
}

/// Return true if all writes for the scalarized allocation are available before `user`.
///
/// The rewrite currently handles straight-line local array construction. Requiring every write to
/// be in the same block and before the consuming read/member write keeps the replacement local and
/// avoids changing behavior for arrays updated through control flow.
static inline bool allWritesAvailableAt(const ScalarizedArrayInfo &info, Operation *user) {
  return llvm::all_of(info.writeOpByIndex, [user](const auto &entry) {
    return strictlyBefore(entry.second, user);
  });
}

/// Convert array access operands to a static index attribute, if possible.
static inline ArrayAttr getIndexAsAttr(ArrayAccessOpInterface op) {
  return op.indexOperandsToAttributeArray();
}

static Type
specializeTypeForArrayIndex(Type type, ArrayAttr idx, const ConversionTracker *tracker = nullptr);

/// Fold an affine-map attribute against the static array index, if the map only depends on that
/// index.
static Attribute
specializeAttrForArrayIndex(Attribute attr, ArrayAttr idx, const ConversionTracker *tracker) {
  if (!attr) {
    return attr;
  }
  if (auto mapAttr = llvm::dyn_cast<AffineMapAttr>(attr)) {
    AffineMap map = mapAttr.getAffineMap();
    if (idx.size() != map.getNumDims() + map.getNumSymbols()) {
      return attr;
    }
    SmallVector<Attribute> operands;
    operands.reserve(idx.size());
    for (Attribute idxPart : idx.getValue()) {
      auto intAttr = llvm::dyn_cast<IntegerAttr>(idxPart);
      if (!intAttr) {
        return attr;
      }
      operands.push_back(intAttr);
    }

    SmallVector<Attribute> result;
    bool hasPoison = false;
    if (failed(map.constantFold(operands, result, &hasPoison)) || hasPoison || result.size() != 1) {
      return attr;
    }
    return result.front();
  }
  if (auto typeAttr = llvm::dyn_cast<TypeAttr>(attr)) {
    Type specialized = specializeTypeForArrayIndex(typeAttr.getValue(), idx, tracker);
    return specialized == typeAttr.getValue() ? attr : TypeAttr::get(specialized);
  }
  if (auto arrayAttr = llvm::dyn_cast<ArrayAttr>(attr)) {
    MLIRContext *ctx = attr.getContext();
    SmallVector<Attribute> specializedAttrs;
    bool changed = false;
    for (Attribute nested : arrayAttr.getValue()) {
      Attribute specialized = specializeAttrForArrayIndex(nested, idx, tracker);
      specializedAttrs.push_back(specialized);
      changed |= specialized != nested;
    }
    return changed ? ArrayAttr::get(ctx, specializedAttrs) : attr;
  }
  return attr;
}

/// Specialize affine-map parameters nested in `type` for the given static array index.
static Type
specializeTypeForArrayIndex(Type type, ArrayAttr idx, const ConversionTracker *tracker) {
  if (auto structTy = llvm::dyn_cast<StructType>(type)) {
    ArrayAttr params = structTy.getParams();
    if (!params) {
      return type;
    }
    SmallVector<Attribute> specializedParams;
    bool changed = false;
    for (Attribute param : params.getValue()) {
      Attribute specialized = specializeAttrForArrayIndex(param, idx, tracker);
      specializedParams.push_back(specialized);
      changed |= specialized != param;
    }
    if (!changed) {
      return type;
    }
    Type specialized = StructType::get(
        structTy.getNameRef(), ArrayAttr::get(type.getContext(), specializedParams)
    );
    if (tracker) {
      if (auto structInstantiation =
              tracker->getInstantiation(llvm::cast<StructType>(specialized))) {
        return *structInstantiation;
      }
    }
    return specialized;
  }
  if (auto arrayTy = llvm::dyn_cast<ArrayType>(type)) {
    SmallVector<Attribute> specializedDims;
    bool dimsChanged = false;
    for (Attribute dim : arrayTy.getDimensionSizes()) {
      Attribute specialized = specializeAttrForArrayIndex(dim, idx, tracker);
      specializedDims.push_back(specialized);
      dimsChanged |= specialized != dim;
    }
    Type specializedElem = specializeTypeForArrayIndex(arrayTy.getElementType(), idx, tracker);
    return (dimsChanged || specializedElem != arrayTy.getElementType())
               ? ArrayType::get(specializedElem, specializedDims)
               : type;
  }
  if (auto podTy = llvm::dyn_cast<pod::PodType>(type)) {
    SmallVector<pod::RecordAttr> specializedRecords;
    bool changed = false;
    for (pod::RecordAttr record : podTy.getRecords()) {
      Type specialized = specializeTypeForArrayIndex(record.getType(), idx, tracker);
      specializedRecords.push_back(
          pod::RecordAttr::get(type.getContext(), record.getName(), specialized)
      );
      changed |= specialized != record.getType();
    }
    return changed ? pod::PodType::get(type.getContext(), specializedRecords) : type;
  }
  return type;
}

/// Seed the scalar value/type maps from the explicit elements of a statically-shaped `array.new`.
static LogicalResult seedValuesFromArrayElements(
    CreateArrayOp createOp, ArrayRef<ArrayAttr> indices, DenseMap<ArrayAttr, Value> &valueByIndex,
    DenseMap<ArrayAttr, Type> *typeByIndex = nullptr, const ConversionTracker *tracker = nullptr
) {
  Operation::operand_range elements = createOp.getElements();
  if (elements.empty()) {
    return success();
  }
  if (elements.size() != indices.size()) {
    return failure();
  }
  for (auto [idx, value] : llvm::zip_equal(indices, elements)) {
    valueByIndex[idx] = value;
    if (typeByIndex) {
      Type specializedType =
          specializeTypeForArrayIndex(createOp.getType().getElementType(), idx, tracker);
      bool canUseSpecializedType =
          typesUnify(value.getType(), specializedType) ||
          (tracker && tracker->isLegalConversion(
                          value.getType(), specializedType, "seedValuesFromArrayElements"
                      ));
      (*typeByIndex)[idx] = canUseSpecializedType ? specializedType : value.getType();
    }
  }
  return success();
}

/// Return true iff the candidate array stores at least two non-unifying element types.
///
/// Homogeneous arrays should continue through the normal type-propagation path. This step exists
/// specifically for pseudo-homogeneous arrays, such as arrays whose element is a templated struct
/// with an affine-map parameter that becomes a different concrete struct at each unrolled index.
static bool hasMultipleIncompatibleElementTypes(const ScalarizedArrayInfo &info) {
  SmallVector<Type> previousTypes;
  DenseSet<Type> seenTypes;
  for (ArrayAttr idx : info.indices) {
    Type nextType = info.typeByIndex.lookup(idx);
    if (!nextType) {
      return false;
    }
    // Repeated types add no compatibility information. Keep comparing every distinct pair:
    // unification is not transitive when a generic type unifies with incompatible concrete types.
    if (!seenTypes.insert(nextType).second) {
      continue;
    }
    for (Type previousType : previousTypes) {
      if (!typesUnify(previousType, nextType)) {
        return true;
      }
    }
    previousTypes.push_back(nextType);
  }
  return false;
}

/// Merge one candidate scalar type into the split type map, keeping the most concrete refinement.
static LogicalResult mergeSplitCandidateType(
    MemberDefOp member, ArrayAttr idx, Type candidateType, Operation *candidateOp,
    DenseMap<ArrayAttr, Type> &splitTypes, DenseMap<ArrayAttr, Operation *> *splitTypeOps,
    const ConversionTracker &tracker
) {
  auto existing = splitTypes.find(idx);
  if (existing == splitTypes.end()) {
    splitTypes[idx] = candidateType;
    if (splitTypeOps) {
      (*splitTypeOps)[idx] = candidateOp;
    }
    return success();
  }

  Type existingType = existing->second;
  if (existingType == candidateType) {
    return success();
  }
  if (tracker.isLegalConversion(existingType, candidateType, "mergeSplitCandidateType")) {
    existing->second = candidateType;
    if (splitTypeOps) {
      (*splitTypeOps)[idx] = candidateOp;
    }
    return success();
  }
  if (tracker.isLegalConversion(candidateType, existingType, "mergeSplitCandidateType") ||
      existingType == candidateType) {
    return success();
  }

  auto commonType = getCommonRefinedType(existingType, candidateType, tracker);
  if (succeeded(commonType)) {
    existing->second = *commonType;
    if (splitTypeOps) {
      (*splitTypeOps)[idx] = candidateOp;
    }
    return success();
  }

  InFlightDiagnostic diag = member.emitError(
      "cannot split heterogeneous array member because candidate writes require incompatible "
      "scalar member types"
  );
  if (splitTypeOps) {
    diag.attachNote(splitTypeOps->lookup(idx)->getLoc())
        << "candidate writes index " << idx << " with type " << existingType;
    diag.attachNote(candidateOp->getLoc())
        << "conflicting candidate writes the same index with type " << candidateType;
  }
  return diag;
}

/// Collect scalarization information for `op` if it is a safe heterogeneous-array candidate.
///
/// A candidate must:
/// - have a static shape and a real element type,
/// - have only direct reads, direct writes, and whole-array struct member writes as users,
/// - use only static array indices,
/// - have an initializer or exactly one write for every static element index,
/// - have all reads/member writes after every explicit element write, and
/// - store multiple incompatible element types.
static FailureOr<ScalarizedArrayInfo>
getScalarizedArrayInfo(CreateArrayOp op, const ConversionTracker &tracker) {
  ArrayType arrTy = op.getType();
  if (!arrTy.hasStaticShape() || llvm::isa<NoneType>(arrTy.getElementType())) {
    return failure();
  }

  std::optional<SmallVector<ArrayAttr>> maybeIndices = arrTy.getSubelementIndices();
  if (!maybeIndices) {
    return failure();
  }

  ScalarizedArrayInfo info;
  info.createOp = op;
  info.indices = std::move(*maybeIndices);
  auto seedRes =
      seedValuesFromArrayElements(op, info.indices, info.valueByIndex, &info.typeByIndex, &tracker);
  if (failed(seedRes)) {
    return failure();
  }
  Value arrayValue = op.getResult();

  for (Operation *user : arrayValue.getUsers()) {
    if (auto writeOp = llvm::dyn_cast<WriteArrayOp>(user)) {
      if (writeOp.getArrRef() != arrayValue) {
        return failure();
      }
      ArrayAttr idx = getIndexAsAttr(writeOp);
      if (!idx) {
        return failure();
      }
      if (info.writeOpByIndex.contains(idx)) {
        return failure();
      }
      info.valueByIndex[idx] = writeOp.getRvalue();
      info.typeByIndex[idx] = writeOp.getRvalue().getType();
      info.writeOpByIndex[idx] = writeOp.getOperation();
      info.writeDiscardableAttrsByIndex[idx] = writeOp->getDiscardableAttrDictionary();
      info.writes.push_back(writeOp);
      continue;
    }
    if (auto readOp = llvm::dyn_cast<ReadArrayOp>(user)) {
      if (readOp.getArrRef() != arrayValue) {
        return failure();
      }
      ArrayAttr idx = getIndexAsAttr(readOp);
      if (!idx) {
        return failure();
      }
      info.reads.push_back(readOp);
      continue;
    }
    if (auto memberWriteOp = llvm::dyn_cast<MemberWriteOp>(user)) {
      if (memberWriteOp.getVal() != arrayValue) {
        return failure();
      }
      info.memberWrites.push_back(memberWriteOp);
      continue;
    }
    return failure();
  }

  for (ArrayAttr idx : info.indices) {
    if (!info.valueByIndex.contains(idx)) {
      return failure();
    }
  }
  for (ReadArrayOp readOp : info.reads) {
    ArrayAttr idx = getIndexAsAttr(readOp);
    if (!info.valueByIndex.contains(idx) || !allWritesAvailableAt(info, readOp)) {
      return failure();
    }
    if (!canReplaceReadResultWithType(
            readOp, info.typeByIndex.lookup(idx), tracker, "getScalarizedArrayInfo"
        )) {
      return failure();
    }
  }
  for (MemberWriteOp memberWriteOp : info.memberWrites) {
    if (!allWritesAvailableAt(info, memberWriteOp)) {
      return failure();
    }
  }
  if (!hasMultipleIncompatibleElementTypes(info)) {
    return failure();
  }
  return info;
}

/// Create scalar replacement members for `member`, or return the replacements already created.
///
/// The replacement members preserve the original member's public/signal/column and discardable
/// metadata, and rely on the containing struct's symbol table to make each generated name unique.
static SplitMemberInfo &getOrCreateSplitMemberInfo(
    MemberDefOp member, ScalarizedArrayInfo &arrayInfo,
    const DenseMap<MemberDefOp, DenseMap<ArrayAttr, Type>> &splitTypesByMember,
    const DenseMap<MemberDefOp, SmallVector<ArrayAttr>> &splitIndicesByMember,
    DenseMap<MemberDefOp, SplitMemberInfo> &splitMembers, SymbolTableCollection &tables,
    PatternRewriter &rewriter, const ConversionTracker &tracker
) {
  auto existing = splitMembers.find(member);
  if (existing != splitMembers.end()) {
    return existing->second;
  }

  SplitMemberInfo &splitInfo = splitMembers[member];
  if (auto indicesIt = splitIndicesByMember.find(member); indicesIt != splitIndicesByMember.end()) {
    splitInfo.indices = indicesIt->second;
  } else {
    splitInfo.indices = arrayInfo.indices;
  }

  StructDefOp parentStruct = getParentOfType<StructDefOp>(member);
  assert(parentStruct && "MemberDefOp parent is always StructDefOp");
  SymbolTable &structSymbols = tables.getSymbolTable(parentStruct);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(member);
  auto splitTypesIt = splitTypesByMember.find(member);
  for (ArrayAttr idx : splitInfo.indices) {
    Type scalarType = arrayInfo.typeByIndex.lookup(idx);
    if (splitTypesIt != splitTypesByMember.end()) {
      if (Type refinedType = splitTypesIt->second.lookup(idx)) {
        scalarType = refinedType;
      }
    }
    if (!scalarType) {
      scalarType =
          specializeTypeForArrayIndex(arrayInfo.createOp.getType().getElementType(), idx, &tracker);
    }
    MemberDefOp newMember = MemberDefOp::create(
        rewriter, member.getLoc(), member.getSymNameAttr(), scalarType, member.getSignal(),
        member.getColumn()
    );
    newMember.setPublicAttr(member.hasPublicAttr());
    newMember->setDiscardableAttrs(member->getDiscardableAttrDictionary());
    StringAttr actualName = structSymbols.insert(newMember);
    splitInfo.memberByIndex[idx] = std::make_pair(actualName, scalarType);
  }
  return splitInfo;
}

/// Value and type cached for one scalarized array index.
struct CachedScalarizedValue {
  /// SSA value currently known to provide this scalarized index.
  Value value;
  /// Type that the scalarized index must use when splitting member writes.
  Type type;
};

/// Return the cached scalar value for `value` when it is a static read from another candidate.
static std::optional<CachedScalarizedValue> getCandidateReadReplacement(
    Value value, const DenseMap<Operation *, ScalarizedArrayInfo *> *candidateInfoByCreateOp
) {
  if (!candidateInfoByCreateOp) {
    return std::nullopt;
  }
  auto readOp = value.getDefiningOp<ReadArrayOp>();
  if (!readOp) {
    return std::nullopt;
  }
  auto createOp = readOp.getArrRef().getDefiningOp<CreateArrayOp>();
  if (!createOp) {
    return std::nullopt;
  }
  auto infoIt = candidateInfoByCreateOp->find(createOp.getOperation());
  if (infoIt == candidateInfoByCreateOp->end()) {
    return std::nullopt;
  }
  ArrayAttr idx = getIndexAsAttr(readOp);
  if (!idx) {
    return std::nullopt;
  }

  ScalarizedArrayInfo *info = infoIt->second;
  Value replacement = info->valueByIndex.lookup(idx);
  Type replacementType = info->typeByIndex.lookup(idx);
  if (!replacement || !replacementType) {
    return std::nullopt;
  }
  return CachedScalarizedValue {replacement, replacementType};
}

/// Refresh cached element values from their still-live array operands.
///
/// Candidates are collected before any rewrites run, so one candidate can cache a read result from
/// another candidate. Rewriting the upstream candidate updates the downstream write operands and
/// initializer operands and erases the reads; re-reading the operands here keeps this candidate
/// from using dangling values.
static LogicalResult refreshValuesFromArrayOperands(
    ScalarizedArrayInfo &info, const ConversionTracker &tracker,
    const DenseMap<Operation *, ScalarizedArrayInfo *> *candidateInfoByCreateOp = nullptr,
    bool *changed = nullptr
) {
  DenseMap<ArrayAttr, Value> oldValueByIndex;
  DenseMap<ArrayAttr, Type> oldTypeByIndex;
  if (changed != nullptr) {
    oldValueByIndex = info.valueByIndex;
    oldTypeByIndex = info.typeByIndex;
  }

  auto setCachedValue = [&](ArrayAttr idx, Value value, Type fallbackType) {
    if (std::optional<CachedScalarizedValue> replacement =
            getCandidateReadReplacement(value, candidateInfoByCreateOp)) {
      value = replacement->value;
      fallbackType = replacement->type;
    }
    info.valueByIndex[idx] = value;
    info.typeByIndex[idx] = fallbackType;
  };

  Operation::operand_range elements = info.createOp.getElements();
  if (!elements.empty() && elements.size() != info.indices.size()) {
    return failure();
  }
  if (!elements.empty()) {
    for (auto [idx, value] : llvm::zip_equal(info.indices, elements)) {
      Type specializedType =
          specializeTypeForArrayIndex(info.createOp.getType().getElementType(), idx, &tracker);
      bool canUseSpecializedType =
          typesUnify(value.getType(), specializedType) ||
          tracker.isLegalConversion(
              value.getType(), specializedType, "refreshValuesFromArrayOperands"
          );
      setCachedValue(idx, value, canUseSpecializedType ? specializedType : value.getType());
    }
  }

  for (WriteArrayOp writeOp : info.writes) {
    ArrayAttr idx = getIndexAsAttr(writeOp);
    if (!idx || !info.valueByIndex.contains(idx)) {
      return failure();
    }
    setCachedValue(idx, writeOp.getRvalue(), writeOp.getRvalue().getType());
  }
  if (changed != nullptr) {
    for (ArrayAttr idx : info.indices) {
      if (oldValueByIndex.lookup(idx) != info.valueByIndex.lookup(idx) ||
          oldTypeByIndex.lookup(idx) != info.typeByIndex.lookup(idx)) {
        *changed = true;
        break;
      }
    }
  }
  return success();
}

/// Refresh all collected candidates through candidate-to-candidate static reads.
static LogicalResult refreshValuesFromDependentCandidates(
    MutableArrayRef<ScalarizedArrayInfo> arraysToScalarize,
    DenseMap<Operation *, ScalarizedArrayInfo *> &candidateInfoByCreateOp,
    const ConversionTracker &tracker
) {
  candidateInfoByCreateOp.clear();
  for (ScalarizedArrayInfo &info : arraysToScalarize) {
    candidateInfoByCreateOp[info.createOp.getOperation()] = &info;
  }

  for (unsigned iteration = 0; iteration <= arraysToScalarize.size(); ++iteration) {
    bool changed = false;
    for (ScalarizedArrayInfo &info : arraysToScalarize) {
      auto res = refreshValuesFromArrayOperands(info, tracker, &candidateInfoByCreateOp, &changed);
      if (failed(res)) {
        return failure();
      }
    }
    if (!changed) {
      return success();
    }
  }
  return failure();
}

/// Verify that refreshing dependent candidates did not invalidate a direct read replacement.
///
/// A candidate can cache a generic read from another candidate as the value for one of its
/// indices. Refreshing the cache replaces that generic value and type with the upstream
/// candidate's concrete scalar value. Recheck all local reads before any candidates are rewritten
/// so an incompatible replacement cannot fail after earlier candidates have been erased.
static LogicalResult verifyRefreshedCandidateReads(
    ArrayRef<ScalarizedArrayInfo> arraysToScalarize, SymbolTableCollection &tables,
    const ConversionTracker &tracker,
    const DenseMap<Operation *, ScalarizedArrayInfo *> &candidateInfoByCreateOp
) {
  for (const ScalarizedArrayInfo &info : arraysToScalarize) {
    for (ReadArrayOp readOp : info.reads) {
      ArrayAttr idx = getIndexAsAttr(readOp);
      if (!idx) {
        return failure();
      }
      bool canReplaceResult = canReplaceReadResultWithType(
          readOp, info.typeByIndex.lookup(idx), tracker, "verifyRefreshedCandidateReads"
      );
      if (!canReplaceResult) {
        return failure();
      }
      auto canReplaceUsers = canReplaceReadUsersWithType(
          readOp, info.typeByIndex.lookup(idx), tables, &candidateInfoByCreateOp
      );
      if (failed(canReplaceUsers)) {
        return failure();
      }
    }
  }
  return success();
}

/// Collect the common specialized type required for each shared nondeterministic initializer.
///
/// An inline initializer can store the same generic witness at multiple array indices. Replacing
/// each read or split-member write with a separate typed witness would lose that sharing, so all
/// observed indices using one source witness must agree on a common refinement. In particular,
/// include merged split-member types: another candidate can refine a member even when this
/// candidate's local read remains generic.
static LogicalResult collectSharedNondetSpecializationTypes(
    ArrayRef<ScalarizedArrayInfo> infos, const ConversionTracker &tracker,
    const DenseMap<MemberDefOp, DenseMap<ArrayAttr, Type>> &splitTypesByMember,
    SymbolTableCollection &tables, DenseMap<Value, Type> &specializedTypeByNondet
) {
  struct FirstUse {
    ArrayAttr idx;
    Location loc;
  };
  DenseMap<Value, FirstUse> firstUseByNondet;
  auto collectType = [&](const ScalarizedArrayInfo &info, Value value, Type type, ArrayAttr idx,
                         Location loc) -> LogicalResult {
    if (!value || !type || value.getType() == type || !value.getDefiningOp<NonDetOp>()) {
      return success();
    }
    auto existing = specializedTypeByNondet.find(value);
    if (existing == specializedTypeByNondet.end()) {
      specializedTypeByNondet.try_emplace(value, type);
      firstUseByNondet.try_emplace(value, FirstUse {idx, loc});
      return success();
    }
    FailureOr<Type> commonType = getCommonRefinedType(existing->second, type, tracker);
    if (failed(commonType)) {
      auto firstUseIt = firstUseByNondet.find(value);
      assert(firstUseIt != firstUseByNondet.end() && "first use must be recorded");
      const FirstUse &firstUse = firstUseIt->second;
      InFlightDiagnostic diag = info.createOp->emitError(
          "cannot scalarize array because a shared nondeterministic initializer requires "
          "incompatible specialized types"
      );
      diag.attachNote(firstUse.loc) << "array index " << firstUse.idx
                                    << " is read with specialized type " << existing->second;
      diag.attachNote(loc) << "array index " << idx << " is read with specialized type " << type;
      return diag;
    }
    existing->second = *commonType;
    return success();
  };
  for (const ScalarizedArrayInfo &info : infos) {
    for (ReadArrayOp readOp : info.reads) {
      ArrayAttr idx = getIndexAsAttr(readOp);
      auto res = collectType(
          info, info.valueByIndex.lookup(idx), info.typeByIndex.lookup(idx), idx, readOp.getLoc()
      );
      if (failed(res)) {
        return failure();
      }
    }
    for (MemberWriteOp memberWriteOp : info.memberWrites) {
      auto memberDef = memberWriteOp.getMemberDefOp(tables);
      if (failed(memberDef)) {
        return failure();
      }
      for (const auto &[idx, type] : splitTypesByMember.lookup(memberDef->get())) {
        auto res =
            collectType(info, info.valueByIndex.lookup(idx), type, idx, memberWriteOp.getLoc());
        if (failed(res)) {
          return failure();
        }
      }
    }
  }
  return success();
}

/// Collect uses that candidate scalarization will rewrite internally.
static DenseSet<Operation *> collectInternalSharedNondetUses(ArrayRef<ScalarizedArrayInfo> infos) {
  DenseSet<Operation *> internalUses;
  for (const ScalarizedArrayInfo &info : infos) {
    internalUses.insert(info.createOp);
    internalUses.insert(info.writes.begin(), info.writes.end());
  }
  return internalUses;
}

/// Validate shared witness observations once, retaining the calls for materialization.
static FailureOr<DenseMap<Value, SmallVector<CallOp>>> collectSharedNondetExternalConstraints(
    ArrayRef<ScalarizedArrayInfo> infos, const DenseMap<Value, Type> &specializedTypeByNondet,
    SymbolTableCollection &tables
) {
  DenseSet<Operation *> internalUses = collectInternalSharedNondetUses(infos);
  DenseMap<Value, CreateArrayOp> candidateBySource;
  for (const ScalarizedArrayInfo &info : infos) {
    for (const auto &[_, value] : info.valueByIndex) {
      if (value.getDefiningOp<NonDetOp>()) {
        candidateBySource.try_emplace(value, info.createOp);
      }
    }
  }
  DenseMap<Value, SmallVector<CallOp>> externalCallsBySource;
  for (const auto &[source, type] : specializedTypeByNondet) {
    auto calls = collectSpecializedWitnessConstraints(source, type, tables, [&](OpOperand &use) {
      return internalUses.contains(use.getOwner());
    }, candidateBySource.lookup(source));
    if (failed(calls)) {
      return failure();
    }
    externalCallsBySource[source] = std::move(*calls);
  }
  return externalCallsBySource;
}

/// Materialize each refined witness once and redirect its already-validated external calls.
/// Place replacements after the source so they dominate calls preceding the array initializer.
static void materializeSharedNondetSpecializations(
    const DenseMap<Value, Type> &specializedTypeByNondet,
    const DenseMap<Value, SmallVector<CallOp>> &externalCallsBySource, PatternRewriter &rewriter,
    DenseMap<Value, Value> &specializedNondetBySource
) {
  for (const auto &[source, type] : specializedTypeByNondet) {
    NonDetOp sourceNondet = source.getDefiningOp<NonDetOp>();
    assert(sourceNondet && "specialization map only contains nondeterministic values");
    Value specializedValue = materializeSpecializedNondet(
        rewriter, sourceNondet, sourceNondet.getLoc(), type, externalCallsBySource.lookup(source)
    );
    specializedNondetBySource[source] = specializedValue;
  }
}

/// Return the value that should replace `readOp` after scalarization.
///
/// The cached scalar value can be less concrete than `info.typeByIndex` for inline initializers,
/// so materialize a cast when a concrete consumer needs the refined per-index type.
static FailureOr<Value> buildReadReplacementValue(
    ReadArrayOp readOp, const ScalarizedArrayInfo &info, PatternRewriter &rewriter,
    const ConversionTracker &tracker, const DenseMap<Value, Type> &specializedTypeByNondet,
    DenseMap<Value, Value> &specializedNondetBySource
) {
  ArrayAttr idx = getIndexAsAttr(readOp);
  if (!idx) {
    return failure();
  }
  Value replacementValue = info.valueByIndex.lookup(idx);
  Type replacementType = info.typeByIndex.lookup(idx);
  if (!replacementValue || !replacementType) {
    return failure();
  }
  if (!canUseScalarizedValueAsType(
          replacementValue.getType(), replacementType, tracker, "buildReadReplacementValue"
      )) {
    return failure();
  }
  if (replacementValue.getDefiningOp<NonDetOp>()) {
    // A generic read can require the shared witness refinement solely because another candidate
    // contributes a concrete type for the same split-member index.
    if (Value cachedNondet = specializedNondetBySource.lookup(replacementValue)) {
      return cachedNondet;
    }
  }
  if (replacementValue.getType() == replacementType) {
    return replacementValue;
  }

  if (NonDetOp sourceNondet = replacementValue.getDefiningOp<NonDetOp>()) {
    // Preserve sharing between every index initialized from the same generic witness.
    if (Value cachedNondet = specializedNondetBySource.lookup(replacementValue)) {
      return cachedNondet;
    }
    Type specializedType = specializedTypeByNondet.lookup(replacementValue);
    if (!specializedType) {
      return failure();
    }
    // The source witness dominates every original use, including external constraints that can
    // precede the array initializer. Materialize directly after it rather than at the first read.
    Value specializedValue =
        materializeSpecializedNondet(rewriter, sourceNondet, readOp.getLoc(), specializedType);
    specializedNondetBySource[replacementValue] = specializedValue;
    return specializedValue;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(readOp);
  if (!typesUnify(replacementValue.getType(), replacementType)) {
    return failure();
  }
  return UnifiableCastOp::create(rewriter, readOp.getLoc(), replacementType, replacementValue)
      .getResult();
}

/// Preserve one scalarized array read's discardable metadata on a distinct value carrier.
///
/// Scalarization normally replaces a read with a cached element SSA value. That value can serve
/// multiple reads, while each read may carry different provenance metadata. Materialize a trivial
/// unifiable cast only for annotated reads so every original attribute dictionary remains attached
/// to one replacement operation without changing unannotated IR.
static Value materializeReadMetadataCarrier(
    ReadArrayOp readOp, Value replacementValue, PatternRewriter &rewriter
) {
  DictionaryAttr discardableAttrs = readOp->getDiscardableAttrDictionary();
  if (discardableAttrs.empty()) {
    return replacementValue;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(readOp);
  UnifiableCastOp carrier = UnifiableCastOp::create(
      rewriter, readOp.getLoc(), replacementValue.getType(), replacementValue
  );
  carrier->setDiscardableAttrs(discardableAttrs);
  return carrier.getResult();
}

/// Emit one scalar `struct.writem` per split index for a whole-array member write.
template <typename GetScalarValueFn>
static LogicalResult emitScalarMemberWrites(
    MemberWriteOp memberWriteOp, const SplitMemberInfo &splitInfo,
    const DenseMap<ArrayAttr, DictionaryAttr> &writeDiscardableAttrsByIndex,
    PatternRewriter &rewriter, GetScalarValueFn getScalarValue
) {
  rewriter.setInsertionPoint(memberWriteOp);
  DictionaryAttr discardableAttrs = memberWriteOp->getDiscardableAttrDictionary();
  for (ArrayAttr idx : splitInfo.indices) {
    MemberInfo memberInfo = splitInfo.memberByIndex.lookup(idx);
    if (!memberInfo.first) {
      return failure();
    }
    FailureOr<Value> scalarValue = getScalarValue(idx, memberInfo.second);
    if (failed(scalarValue)) {
      return failure();
    }
    MemberWriteOp scalarWrite = MemberWriteOp::create(
        rewriter, memberWriteOp.getLoc(), memberWriteOp.getComponent(),
        FlatSymbolRefAttr::get(memberInfo.first), *scalarValue
    );
    scalarWrite->setDiscardableAttrs(discardableAttrs);
    if (DictionaryAttr writeAttrs = writeDiscardableAttrsByIndex.lookup(idx)) {
      for (NamedAttribute attr : writeAttrs) {
        // The element write is the effective replacement, so preserve its provenance if the
        // whole-array and element writes use the same discardable metadata key.
        scalarWrite->setDiscardableAttr(attr.getName(), attr.getValue());
      }
    }
  }
  return success();
}

/// Return the value for `idx` at `type`, creating and caching a nondeterministic value if the
/// local array element has not been written.
///
/// The caller must have already rejected initialized expandable arrays that store the same SSA
/// value into indices requiring incompatible split-member types.
static FailureOr<Value> getOrCreateScalarizedLocalArrayValue(
    DenseMap<ArrayAttr, Value> &valueByIndex, DenseSet<Value> &generatedMaterializations,
    ArrayAttr idx, Type type, Location loc, PatternRewriter &rewriter,
    const ConversionTracker &tracker, Operation *materializationPoint = nullptr
) {
  auto createNonDet = [&](Type nondetType, NonDetOp sourceNondet = nullptr) {
    OpBuilder::InsertionGuard guard(rewriter);
    if (materializationPoint) {
      rewriter.setInsertionPointAfter(materializationPoint);
    }
    NonDetOp nondet = NonDetOp::create(rewriter, loc, nondetType);
    if (sourceNondet) {
      nondet->setDiscardableAttrs(sourceNondet->getDiscardableAttrDictionary());
    }
    return nondet.getResult();
  };
  Value scalarValue = valueByIndex.lookup(idx);
  if (!scalarValue) {
    scalarValue = createNonDet(type);
    valueByIndex[idx] = scalarValue;
    generatedMaterializations.insert(scalarValue);
  }
  if (scalarValue.getType() == type) {
    return scalarValue;
  }
  FailureOr<Type> commonType = getCommonRefinedType(scalarValue.getType(), type, tracker);
  if (failed(commonType)) {
    return failure();
  }
  if (*commonType == scalarValue.getType()) {
    return scalarValue;
  }
  if (generatedMaterializations.contains(scalarValue)) {
    NonDetOp nondetOp = scalarValue.getDefiningOp<NonDetOp>();
    if (!nondetOp) {
      return failure();
    }
    rewriter.modifyOpInPlace(nondetOp, [&scalarValue, &commonType]() {
      scalarValue.setType(*commonType);
    });
    return scalarValue;
  }
  if (NonDetOp sourceNondet = scalarValue.getDefiningOp<NonDetOp>()) {
    scalarValue = createNonDet(*commonType, sourceNondet);
    valueByIndex[idx] = scalarValue;
    generatedMaterializations.insert(scalarValue);
  }
  return scalarValue;
}

/// Rewrite one local heterogeneous array allocation into its index-specific scalar values.
///
/// Direct array reads are replaced with the value written at the requested static index.
/// Whole-array member writes are expanded into one scalar member write per index, creating
/// replacement members as needed. Once all consumers are rewritten, the original array writes and
/// allocation are erased.
static LogicalResult rewriteLocalArray(
    ScalarizedArrayInfo &info, DenseMap<MemberDefOp, SplitMemberInfo> &splitMembers,
    const DenseMap<MemberDefOp, DenseMap<ArrayAttr, Type>> &splitTypesByMember,
    const DenseMap<MemberDefOp, SmallVector<ArrayAttr>> &splitIndicesByMember,
    SymbolTableCollection &tables, PatternRewriter &rewriter, const ConversionTracker &tracker,
    const DenseMap<Operation *, ScalarizedArrayInfo *> *candidateInfoByCreateOp,
    const DenseMap<Value, Type> &specializedTypeByNondet,
    DenseMap<Value, Value> &specializedNondetBySource
) {
  if (failed(refreshValuesFromArrayOperands(info, tracker, candidateInfoByCreateOp))) {
    return failure();
  }

  for (ReadArrayOp readOp : llvm::make_early_inc_range(info.reads)) {
    ArrayAttr idx = getIndexAsAttr(readOp);
    auto canReplaceResult = canReplaceReadResultWithType(
        readOp, info.typeByIndex.lookup(idx), tracker, "rewriteLocalArray"
    );
    if (!canReplaceResult) {
      return failure();
    }
    auto canReplaceUsers = canReplaceReadUsersWithType(
        readOp, info.typeByIndex.lookup(idx), tables, candidateInfoByCreateOp
    );
    if (failed(canReplaceUsers)) {
      return failure();
    }
    FailureOr<Value> replacementValue = buildReadReplacementValue(
        readOp, info, rewriter, tracker, specializedTypeByNondet, specializedNondetBySource
    );
    if (failed(replacementValue)) {
      return failure();
    }
    Value replacementWithMetadata =
        materializeReadMetadataCarrier(readOp, *replacementValue, rewriter);
    replaceAllUsesIgnoringType(readOp.getResult(), replacementWithMetadata);
    rewriter.eraseOp(readOp);
  }

  for (MemberWriteOp memberWriteOp : llvm::make_early_inc_range(info.memberWrites)) {
    auto memberDef = memberWriteOp.getMemberDefOp(tables);
    if (failed(memberDef)) {
      return failure();
    }
    SplitMemberInfo &splitInfo = getOrCreateSplitMemberInfo(
        memberDef->get(), info, splitTypesByMember, splitIndicesByMember, splitMembers, tables,
        rewriter, tracker
    );

    DenseSet<Value> generatedMaterializations;
    Location memberWriteLoc = memberWriteOp.getLoc();
    auto getScalarValue = [&](ArrayAttr idx, Type type) -> FailureOr<Value> {
      if (Value scalarValue = info.valueByIndex.lookup(idx)) {
        // Reads of a generic initializer materialize one refined witness. Reuse it for the
        // corresponding split-member write so both uses retain the original array equality.
        if (Value specializedNondet = specializedNondetBySource.lookup(scalarValue)) {
          return specializedNondet;
        }
        return scalarValue;
      }
      return getOrCreateScalarizedLocalArrayValue(
          info.valueByIndex, generatedMaterializations, idx, type, memberWriteLoc, rewriter, tracker
      );
    };
    auto emitRes = emitScalarMemberWrites(
        memberWriteOp, splitInfo, info.writeDiscardableAttrsByIndex, rewriter, getScalarValue
    );
    if (failed(emitRes)) {
      return failure();
    }
    rewriter.eraseOp(memberWriteOp);
  }

  for (WriteArrayOp writeOp : llvm::make_early_inc_range(info.writes)) {
    rewriter.eraseOp(writeOp);
  }
  if (info.createOp.getResult().use_empty()) {
    rewriter.eraseOp(info.createOp);
  }
  return success();
}

/// Rewrite reads from array-typed members that were split by `rewriteLocalArray()`.
///
/// The only supported use of the original whole-array member read is a static `array.read`.
/// Supporting arbitrary uses would require reconstructing a pseudo-homogeneous array value, which
/// is exactly the invalid representation this step removes.
static LogicalResult rewriteSplitMemberReads(
    ModuleOp modOp, DenseMap<MemberDefOp, SplitMemberInfo> &splitMembers,
    SymbolTableCollection &tables, PatternRewriter &rewriter, const ConversionTracker &tracker
) {
  for (MemberReadOp memberReadOp : walkCollect<MemberReadOp>(modOp)) {
    auto memberDef = memberReadOp.getMemberDefOp(tables);
    if (failed(memberDef)) {
      return failure();
    }
    auto splitIt = splitMembers.find(memberDef->get());
    if (splitIt == splitMembers.end()) {
      continue;
    }

    SmallVector<ReadArrayOp> arrayReads;
    for (Operation *user : memberReadOp.getResult().getUsers()) {
      auto readOp = llvm::dyn_cast<ReadArrayOp>(user);
      if (!readOp || readOp.getArrRef() != memberReadOp.getResult() || !getIndexAsAttr(readOp)) {
        return failure();
      }
      arrayReads.push_back(readOp);
    }

    ValueRange mapOperands;
    std::optional<int32_t> numDims;
    if (!memberReadOp.getMapOperands().empty()) {
      mapOperands = memberReadOp.getMapOperands().front();
      numDims = memberReadOp.getNumDimsPerMap().front();
    }

    // A scalar read can be shared only when the source reads have identical
    // discardable attributes. These attributes are preserved on the replacement
    // operation, so sharing reads with different dictionaries would lose the
    // metadata from every read except the last one processed.
    DenseMap<ArrayAttr, SmallVector<std::pair<DictionaryAttr, Value>>> scalarValuesByIndex;
    DenseMap<Operation *, Value> scalarValueByRead;
    rewriter.setInsertionPoint(memberReadOp);
    DictionaryAttr discardableAttrs = memberReadOp->getDiscardableAttrDictionary();
    for (ReadArrayOp readOp : arrayReads) {
      ArrayAttr idx = getIndexAsAttr(readOp);
      MemberInfo memberInfo = splitIt->second.memberByIndex.lookup(idx);
      if (!memberInfo.first) {
        return failure();
      }
      if (!canReplaceReadResultWithType(
              readOp, memberInfo.second, tracker, "rewriteSplitMemberReads"
          )) {
        return failure();
      }
      if (failed(canReplaceReadUsersWithType(readOp, memberInfo.second, tables))) {
        return failure();
      }
      MemberReadOp scalarRead;
      DictionaryAttr readAttrs = readOp->getDiscardableAttrDictionary();
      auto &cachedReads = scalarValuesByIndex[idx];
      auto *cachedIt = llvm::find_if(cachedReads, [&readAttrs](const auto &cachedRead) {
        return cachedRead.first == readAttrs;
      });
      if (cachedIt != cachedReads.end()) {
        scalarRead = llvm::cast<MemberReadOp>(cachedIt->second.getDefiningOp());
      } else {
        scalarRead = MemberReadOp::create(
            rewriter, memberReadOp.getLoc(), memberInfo.second, memberReadOp.getComponent(),
            memberInfo.first, memberReadOp.getTableOffset().value_or(Attribute {}), mapOperands,
            numDims
        );
        scalarRead->setDiscardableAttrs(discardableAttrs);
        // Preserve provenance and other discardable metadata from the static array read that this
        // scalar member read replaces. Per-read metadata takes precedence over metadata inherited
        // from the whole-array member read.
        for (NamedAttribute attr : readAttrs.getValue()) {
          scalarRead->setDiscardableAttr(attr.getName(), attr.getValue());
        }
        cachedReads.emplace_back(readAttrs, scalarRead.getResult());
      }
      scalarValueByRead[readOp] = scalarRead.getResult();
    }

    for (ReadArrayOp readOp : llvm::make_early_inc_range(arrayReads)) {
      replaceAllUsesIgnoringType(readOp.getResult(), scalarValueByRead.lookup(readOp));
      rewriter.eraseOp(readOp);
    }
    if (memberReadOp.getResult().use_empty()) {
      rewriter.eraseOp(memberReadOp);
    }
  }
  return success();
}

/// Reconstruct the scalar member type map established by the collected candidates.
static LogicalResult collectSplitTypesByMember(
    ArrayRef<ScalarizedArrayInfo> arraysToScalarize, SymbolTableCollection &tables,
    DenseMap<MemberDefOp, SmallVector<ArrayAttr>> &splitIndicesByMember,
    DenseMap<MemberDefOp, DenseMap<ArrayAttr, Type>> &splitTypesByMember,
    const ConversionTracker &tracker
) {
  for (const ScalarizedArrayInfo &info : arraysToScalarize) {
    for (MemberWriteOp memberWriteOp : info.memberWrites) {
      auto memberDef = memberWriteOp.getMemberDefOp(tables);
      if (failed(memberDef)) {
        return failure();
      }
      MemberDefOp member = memberDef->get();
      appendMissingIndices(splitIndicesByMember[member], info.indices);
      DenseMap<ArrayAttr, Type> &splitTypes = splitTypesByMember[member];
      for (ArrayAttr idx : info.indices) {
        auto res = mergeSplitCandidateType(
            member, idx, info.typeByIndex.lookup(idx), memberWriteOp.getOperation(), splitTypes,
            nullptr, tracker
        );
        if (failed(res)) {
          return failure();
        }
      }
    }
  }
  return success();
}

/// Verify one read of an array-typed member before the defining member is split.
static LogicalResult verifySplitMemberReadRewritable(
    MemberReadOp memberReadOp, MemberDefOp member, const DenseMap<ArrayAttr, Type> &splitTypes,
    SymbolTableCollection &tables, const ConversionTracker &tracker
) {
  for (Operation *user : memberReadOp.getResult().getUsers()) {
    auto readOp = llvm::dyn_cast<ReadArrayOp>(user);
    if (!readOp || readOp.getArrRef() != memberReadOp.getResult()) {
      InFlightDiagnostic diag = memberReadOp.emitError(
          "cannot split heterogeneous array member because a whole-array read has an "
          "unsupported use"
      );
      diag.attachNote(member.getLoc()) << "split member defined here";
      diag.attachNote(user->getLoc()) << "unsupported use is here";
      return diag;
    }

    ArrayAttr idx = getIndexAsAttr(readOp);
    if (!idx) {
      InFlightDiagnostic diag = readOp.emitError(
          "cannot split heterogeneous array member because a read of it uses a dynamic array index"
      );
      diag.attachNote(member.getLoc()) << "split member defined here";
      return diag;
    }

    Type replacementType = splitTypes.lookup(idx);
    if (!replacementType) {
      InFlightDiagnostic diag = readOp.emitError(
          "cannot split heterogeneous array member because a read of it uses an array index "
          "that is not written by any scalarization candidate"
      );
      diag.attachNote(member.getLoc()) << "split member defined here";
      return diag;
    }

    if (!canReplaceReadResultWithType(
            readOp, replacementType, tracker, "verifySplitMemberReadsRewritable"
        )) {
      InFlightDiagnostic diag = readOp.emitError(
          "cannot split heterogeneous array member because a read result type is incompatible "
          "with the split scalar member type"
      );
      diag.attachNote(member.getLoc()) << "split member defined here";
      return diag;
    }

    if (failed(canReplaceReadUsersWithType(readOp, replacementType, tables))) {
      return failure();
    }
  }
  return success();
}

/// Verify reads of array-typed members before their defining members are split.
static LogicalResult verifySplitMemberReadsRewritable(
    ModuleOp modOp, const DenseMap<MemberDefOp, DenseMap<ArrayAttr, Type>> &splitTypesByMember,
    SymbolTableCollection &tables, const ConversionTracker &tracker
) {
  for (MemberReadOp memberReadOp : walkCollect<MemberReadOp>(modOp)) {
    auto memberDef = memberReadOp.getMemberDefOp(tables);
    if (failed(memberDef)) {
      return failure();
    }
    auto splitIt = splitTypesByMember.find(memberDef->get());
    if (splitIt == splitTypesByMember.end()) {
      continue;
    }
    auto verifyRes = verifySplitMemberReadRewritable(
        memberReadOp, memberDef->get(), splitIt->second, tables, tracker
    );
    if (failed(verifyRes)) {
      return failure();
    }
  }
  return success();
}

/// Return the local array allocation stored by `memberWriteOp`, if it has a static shape.
static FailureOr<CreateArrayOp> getStaticLocalArrayCreate(MemberWriteOp memberWriteOp) {
  auto createOp = memberWriteOp.getVal().getDefiningOp<CreateArrayOp>();
  if (!createOp) {
    return failure();
  }
  ArrayType arrTy = createOp.getType();
  if (!arrTy.hasStaticShape() || llvm::isa<NoneType>(arrTy.getElementType())) {
    return failure();
  }
  return createOp;
}

/// Verify that `createOp` can be expanded while splitting one or more members.
///
/// Static writes update the local value map, static reads consume it, and missing indices are
/// materialized as shared `llzk.nondet` values. Other users would observe or escape the array in a
/// way this local scalarization cannot preserve.
static LogicalResult verifyExpandableLocalArrayUsers(
    CreateArrayOp createOp, const DenseSet<MemberDefOp> &splitMemberSet,
    SymbolTableCollection &tables, const DominanceInfo &domInfo
) {
  Value arrayValue = createOp.getResult();
  ArrayType arrTy = createOp.getType();
  std::optional<SmallVector<ArrayAttr>> maybeIndices = arrTy.getSubelementIndices();
  if (!maybeIndices) {
    return failure();
  }
  bool allowCrossBlockMemberWrites = isInitializerOnlyLocalArray(createOp, *maybeIndices);
  for (Operation *user : createOp.getResult().getUsers()) {
    if (auto writeOp = llvm::dyn_cast<WriteArrayOp>(user)) {
      if (writeOp.getArrRef() != arrayValue || writeOp->getBlock() != createOp->getBlock() ||
          !getIndexAsAttr(writeOp)) {
        return failure();
      }
      continue;
    }
    if (auto readOp = llvm::dyn_cast<ReadArrayOp>(user)) {
      if (readOp.getArrRef() != arrayValue || readOp->getBlock() != createOp->getBlock() ||
          !getIndexAsAttr(readOp)) {
        return failure();
      }
      continue;
    }
    if (auto userWriteOp = llvm::dyn_cast<MemberWriteOp>(user)) {
      auto memberDef = userWriteOp.getMemberDefOp(tables);
      if (userWriteOp.getVal() == arrayValue && succeeded(memberDef) &&
          splitMemberSet.contains(memberDef->get())) {
        if (userWriteOp->getBlock() == createOp->getBlock()) {
          continue;
        }
        if (allowCrossBlockMemberWrites &&
            initializerValuesDominateUse(createOp, userWriteOp, domInfo)) {
          continue;
        }
      }
    }
    return failure();
  }
  return success();
}

/// Verify that every split-member write fed by `createOp` has the same static index set as the
/// local array allocation.
///
/// `rewriteExpandableLocalArray()` emits one scalar write per target split index. If the source
/// allocation has extra indices, those values would be dropped; if it has missing indices, fresh
/// nondeterministic values would be materialized. Either changes the whole-array write semantics.
static LogicalResult verifyExpandableLocalArraySplitIndices(
    CreateArrayOp createOp,
    const DenseMap<MemberDefOp, SmallVector<ArrayAttr>> &splitIndicesByMember,
    const DenseMap<MemberDefOp, Operation *> *splitIndexOpsByMember, SymbolTableCollection &tables
) {
  ArrayType arrTy = createOp.getType();
  std::optional<SmallVector<ArrayAttr>> maybeIndices = arrTy.getSubelementIndices();
  if (!maybeIndices) {
    return failure();
  }
  ArrayRef<ArrayAttr> indices = *maybeIndices;
  Value arrayValue = createOp.getResult();

  for (Operation *user : arrayValue.getUsers()) {
    auto memberWriteOp = llvm::dyn_cast<MemberWriteOp>(user);
    if (!memberWriteOp || memberWriteOp.getVal() != arrayValue) {
      continue;
    }
    auto memberDef = memberWriteOp.getMemberDefOp(tables);
    if (failed(memberDef)) {
      return failure();
    }
    MemberDefOp member = memberDef->get();
    auto splitIt = splitIndicesByMember.find(member);
    if (splitIt == splitIndicesByMember.end()) {
      continue;
    }
    ArrayRef<ArrayAttr> splitIndices = splitIt->second;
    if (haveSameIndexSet(indices, splitIndices)) {
      continue;
    }

    InFlightDiagnostic diag = member.emitError(
        "cannot split heterogeneous array member because expandable whole-array write uses "
        "different index set"
    );
    if (splitIndexOpsByMember) {
      diag.attachNote(splitIndexOpsByMember->lookup(member)->getLoc())
          << "candidate establishes " << splitIndices.size() << " split member indices";
    }
    Diagnostic &note = diag.attachNote(memberWriteOp.getLoc())
                       << "expandable whole-array write has " << indices.size() << " array indices";
    if (ArrayAttr extraIndex = findIndexMissingFrom(indices, splitIndices)) {
      note << ", including extra index " << extraIndex;
    } else if (ArrayAttr missingIndex = findIndexMissingFrom(splitIndices, indices)) {
      note << ", missing index " << missingIndex;
    }
    return diag;
  }
  return success();
}

/// Validated decisions for one expandable array, retained until its rewrite.
struct ExpandableArrayPlan {
  /// Original array users in execution order (or initializer-only cross-block users).
  SmallVector<Operation *> users;
  /// Final shared scalar type for each read, including later uses before an overwrite.
  DenseMap<Operation *, Type> materializationTypeByRead;
  /// Stable source operands and requested types for initialized scalar witnesses. Operands belong
  /// to this array's create/write operations and follow RAUW by earlier candidate rewrites; storing
  /// transient read results here would leave dangling Value handles when those reads are erased.
  SmallVector<std::pair<OpOperand *, Type>> witnessRequests;
  /// Cross-block initializer materializations must dominate all uses of the allocation.
  Operation *materializationPoint = nullptr;
};

/// Plan scalar replacements while rejecting uses that cannot share one materialized value.
///
/// Later type propagation may refine the value operand of each scalar member write. If two writes
/// keep the same SSA value, refining one write also refines the other because MLIR values carry one
/// global type. Missing values are materialized as fresh `llzk.nondet` values during rewriting and
/// then cached by index, so the check tracks those pending materializations until an `array.write`
/// overwrites the index. Static reads of unwritten indices participate in the same pending
/// materialization state because the rewrite caches their result-typed `llzk.nondet` value for
/// later scalar member writes at that index.
static LogicalResult planExpandableLocalArray(
    CreateArrayOp createOp,
    const DenseMap<MemberDefOp, DenseMap<ArrayAttr, Type>> &splitTypesByMember,
    SymbolTableCollection &tables, const ConversionTracker &tracker, ExpandableArrayPlan &plan,
    StringRef arrayDescription = "an expandable array"
) {
  if (splitTypesByMember.empty()) {
    return success();
  }

  bool targetsSplitMember = false;
  for (Operation *user : createOp.getResult().getUsers()) {
    auto memberWriteOp = llvm::dyn_cast<MemberWriteOp>(user);
    if (!memberWriteOp || memberWriteOp.getVal() != createOp.getResult()) {
      continue;
    }
    auto memberDef = memberWriteOp.getMemberDefOp(tables);
    if (failed(memberDef)) {
      return failure();
    }
    if (splitTypesByMember.contains(memberDef->get())) {
      targetsSplitMember = true;
      break;
    }
  }
  if (!targetsSplitMember) {
    return success();
  }

  ArrayType arrTy = createOp.getType();
  std::optional<SmallVector<ArrayAttr>> maybeIndices = arrTy.getSubelementIndices();
  if (!maybeIndices) {
    return failure();
  }
  ArrayRef<ArrayAttr> indices = *maybeIndices;

  DenseMap<ArrayAttr, OpOperand *> sourceByIndex;
  if (!createOp.getElements().empty()) {
    if (createOp.getElements().size() != indices.size()) {
      return failure();
    }
    unsigned firstElement = createOp.getElements().getBeginOperandIndex();
    for (auto [i, idx] : llvm::enumerate(indices)) {
      sourceByIndex[idx] = &createOp->getOpOperand(firstElement + i);
    }
  }

  FailureOr<SmallVector<Operation *>> maybeUsers = getUsersInBlockOrder(createOp.getResult());
  SmallVector<Operation *> &users = plan.users;
  if (succeeded(maybeUsers)) {
    users = std::move(*maybeUsers);
  } else {
    if (hasDirectArrayWrites(createOp)) {
      return failure();
    }
    users.assign(createOp.getResult().user_begin(), createOp.getResult().user_end());
    plan.materializationPoint = createOp.getOperation();
  }

  /// First consumer that forced an array index to be materialized at a concrete type.
  ///
  /// Rewriting creates and caches one `llzk.nondet` per unwritten index. It also caches the
  /// refined type of a read from a stored `llzk.nondet` value. Recording the first requested type
  /// lets verification reject a later read or split-member write that would need the same cached
  /// value at an incompatible type. A stored `llzk.nondet` can be materialized at a common
  /// refinement, as can a `poly.unifiable_cast` whose input can supply that refinement. Other
  /// non-nondeterministic stored values cannot serve distinct split-member types.
  struct PendingMaterializationUse {
    /// Concrete type requested for an index before a write defines a new value there.
    Type type;
    /// Operation location used to explain the first pending materialization request.
    Location loc;
    /// Short diagnostic phrase naming the consumer kind that requested `type`.
    StringRef description;
    /// Diagnostic phrase naming whether this was an unwritten or already-stored index.
    StringRef indexDescription;
  };

  DenseMap<Value, std::pair<Type, Location>> firstUseByValue;
  DenseMap<ArrayAttr, PendingMaterializationUse> firstMaterializedUseByIndex;
  DenseMap<ArrayAttr, SmallVector<ReadArrayOp>> readsByIndex;
  auto valueCompatibleWithTargetType = [&tracker](Value scalarValue, Type targetType) {
    return canUseScalarizedValueAsType(
        scalarValue.getType(), targetType, tracker, "planExpandableLocalArray"
    );
  };
  auto verifyExternalMemberWriteUses = [&](Value scalarValue, ArrayAttr idx, Type targetType,
                                           Location targetLoc) -> LogicalResult {
    for (OpOperand &use : scalarValue.getUses()) {
      auto externalWrite = llvm::dyn_cast<MemberWriteOp>(use.getOwner());
      if (!externalWrite || externalWrite.getVal() != use.get()) {
        continue;
      }
      auto memberDef = externalWrite.getMemberDefOp(tables);
      if (failed(memberDef)) {
        return failure();
      }
      Type externalType = memberDef->get().getType();
      if (succeeded(getCommonRefinedType(externalType, targetType, tracker))) {
        continue;
      }

      InFlightDiagnostic diag =
          createOp.emitError("cannot split heterogeneous array member because ")
          << arrayDescription << " reuses one SSA value for incompatible scalar member types";
      diag.attachNote(externalWrite.getLoc()) << "value is used for member type " << externalType;
      diag.attachNote(targetLoc) << "same value is also used for array index " << idx
                                 << " scalar member type " << targetType;
      return diag;
    }

    // Refining a stored generic nondet creates a replacement witness during rewriting. Every
    // observation outside this array's initializer/write path would otherwise still refer to the
    // original generic witness, splitting one witness into two independent values.
    if (scalarValue.getType() != targetType && scalarValue.getDefiningOp<NonDetOp>()) {
      FailureOr<Type> commonType = getCommonRefinedType(scalarValue.getType(), targetType, tracker);
      if (succeeded(commonType) && *commonType != scalarValue.getType()) {
        for (OpOperand &use : scalarValue.getUses()) {
          Operation *owner = use.getOwner();
          if (owner == createOp.getOperation()) {
            continue;
          }
          auto arrayWrite = llvm::dyn_cast<WriteArrayOp>(owner);
          if (arrayWrite && arrayWrite.getRvalue() == scalarValue &&
              arrayWrite.getArrRef() == createOp.getResult()) {
            continue;
          }
          // Struct constraint calls are retargeted to the specialized witness by
          // rewriteExpandableLocalArray(). Other observations must retain the original witness;
          // reject them before any candidate or expandable-array rewrite can create a clone.
          if (use.getOperandNumber() == 0 && llvm::isa<StructType>(*commonType)) {
            if (auto call = llvm::dyn_cast<CallOp>(owner); call && call.calleeIsStructConstrain()) {
              continue;
            }
          }
          InFlightDiagnostic diag = createOp.emitError(
              "cannot scalarize array because a generic nondeterministic initializer has an "
              "unsupported external use"
          );
          diag.attachNote(owner->getLoc()) << "source witness is also used here";
          diag.attachNote(targetLoc)
              << "array index " << idx << " requires refined scalar member type " << targetType;
          return diag;
        }
      }
    }
    return success();
  };
  auto noteMaterializedUse = [&](ArrayAttr idx, Type targetType, Location loc,
                                 StringRef description,
                                 StringRef indexDescription) -> LogicalResult {
    auto existing = firstMaterializedUseByIndex.find(idx);
    if (existing == firstMaterializedUseByIndex.end()) {
      firstMaterializedUseByIndex.try_emplace(
          idx, PendingMaterializationUse {targetType, loc, description, indexDescription}
      );
      return success();
    }
    Type existingType = existing->second.type;
    FailureOr<Type> commonType = getCommonRefinedType(existingType, targetType, tracker);
    if (failed(commonType)) {
      InFlightDiagnostic diag =
          createOp.emitError("cannot split heterogeneous array member because ")
          << arrayDescription << " reuses one SSA value for incompatible scalar member types";
      diag.attachNote(existing->second.loc)
          << existing->second.indexDescription << ' ' << idx << " is materialized for "
          << existing->second.description << ' ' << existingType;
      diag.attachNote(loc) << "same index is also materialized for " << description << ' '
                           << targetType;
      return diag;
    }
    existing->second.type = *commonType;
    return success();
  };
  // The rewrite replaces every read in one array-value lifetime with the same materialized
  // scalar value. Check the final type against both the read result and its typed consumers before
  // any candidate or expandable array rewrite can erase an earlier read.
  auto verifyMaterializedReadUsers = [&](ArrayAttr idx) -> LogicalResult {
    auto materialization = firstMaterializedUseByIndex.find(idx);
    if (materialization == firstMaterializedUseByIndex.end()) {
      return success();
    }
    for (ReadArrayOp readOp : readsByIndex.lookup(idx)) {
      plan.materializationTypeByRead[readOp] = materialization->second.type;
      if (!canReplaceReadResultWithType(
              readOp, materialization->second.type, tracker, "planExpandableLocalArray"
          ) ||
          failed(canReplaceReadUsersWithType(readOp, materialization->second.type, tables))) {
        return failure();
      }
    }
    readsByIndex.erase(idx);
    firstMaterializedUseByIndex.erase(idx);
    return success();
  };
  // Reads are replaced with their stored SSA values during rewriting. Preserve that identity
  // through subsequent local writes so preflight sees external observations of the actual source.
  // Capture the value at the read, since a later write can overwrite its original array index.
  DenseMap<Value, OpOperand *> sourceByRead;
  for (Operation *user : users) {
    if (auto writeOp = llvm::dyn_cast<WriteArrayOp>(user)) {
      ArrayAttr idx = getIndexAsAttr(writeOp);
      if (failed(verifyMaterializedReadUsers(idx))) {
        return failure();
      }
      OpOperand *source = sourceByRead.lookup(writeOp.getRvalue());
      if (!source) {
        source = &writeOp.getRvalueMutable();
      }
      sourceByIndex[idx] = source;
      continue;
    }
    if (auto readOp = llvm::dyn_cast<ReadArrayOp>(user)) {
      ArrayAttr idx = getIndexAsAttr(readOp);
      Type readType = readOp.getResult().getType();
      if (OpOperand *source = sourceByIndex.lookup(idx)) {
        Value scalarValue = source->get();
        sourceByRead[readOp.getResult()] = source;
        plan.witnessRequests.emplace_back(source, readType);
        if (failed(getCommonRefinedType(scalarValue.getType(), readType, tracker))) {
          InFlightDiagnostic diag =
              createOp.emitError("cannot split heterogeneous array member because ")
              << arrayDescription << " stores a scalar value with an incompatible read result type";
          diag.attachNote(scalarValue.getLoc())
              << "array index " << idx << " stores value type " << scalarValue.getType();
          diag.attachNote(readOp.getLoc()) << "same index is read as " << readType;
          return diag;
        }
        auto res =
            noteMaterializedUse(idx, readType, readOp.getLoc(), "read result type", "array index");
        if (failed(res)) {
          return failure();
        }
      } else {
        auto res = noteMaterializedUse(
            idx, readType, readOp.getLoc(), "read result type", "unwritten index"
        );
        if (failed(res)) {
          return failure();
        }
      }
      readsByIndex[idx].push_back(readOp);
      continue;
    }

    auto memberWriteOp = llvm::dyn_cast<MemberWriteOp>(user);
    if (!memberWriteOp) {
      continue;
    }
    auto memberDef = memberWriteOp.getMemberDefOp(tables);
    if (failed(memberDef)) {
      return failure();
    }
    auto splitIt = splitTypesByMember.find(memberDef->get());
    if (splitIt == splitTypesByMember.end()) {
      continue;
    }

    for (const auto &[idx, targetType] : splitIt->second) {
      OpOperand *source = sourceByIndex.lookup(idx);
      if (!source) {
        auto res = noteMaterializedUse(
            idx, targetType, memberWriteOp.getLoc(), "scalar member type", "unwritten index"
        );
        if (failed(res)) {
          return failure();
        }
        continue;
      }
      Value scalarValue = source->get();
      plan.witnessRequests.emplace_back(source, targetType);
      auto existing = firstUseByValue.find(scalarValue);
      if (!valueCompatibleWithTargetType(scalarValue, targetType)) {
        InFlightDiagnostic diag =
            createOp.emitError("cannot split heterogeneous array member because ")
            << arrayDescription << " stores a scalar value with an incompatible split-member type";
        diag.attachNote(scalarValue.getLoc())
            << "array index " << idx << " stores value type " << scalarValue.getType();
        diag.attachNote(memberWriteOp.getLoc())
            << "same index is written with scalar member type " << targetType;
        return diag;
      }
      auto verifyWritesRes =
          verifyExternalMemberWriteUses(scalarValue, idx, targetType, memberWriteOp.getLoc());
      if (failed(verifyWritesRes)) {
        return failure();
      }
      auto noteMaterializedUseRes = noteMaterializedUse(
          idx, targetType, memberWriteOp.getLoc(), "scalar member type", "array index"
      );
      if (failed(noteMaterializedUseRes)) {
        return failure();
      }
      if (existing == firstUseByValue.end()) {
        bool canMaterializeTarget;
        if (UnifiableCastOp castOp = scalarValue.getDefiningOp<UnifiableCastOp>()) {
          canMaterializeTarget = canUseScalarizedValueAsType(
              castOp.getInput().getType(), targetType, tracker, "planExpandableLocalArray"
          );
        } else {
          // Array reads are replaced before member writes, so their replacement can carry the
          // refined type. Other ordinary SSA values would be emitted unchanged.
          canMaterializeTarget =
              scalarValue.getDefiningOp<NonDetOp>() || scalarValue.getDefiningOp<ReadArrayOp>();
        }
        if (targetType != scalarValue.getType() && !canMaterializeTarget) {
          InFlightDiagnostic diag =
              createOp.emitError("cannot split heterogeneous array member because ")
              << arrayDescription
              << " uses a non-nondeterministic SSA value for a refined scalar member type";
          diag.attachNote(scalarValue.getLoc())
              << "array index " << idx << " stores value type " << scalarValue.getType();
          diag.attachNote(memberWriteOp.getLoc())
              << "same index is written with scalar member type " << targetType;
          return diag;
        }
        firstUseByValue.try_emplace(
            scalarValue, std::make_pair(targetType, memberWriteOp.getLoc())
        );
        continue;
      }
      Type existingType = existing->second.first;
      FailureOr<Type> commonType = getCommonRefinedType(existingType, targetType, tracker);
      if (failed(commonType)) {
        InFlightDiagnostic diag =
            createOp.emitError("cannot split heterogeneous array member because ")
            << arrayDescription << " reuses one SSA value for incompatible scalar member types";
        diag.attachNote(existing->second.second)
            << "value is used for scalar member type " << existingType;
        diag.attachNote(memberWriteOp.getLoc())
            << "same value is also used for scalar member type " << targetType;
        return diag;
      }
      bool canMaterializeCommonType;
      if (UnifiableCastOp castOp = scalarValue.getDefiningOp<UnifiableCastOp>()) {
        canMaterializeCommonType = canUseScalarizedValueAsType(
            castOp.getInput().getType(), *commonType, tracker, "planExpandableLocalArray"
        );
      } else {
        canMaterializeCommonType = scalarValue.getDefiningOp<NonDetOp>();
      }
      if (*commonType != scalarValue.getType() && !canMaterializeCommonType) {
        InFlightDiagnostic diag =
            createOp.emitError("cannot split heterogeneous array member because ")
            << arrayDescription
            << " reuses a non-nondeterministic SSA value for distinct scalar member types";
        diag.attachNote(existing->second.second)
            << "value is used for scalar member type " << existingType;
        diag.attachNote(memberWriteOp.getLoc())
            << "same value is also used for scalar member type " << targetType;
        return diag;
      }
      existing->second.first = *commonType;
    }
  }
  for (ArrayAttr idx : indices) {
    if (failed(verifyMaterializedReadUsers(idx))) {
      return failure();
    }
  }
  return success();
}

/// Rewrite one expandable local array allocation in block order.
static LogicalResult rewriteExpandableLocalArray(
    CreateArrayOp createOp, const DenseMap<MemberDefOp, SplitMemberInfo> &splitMembers,
    SymbolTableCollection &tables, PatternRewriter &rewriter, const ConversionTracker &tracker,
    const ExpandableArrayPlan &plan
) {
  ArrayType arrTy = createOp.getType();
  std::optional<SmallVector<ArrayAttr>> maybeIndices = arrTy.getSubelementIndices();
  if (!maybeIndices) {
    return failure();
  }
  ArrayRef<ArrayAttr> indices = *maybeIndices;

  DenseMap<ArrayAttr, Value> valueByIndex;
  DenseMap<ArrayAttr, DictionaryAttr> writeDiscardableAttrsByIndex;
  if (failed(seedValuesFromArrayElements(createOp, indices, valueByIndex))) {
    return failure();
  }

  // Earlier candidate rewrites may have replaced an input read with its scalar witness. Resolve
  // the planned source operands now, without replaying array writes or materialization lifetimes.
  DenseMap<Value, Type> specializedTypeByNondet;
  auto noteNondetSpecialization = [&](Value value, Type type) -> LogicalResult {
    if (!value || value.getType() == type || !value.getDefiningOp<NonDetOp>()) {
      return success();
    }
    auto existing = specializedTypeByNondet.find(value);
    if (existing == specializedTypeByNondet.end()) {
      specializedTypeByNondet.try_emplace(value, type);
      return success();
    }
    FailureOr<Type> commonType = getCommonRefinedType(existing->second, type, tracker);
    if (failed(commonType)) {
      return createOp.emitError(
          "cannot scalarize array because a shared nondeterministic initializer requires "
          "incompatible specialized types"
      );
    }
    existing->second = *commonType;
    return success();
  };
  for (const auto &[source, type] : plan.witnessRequests) {
    if (failed(noteNondetSpecialization(source->get(), type))) {
      return failure();
    }
  }

  DenseMap<Value, SmallVector<CallOp>> externalCallsBySource;
  for (const auto &[source, type] : specializedTypeByNondet) {
    auto calls = collectSpecializedWitnessConstraints(source, type, tables, [&](OpOperand &use) {
      Operation *owner = use.getOwner();
      if (owner == createOp.getOperation()) {
        return true;
      }
      auto writeOp = dyn_cast<WriteArrayOp>(owner);
      return writeOp && writeOp.getRvalue() == source &&
             writeOp.getArrRef() == createOp.getResult();
    }, createOp);
    if (failed(calls)) {
      return failure();
    }
    externalCallsBySource[source] = std::move(*calls);
  }
  DenseMap<Value, Value> specializedNondetBySource;
  materializeSharedNondetSpecializations(
      specializedTypeByNondet, externalCallsBySource, rewriter, specializedNondetBySource
  );

  // Earlier candidate rewrites (and the witness specializations above) may retarget a call
  // consuming this array's read. Recheck those live consumers against the already-planned type;
  // the array's write/read history and common materialization types do not need recomputing.
  for (const auto &[read, finalType] : plan.materializationTypeByRead) {
    if (failed(canReplaceReadUsersWithType(llvm::cast<ReadArrayOp>(read), finalType, tables))) {
      return failure();
    }
  }
  DenseSet<Value> generatedMaterializations;
  SmallVector<WriteArrayOp> writesToErase;
  auto getScalarizedValue = [&](ArrayAttr idx, Type type, Location loc) -> FailureOr<Value> {
    if (Value scalarValue = valueByIndex.lookup(idx)) {
      if (Value specializedNondet = specializedNondetBySource.lookup(scalarValue)) {
        return specializedNondet;
      }
    }
    return getOrCreateScalarizedLocalArrayValue(
        valueByIndex, generatedMaterializations, idx, type, loc, rewriter, tracker,
        plan.materializationPoint
    );
  };
  for (Operation *user : plan.users) {
    if (auto writeOp = llvm::dyn_cast<WriteArrayOp>(user)) {
      ArrayAttr idx = getIndexAsAttr(writeOp);
      valueByIndex[idx] = writeOp.getRvalue();
      writeDiscardableAttrsByIndex[idx] = writeOp->getDiscardableAttrDictionary();
      writesToErase.push_back(writeOp);
      continue;
    }

    if (auto readOp = llvm::dyn_cast<ReadArrayOp>(user)) {
      ArrayAttr idx = getIndexAsAttr(readOp);
      Type requestedType = readOp.getResult().getType();
      if (Type materializationType = plan.materializationTypeByRead.lookup(readOp.getOperation())) {
        requestedType = materializationType;
      }
      rewriter.setInsertionPoint(readOp);
      FailureOr<Value> scalarValue = getScalarizedValue(idx, requestedType, readOp.getLoc());
      if (failed(scalarValue)) {
        return failure();
      }
      Value replacementWithMetadata =
          materializeReadMetadataCarrier(readOp, *scalarValue, rewriter);
      replaceAllUsesIgnoringType(readOp.getResult(), replacementWithMetadata);
      rewriter.eraseOp(readOp);
      continue;
    }

    auto memberWriteOp = llvm::dyn_cast<MemberWriteOp>(user);
    if (!memberWriteOp) {
      return failure();
    }

    auto memberDef = memberWriteOp.getMemberDefOp(tables);
    if (failed(memberDef)) {
      return failure();
    }
    auto splitIt = splitMembers.find(memberDef->get());
    if (splitIt == splitMembers.end()) {
      return failure();
    }
    const SplitMemberInfo &splitInfo = splitIt->second;

    auto getScalarValue = [&](ArrayAttr idx, Type type) {
      return getScalarizedValue(idx, type, memberWriteOp.getLoc());
    };
    auto emitRes = emitScalarMemberWrites(
        memberWriteOp, splitInfo, writeDiscardableAttrsByIndex, rewriter, getScalarValue
    );
    if (failed(emitRes)) {
      return failure();
    }
    rewriter.eraseOp(memberWriteOp);
  }

  for (WriteArrayOp writeOp : llvm::make_early_inc_range(writesToErase)) {
    rewriter.eraseOp(writeOp);
  }
  if (createOp.getResult().use_empty()) {
    rewriter.eraseOp(createOp);
  }
  // All local read/write snapshots have been consumed; discard replaced witnesses only now.
  for (const auto &[source, _] : specializedTypeByNondet) {
    if (source.use_empty()) {
      rewriter.eraseOp(source.getDefiningOp<NonDetOp>());
    }
  }
  return success();
}

/// Rewrite remaining expandable whole-array writes to split members.
static LogicalResult rewriteExpandableMemberWrites(
    DenseMap<MemberDefOp, SplitMemberInfo> &splitMembers, SymbolTableCollection &tables,
    PatternRewriter &rewriter, const ConversionTracker &tracker,
    const DenseMap<CreateArrayOp, ExpandableArrayPlan> &plans
) {
  DenseSet<MemberDefOp> splitMemberSet;
  DenseMap<MemberDefOp, SmallVector<ArrayAttr>> splitIndicesByMember;
  for (const auto &entry : splitMembers) {
    splitMemberSet.insert(entry.first);
    splitIndicesByMember[entry.first] = entry.second.indices;
  }

  for (const auto &entry : splitMembers) {
    MemberDefOp member = entry.first;
    StructDefOp parentStruct = getParentOfType<StructDefOp>(member);
    assert(parentStruct && "MemberDefOp parent is always StructDefOp");

    auto uses = llzk::getSymbolUses(member, parentStruct);
    if (!uses) {
      return failure();
    }

    SmallVector<CreateArrayOp> createsToExpand;
    DenseSet<CreateArrayOp> seenCreates;
    for (SymbolTable::SymbolUse symUse : uses.value()) {
      auto memberWriteOp = llvm::dyn_cast<MemberWriteOp>(symUse.getUser());
      if (!memberWriteOp) {
        continue;
      }
      auto memberDef = memberWriteOp.getMemberDefOp(tables);
      FailureOr<CreateArrayOp> maybeCreateOp = getStaticLocalArrayCreate(memberWriteOp);
      if (succeeded(memberDef) && memberDef->get() == member && succeeded(maybeCreateOp) &&
          !seenCreates.contains(*maybeCreateOp)) {
        createsToExpand.push_back(*maybeCreateOp);
        seenCreates.insert(*maybeCreateOp);
      }
    }

    for (CreateArrayOp createOp : createsToExpand) {
      DominanceInfo domInfo;
      if (failed(verifyExpandableLocalArrayUsers(createOp, splitMemberSet, tables, domInfo))) {
        return failure();
      }
      auto res = verifyExpandableLocalArraySplitIndices(
          createOp, splitIndicesByMember, /*splitIndexOpsByMember=*/nullptr, tables
      );
      if (failed(res)) {
        return failure();
      }
      auto plan = plans.find(createOp);
      assert(plan != plans.end() && "expandable array must have passed preflight");
      if (failed(rewriteExpandableLocalArray(
              createOp, splitMembers, tables, rewriter, tracker, plan->second
          ))) {
        return failure();
      }
    }
  }
  return success();
}

/// Verify that all writes to any split member are either candidate or otherwise expandable.
///
/// Splitting a member is global: after replacement, every read of the original array-typed member
/// is redirected to the split scalar members. That is only sound when every write to that member
/// can also be expanded.
static LogicalResult verifySplitMemberWritesExpandable(
    ArrayRef<ScalarizedArrayInfo> arraysToScalarize, SymbolTableCollection &tables,
    const ConversionTracker &tracker, DenseMap<CreateArrayOp, ExpandableArrayPlan> &plans
) {
  DenseMap<MemberDefOp, DenseSet<Operation *>> candidateWritesByMember;
  DenseMap<MemberDefOp, SmallVector<ArrayAttr>> splitIndicesByMember;
  DenseMap<MemberDefOp, Operation *> splitIndexOpsByMember;
  DenseMap<MemberDefOp, DenseMap<ArrayAttr, Type>> splitTypesByMember;
  DenseMap<MemberDefOp, DenseMap<ArrayAttr, Operation *>> splitTypeOpsByMember;
  for (const ScalarizedArrayInfo &info : arraysToScalarize) {
    for (MemberWriteOp memberWriteOp : info.memberWrites) {
      auto memberDef = memberWriteOp.getMemberDefOp(tables);
      if (succeeded(memberDef)) {
        MemberDefOp member = memberDef->get();
        candidateWritesByMember[member].insert(memberWriteOp.getOperation());
        auto [indicesIt, inserted] = splitIndicesByMember.try_emplace(member, info.indices);
        if (inserted) {
          splitIndexOpsByMember[member] = memberWriteOp.getOperation();
        } else if (!haveSameIndexSet(indicesIt->second, info.indices)) {
          InFlightDiagnostic diag = member.emitError(
              "cannot split heterogeneous array member because candidate whole-array writes use "
              "different index set"
          );
          diag.attachNote(splitIndexOpsByMember.lookup(member)->getLoc())
              << "candidate establishes " << indicesIt->second.size() << " split member indices";
          Diagnostic &note = diag.attachNote(memberWriteOp.getLoc())
                             << "candidate whole-array write has " << info.indices.size()
                             << " array indices";
          if (ArrayAttr extraIndex = findIndexMissingFrom(info.indices, indicesIt->second)) {
            note << ", including extra index " << extraIndex;
          } else if (
              ArrayAttr missingIndex = findIndexMissingFrom(indicesIt->second, info.indices)
          ) {
            note << ", missing index " << missingIndex;
          }
          return diag;
        }
        DenseMap<ArrayAttr, Type> &splitTypes = splitTypesByMember[member];
        DenseMap<ArrayAttr, Operation *> &splitTypeOps = splitTypeOpsByMember[member];
        for (ArrayAttr idx : info.indices) {
          Type candidateType = info.typeByIndex.lookup(idx);
          auto res = mergeSplitCandidateType(
              member, idx, candidateType, memberWriteOp.getOperation(), splitTypes, &splitTypeOps,
              tracker
          );
          if (failed(res)) {
            return failure();
          }
        }
      }
    }
  }
  DenseSet<MemberDefOp> splitMemberSet;
  for (const auto &entry : candidateWritesByMember) {
    splitMemberSet.insert(entry.first);
  }
  DominanceInfo domInfo;

  for (const ScalarizedArrayInfo &info : arraysToScalarize) {
    ExpandableArrayPlan candidatePlan;
    auto verifyNoIncompatibleShares = planExpandableLocalArray(
        info.createOp, splitTypesByMember, tables, tracker, candidatePlan,
        "a scalarization candidate"
    );
    if (failed(verifyNoIncompatibleShares)) {
      return failure();
    }
  }

  for (const auto &entry : candidateWritesByMember) {
    MemberDefOp member = entry.first;
    const DenseSet<Operation *> &candidateWrites = entry.second;
    StructDefOp parentStruct = getParentOfType<StructDefOp>(member);
    assert(parentStruct && "MemberDefOp parent is always StructDefOp");

    auto uses = llzk::getSymbolUses(member, parentStruct);
    if (!uses) {
      return member.emitError(
          "cannot split heterogeneous array member because its symbol uses could not be inspected"
      );
    }

    for (SymbolTable::SymbolUse symUse : uses.value()) {
      auto writeOp = llvm::dyn_cast<MemberWriteOp>(symUse.getUser());
      if (!writeOp || candidateWrites.contains(writeOp.getOperation())) {
        continue;
      }
      FailureOr<CreateArrayOp> maybeCreateOp = getStaticLocalArrayCreate(writeOp);
      if (failed(maybeCreateOp) || failed(verifyExpandableLocalArrayUsers(
                                       *maybeCreateOp, splitMemberSet, tables, domInfo
                                   ))) {
        InFlightDiagnostic diag = member.emitError(
            "cannot split heterogeneous array member because not every write to it can be "
            "scalarized"
        );
        diag.attachNote(writeOp.getLoc()) << "whole-array write is not backed by a scalarization "
                                             "candidate";
        return diag;
      }
      auto verifyExpandable = verifyExpandableLocalArraySplitIndices(
          *maybeCreateOp, splitIndicesByMember, &splitIndexOpsByMember, tables
      );
      if (failed(verifyExpandable)) {
        return failure();
      }
      if (plans.contains(*maybeCreateOp)) {
        continue;
      }
      auto verifyNoIncompatibleShares = planExpandableLocalArray(
          *maybeCreateOp, splitTypesByMember, tables, tracker, plans[*maybeCreateOp]
      );
      if (failed(verifyNoIncompatibleShares)) {
        return failure();
      }
    }
  }
  return success();
}

/// Erase original array-typed members after all symbol uses have been redirected.
static void eraseUnusedOriginalMembers(
    DenseMap<MemberDefOp, SplitMemberInfo> &splitMembers, PatternRewriter &rewriter
) {
  for (const auto &[member, _] : splitMembers) {
    StructDefOp parentStruct = getParentOfType<StructDefOp>(member);
    assert(parentStruct && "MemberDefOp parent is always StructDefOp");
    auto uses = llzk::getSymbolUses(member, parentStruct);
    if (uses && uses->empty()) {
      rewriter.eraseOp(member);
    }
  }
}

} // namespace

/// Scalarize all safe pseudo-homogeneous arrays exposed in the current flattening iteration.
///
/// Running this before general type propagation prevents the propagation step from choosing one
/// concrete element type for an array that semantically contains a different concrete type at each
/// static index.
LogicalResult scalarizeHeterogeneousArrays(ModuleOp modOp, ConversionTracker &tracker) {
  SmallVector<ScalarizedArrayInfo, 2> arraysToScalarize;
  modOp.walk([&arraysToScalarize, &tracker](CreateArrayOp op) {
    FailureOr<ScalarizedArrayInfo> info = getScalarizedArrayInfo(op, tracker);
    if (succeeded(info)) {
      arraysToScalarize.push_back(*info);
    }
  });
  if (arraysToScalarize.empty()) {
    return success();
  }

  PatternRewriter rewriter(modOp.getContext());
  SymbolTableCollection tables;
  DenseMap<MemberDefOp, SmallVector<ArrayAttr>> splitIndicesByMember;
  DenseMap<MemberDefOp, DenseMap<ArrayAttr, Type>> splitTypesByMember;
  DenseMap<Operation *, ScalarizedArrayInfo *> candidateInfoByCreateOp;
  auto refreshRes =
      refreshValuesFromDependentCandidates(arraysToScalarize, candidateInfoByCreateOp, tracker);
  if (failed(refreshRes)) {
    return failure();
  }
  auto verifyRes =
      verifyRefreshedCandidateReads(arraysToScalarize, tables, tracker, candidateInfoByCreateOp);
  if (failed(verifyRes)) {
    return failure();
  }
  DenseMap<CreateArrayOp, ExpandableArrayPlan> expandablePlans;
  if (failed(
          verifySplitMemberWritesExpandable(arraysToScalarize, tables, tracker, expandablePlans)
      )) {
    return failure();
  }
  auto collectRes = collectSplitTypesByMember(
      arraysToScalarize, tables, splitIndicesByMember, splitTypesByMember, tracker
  );
  if (failed(collectRes) ||
      failed(verifySplitMemberReadsRewritable(modOp, splitTypesByMember, tables, tracker))) {
    return failure();
  }
  DenseMap<Value, Type> specializedTypeByNondet;
  if (failed(collectSharedNondetSpecializationTypes(
          arraysToScalarize, tracker, splitTypesByMember, tables, specializedTypeByNondet
      ))) {
    return failure();
  }
  auto externalCalls =
      collectSharedNondetExternalConstraints(arraysToScalarize, specializedTypeByNondet, tables);
  if (failed(externalCalls)) {
    return failure();
  }

  DenseMap<Value, Value> specializedNondetBySource;
  materializeSharedNondetSpecializations(
      specializedTypeByNondet, *externalCalls, rewriter, specializedNondetBySource
  );

  DenseSet<Value> sourcesUsedByReads;
  for (const ScalarizedArrayInfo &info : arraysToScalarize) {
    for (ReadArrayOp readOp : info.reads) {
      sourcesUsedByReads.insert(info.valueByIndex.lookup(getIndexAsAttr(readOp)));
    }
  }
  DenseMap<MemberDefOp, SplitMemberInfo> splitMembers;
  for (ScalarizedArrayInfo &info : arraysToScalarize) {
    auto rewriteRes = rewriteLocalArray(
        info, splitMembers, splitTypesByMember, splitIndicesByMember, tables, rewriter, tracker,
        &candidateInfoByCreateOp, specializedTypeByNondet, specializedNondetBySource
    );
    if (failed(rewriteRes)) {
      return failure();
    }
  }
  // Shared maps and candidate caches retain source Value handles throughout the rewrite loop.
  // Erase unused original witnesses only after no remaining candidate can inspect those handles.
  for (const auto &[source, _] : specializedTypeByNondet) {
    if (!sourcesUsedByReads.contains(source) && source.use_empty()) {
      rewriter.eraseOp(source.getDefiningOp<NonDetOp>());
    }
  }
  if (failed(
          rewriteExpandableMemberWrites(splitMembers, tables, rewriter, tracker, expandablePlans)
      )) {
    return failure();
  }
  if (failed(rewriteSplitMemberReads(modOp, splitMembers, tables, rewriter, tracker))) {
    return failure();
  }
  eraseUnusedOriginalMembers(splitMembers, rewriter);
  tracker.updateModifiedFlag(true);
  return success();
}

} // namespace llzk::polymorphic::detail::flattening
