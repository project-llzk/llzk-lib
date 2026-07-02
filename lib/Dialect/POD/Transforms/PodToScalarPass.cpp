//===-- PodToScalarPass.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-pod-to-scalar` pass.
///
/// The steps of this transformation are as follows:
///
/// 0. Rewrite pod-typed `llzk.nondet` allocations into `pod.new` so later stages only need to
///    reason about POD storage through POD dialect operations.
///
/// 1. Run a dialect conversion that replaces pod-typed struct members with one scalar member per
///    POD record, replaces array-typed struct members whose element type is a POD with one parallel
///    array member per POD record, and remembers how each original member was split for the later
///    rewriting steps.
///
/// 2. Run a dialect conversion that splits arrays whose element type is a POD into parallel arrays
///    in `llzk.nondet`, `array.*`, `constrain.eq`, `constrain.in`, `struct.readm`, `struct.writem`,
///    `function.def`, `function.call`, `function.return`, and bool quantifiers.
///
/// 3. Run a dialect conversion that does the following:
///
///    - Replace `MemberReadOp` and `MemberWriteOp` targeting the pod-typed struct members split in
///      step 1 so they instead perform reads and writes on the new scalar members. Reads and writes
///      are tracked through virtual POD placeholders so the conversion can keep propagating scalar
///      leaves instead of re-introducing aggregate POD storage.
///
///    - Remove optional initialization from `NewPodOp` and instead insert a list of `WritePodOp`
///      immediately following.
///
///    - Split remaining direct POD values to scalars in `FuncDefOp`, `CallOp`, and `ReturnOp`.
///      When a rewritten op still needs POD contents locally, keep them in the same virtual
///      placeholder form for as long as possible and only materialize concrete `pod.write`
///      operations as a fallback for unresolved uses.
///
/// 4. Promote pod reads and writes out of `scf.if`, `scf.for`, and `scf.while` regions when the
///    access can be modeled as an SSA value flowing through the region boundary. This puts the
///    pod accesses that mem2reg must eliminate into a parent block or loop-carried value.
///
/// 5. Run MLIR "sroa" pass to split remaining POD allocations into single-record POD allocations
///    (to prepare for the "mem2reg" pass because its API cannot split memory by itself).
///
/// 6. Run MLIR "mem2reg" pass to convert all single-record POD allocations and accesses into SSA
///    values.
///
/// 7. Remove POD allocations that become unread after memory promotion, then remove SSA values
///    made dead by that cleanup.
///
/// Steps 5-7 are rerun while nested POD types are still being exposed, until a fixpoint.
///
/// Note: This transformation imposes a "last write wins" semantics on pod records. If
/// different/configurable semantics are added in the future, some additional transformation would
/// be necessary before/during this pass so that multiple writes to the same record can be handled
/// properly while they still exist.
///
/// Note: This transformation will introduce a `nondet` op when there exists a read from a pod
/// record that was not earlier written to.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Dialect.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Bool/IR/Dialect.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Include/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Dialect.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/POD/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Polymorphic/IR/Dialect.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/RAM/IR/Dialect.h"
#include "llzk/Dialect/String/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKConversionUtils.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Transforms/SpecializedMemoryPasses.h"
#include "llzk/Util/Concepts.h"
#include "llzk/Util/TypeHelper.h"
#include "llzk/Util/Walk.h"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>

#include <functional>
#include <optional>

// Include the generated base pass class definitions.
namespace llzk::pod {
#define GEN_PASS_DEF_PODTOSCALARPASS
#include "llzk/Dialect/POD/Transforms/TransformationPasses.h.inc"
} // namespace llzk::pod

using namespace mlir;
using namespace llzk;
using namespace llzk::array;
using namespace llzk::pod;
using namespace llzk::function;
using namespace llzk::component;
using namespace llzk::polymorphic;

#define DEBUG_TYPE "llzk-pod-to-scalar"

namespace {

/// Path of nested POD record names from the original member to a scalar leaf record.
struct RecordChain {
  SmallVector<StringAttr> nameList;

  RecordChain() = default;

  explicit RecordChain(ArrayRef<StringAttr> names) : nameList(names.begin(), names.end()) {}

  bool operator==(const RecordChain &other) const { return nameList == other.nameList; }
};

} // namespace

namespace llvm {
template <> struct DenseMapInfo<RecordChain> {
  static RecordChain getEmptyKey() {
    return RecordChain {{DenseMapInfo<StringAttr>::getEmptyKey()}};
  }

  static RecordChain getTombstoneKey() {
    return RecordChain {{DenseMapInfo<StringAttr>::getTombstoneKey()}};
  }

  static unsigned getHashValue(const RecordChain &chain) {
    return llvm::hash_combine_range(chain.nameList.begin(), chain.nameList.end());
  }

  static bool isEqual(const RecordChain &lhs, const RecordChain &rhs) { return lhs == rhs; }
};
} // namespace llvm

namespace {

/// Return whether the given read/write access targets the same POD record.
inline static bool isSamePodRecord(ReadPodOp readOp, Value podRef, StringAttr recordName) {
  return readOp.getPodRef() == podRef && readOp.getRecordNameAttr() == recordName;
}

/// Return whether the given read/write access targets the same POD record.
inline static bool isSamePodRecord(WritePodOp writeOp, Value podRef, StringAttr recordName) {
  return writeOp.getPodRef() == podRef && writeOp.getRecordNameAttr() == recordName;
}

/// Return whether `op` contains a nested write to `podRef.recordName`.
static bool hasNestedWriteToRecord(Operation &op, Value podRef, StringAttr recordName) {
  return walkContainsMatch<WritePodOp>(op, [&](WritePodOp writeOp) {
    return writeOp.getOperation() != &op && isSamePodRecord(writeOp, podRef, recordName);
  });
}

/// Return whether `op` contains a nested write to any record of `podRef`.
static bool hasNestedWriteToPod(Operation &op, Value podRef) {
  return walkContainsMatch<WritePodOp>(op, [&](WritePodOp writeOp) {
    return writeOp.getOperation() != &op && writeOp.getPodRef() == podRef;
  });
}

/// Return whether `op` contains any read from `podRef.recordName`.
static bool hasReadFromRecord(Operation &op, Value podRef, StringAttr recordName) {
  return walkContainsMatch<ReadPodOp>(op, [&podRef, &recordName](ReadPodOp readOp) {
    return isSamePodRecord(readOp, podRef, recordName);
  });
}

/// Return whether `op` or any nested operation uses `value` as an operand.
static bool hasValueUse(Operation &op, Value value) {
  return walkContainsMatch<Operation *>(op, [&value](Operation *nestedOp) {
    return llvm::is_contained(nestedOp->getOperands(), value);
  });
}

/// Return the nearest preceding same-record write that can be forwarded to `readOp`.
///
/// This fold is intentionally conservative: it only forwards through intervening operations that do
/// not use the POD value at all. That keeps the rewrite local and avoids reasoning about other
/// whole-POD uses or record accesses that may observe mutation ordering.
static WritePodOp findNearestForwardableWriteInBlock(ReadPodOp readOp) {
  Value podRef = readOp.getPodRef();
  StringAttr recordName = readOp.getRecordNameAttr();

  for (Operation *op = readOp->getPrevNode(); op; op = op->getPrevNode()) {
    if (!hasValueUse(*op, podRef)) {
      continue;
    }

    auto writeOp = dyn_cast<WritePodOp>(op);
    return writeOp && isSamePodRecord(writeOp, podRef, recordName) ? writeOp : nullptr;
  }
  return nullptr;
}

/// If the given PodType can be split into scalars (always true for PodType), return it.
inline static PodType splittablePod(PodType pt) { return pt; }

/// If the given Type is a PodType that can be split into scalars, return it, otherwise nullptr.
inline static PodType splittablePod(Type t) {
  if (PodType pt = dyn_cast<PodType>(t)) {
    return splittablePod(pt);
  } else {
    return nullptr;
  }
}

/// Return `true` iff the given range contains any top-level PodType that this pass can split into
/// scalars.
inline static bool containsSplittablePodType(ArrayRef<Type> types) {
  for (Type t : types) {
    if (splittablePod(t)) {
      return true;
    }
  }
  return false;
}

/// Return `true` iff the given range contains any top-level PodType that this pass can split into
/// scalars.
template <typename T> static bool containsSplittablePodType(ValueTypeRange<T> types) {
  for (Type t : types) {
    if (splittablePod(t)) {
      return true;
    }
  }
  return false;
}

/// If the input ArrayType has a POD element type, return the input, else nullptr.
inline static ArrayType splittablePodArray(ArrayType at) {
  return isa<PodType>(at.getElementType()) ? at : nullptr;
}

/// If the input Type is an ArrayType with a POD element type, return the input, else nullptr.
inline static ArrayType splittablePodArray(Type t) {
  if (ArrayType at = dyn_cast<ArrayType>(t)) {
    return splittablePodArray(at);
  }
  return nullptr;
}

/// Return the flattened leaf type addressed by `recordChain` within `type`.
static Type getFlattenedTypeAlongPath(Type type, ArrayRef<StringAttr> recordChain) {
  if (recordChain.empty()) {
    return type;
  }

  if (PodType podTy = dyn_cast<PodType>(type)) {
    Type nextType = podTy.getRecordMap().lookup(recordChain.front().getValue());
    assert(nextType && "record path must exist in the containing POD");
    return getFlattenedTypeAlongPath(nextType, recordChain.drop_front());
  }

  if (ArrayType arrTy = splittablePodArray(type)) {
    auto elemPodTy = llvm::cast<PodType>(arrTy.getElementType());
    Type nextType = elemPodTy.getRecordMap().lookup(recordChain.front().getValue());
    assert(nextType && "record path must exist in the POD array element type");
    return flattenArrayElementType(
        arrTy, getFlattenedTypeAlongPath(nextType, recordChain.drop_front())
    );
  }

  llvm_unreachable("record path cannot continue through a non-POD leaf");
}

/// Visit each non-POD leaf record in `podTy`, providing its record-name chain and leaf type.
template <typename Fn>
static void forEachPodLeaf(PodType podTy, SmallVectorImpl<StringAttr> &recordChain, Fn &&callback) {
  std::function<void(Type)> walk = [&](Type type) {
    if (PodType nestedPodTy = llvm::dyn_cast<PodType>(type)) {
      for (RecordAttr record : nestedPodTy.getRecords()) {
        recordChain.push_back(record.getName());
        walk(record.getType());
        recordChain.pop_back();
      }
    } else if (ArrayType arrTy = splittablePodArray(type)) {
      auto elemPodTy = llvm::cast<PodType>(arrTy.getElementType());
      for (RecordAttr record : elemPodTy.getRecords()) {
        recordChain.push_back(record.getName());
        walk(flattenArrayElementType(arrTy, record.getType()));
        recordChain.pop_back();
      }
    } else {
      callback(RecordChain(recordChain), type);
    }
  };

  walk(podTy);
}

/// If the given Type is a PodType that can be split into scalars, append `collect` with all of
/// the scalar types that result from splitting the PodType. Otherwise, just push the `Type`.
size_t splitPodTypeTo(Type t, SmallVector<Type> &collect) {
  if (PodType pt = splittablePod(t)) {
    SmallVector<StringAttr> recordChain;
    size_t originalSize = collect.size();
    forEachPodLeaf(pt, recordChain, [&collect](const RecordChain &, Type leafType) {
      collect.push_back(leafType);
    });
    return collect.size() - originalSize;
  } else {
    collect.push_back(t);
    return 1;
  }
}

/// For each Type in the given input collection, call `splitPodTypeTo(Type,...)`.
template <typename TypeCollection>
inline void splitPodTypeTo(
    TypeCollection types, SmallVector<Type> &collect, SmallVector<size_t> *originalIdxToSize
) {
  for (Type t : types) {
    size_t count = splitPodTypeTo(t, collect);
    if (originalIdxToSize) {
      originalIdxToSize->push_back(count);
    }
  }
}

/// Return a list such that each scalar Type is directly added to the list but for each splittable
/// PodType, the proper number of scalar element types are added instead.
template <typename TypeCollection>
inline SmallVector<Type>
splitPodType(TypeCollection types, SmallVector<size_t> *originalIdxToSize = nullptr) {
  SmallVector<Type> collect;
  splitPodTypeTo(types, collect, originalIdxToSize);
  return collect;
}

/// Return `true` iff any type in the range is an array whose element type is a POD.
inline static bool containsSplittablePodArrayType(ArrayRef<Type> types) {
  return llvm::any_of(types, [](Type t) { return splittablePodArray(t); });
}

/// Return `true` iff any type in the range is an array whose element type is a POD.
template <typename T> static bool containsSplittablePodArrayType(ValueTypeRange<T> types) {
  return llvm::any_of(types, [](Type t) { return splittablePodArray(t); });
}

/// If `t` is an array with POD element type, append one parallel array type for each POD leaf.
static size_t splitPodArrayTypeTo(
    Type t, SmallVectorImpl<Type> &collect, SmallVector<RecordChain> *splitIds = nullptr
) {
  if (ArrayType at = splittablePodArray(t)) {
    auto podTy = llvm::cast<PodType>(at.getElementType());
    SmallVector<StringAttr> recordChain;
    size_t originalSize = collect.size();
    forEachPodLeaf(podTy, recordChain, [&](RecordChain id, Type leafType) {
      collect.push_back(flattenArrayElementType(at, leafType));
      if (splitIds) {
        splitIds->push_back(std::move(id));
      }
    });
    return collect.size() - originalSize;
  }

  collect.push_back(t);
  return 1;
}

/// For each Type in the given input collection, call `splitPodArrayTypeTo(Type,...)`.
template <typename TypeCollection>
inline void splitPodArrayTypeTo(
    TypeCollection types, SmallVectorImpl<Type> &collect, SmallVector<size_t> *originalIdxToSize
) {
  for (Type t : types) {
    size_t count = splitPodArrayTypeTo(t, collect);
    if (originalIdxToSize) {
      originalIdxToSize->push_back(count);
    }
  }
}

/// Return a list such that each non-array POD type is kept as-is, while each array-of-POD type is
/// replaced by one parallel array type per non-POD leaf record in the element POD.
template <typename TypeCollection>
inline SmallVector<Type>
splitPodArrayType(TypeCollection types, SmallVector<size_t> *originalIdxToSize = nullptr) {
  SmallVector<Type> collect;
  splitPodArrayTypeTo(types, collect, originalIdxToSize);
  return collect;
}

/// Return the suffixes to append to a function arg/result name when splitting an array of PODs.
static SmallVector<std::string> getSplitPodArrayRecordNameSuffixes(Type type) {
  SmallVector<std::string> suffixes;
  if (ArrayType at = splittablePodArray(type)) {
    SmallVector<RecordChain> splitIds;
    SmallVector<Type> ignoredTypes;
    splitPodArrayTypeTo(at, ignoredTypes, &splitIds);
    suffixes.reserve(splitIds.size());
    for (const RecordChain &id : splitIds) {
      std::string suffix;
      llvm::raw_string_ostream os(suffix);
      for (StringAttr recordName : id.nameList) {
        os << '.' << recordName.getValue();
      }
      suffixes.push_back(std::move(suffix));
    }
  }
  return suffixes;
}

/// Insert a `poly.unifiable_cast` when a rewritten value must match a more specific type.
///
/// This is the common bridge between wildcard-backed storage values and the more precise types
/// expected by surrounding rewritten IR. The cast is only emitted when the source and target
/// types unify and differ syntactically.
static Value castValueToTypeIfNeeded(OpBuilder &bldr, Location loc, Value value, Type targetType) {
  if (value.getType() == targetType) {
    return value;
  }
  assert(typesUnify(value.getType(), targetType) && "expected compatible rewritten types");
  return bldr.create<UnifiableCastOp>(loc, targetType, value);
}

/// Create a `pod.read` for one record of `podRef`.
inline static ReadPodOp
genRead(OpBuilder &bldr, Location loc, Value podRef, StringAttr recordName) {
  Type resultType =
      llvm::cast<PodType>(podRef.getType()).getRecordMap().lookup(recordName.getValue());
  return bldr.create<ReadPodOp>(loc, resultType, podRef, recordName);
}

/// Create a `pod.write` for one record of `podRef`.
inline static WritePodOp
genWrite(OpBuilder &bldr, Location loc, Value podRef, StringAttr recordName, Value value) {
  Type recordType =
      llvm::cast<PodType>(podRef.getType()).getRecordMap().lookup(recordName.getValue());
  return bldr.create<WritePodOp>(
      loc, podRef, recordName, castValueToTypeIfNeeded(bldr, loc, value, recordType)
  );
}

/// Return the single converted value from a 1:N adaptor range.
inline static Value getSingleConvertedValue(ValueRange values) {
  assert(values.size() == 1 && "expected a 1:1 converted value range");
  return values.front();
}

/// Materialize a scalar array value that preserves the shape of `originalArrTy`.
///
/// This is used as a shape-only carrier for `array.len` when an array-of-POD splits to
/// zero parallel leaf arrays (for example, `!array.type<... x !pod.type<[]>>`).
static Value materializeArrayLengthCarrier(
    Value originalArrRef, ArrayType originalArrTy, Location loc, ConversionPatternRewriter &rewriter
) {
  ArrayType carrierTy = originalArrTy.cloneWith(IndexType::get(rewriter.getContext()));

  if (auto create = originalArrRef.getDefiningOp<CreateArrayOp>()) {
    if (create.getMapOperands().empty()) {
      return rewriter.create<CreateArrayOp>(loc, carrierTy);
    }

    SmallVector<ValueRange> mapOperands;
    mapOperands.reserve(create.getMapOperands().size());
    for (OperandRange mapOperandGroup : create.getMapOperands()) {
      mapOperands.push_back(mapOperandGroup);
    }
    return rewriter.create<CreateArrayOp>(
        loc, carrierTy, mapOperands, create.getNumDimsPerMapAttr()
    );
  }

  bool hasAffineDims = llvm::any_of(originalArrTy.getDimensionSizes(), [](Attribute dimSize) {
    return llvm::isa<AffineMapAttr>(dimSize);
  });
  if (!hasAffineDims) {
    return rewriter.create<CreateArrayOp>(loc, carrierTy);
  }

  return rewriter.create<NonDetOp>(loc, carrierTy);
}

/// Flatten a range of converted value ranges into a single list of values.
template <typename RangeOfRanges>
static SmallVector<Value> flattenConvertedValues(RangeOfRanges ranges) {
  SmallVector<Value> values;
  for (ValueRange range : ranges) {
    llvm::append_range(values, range);
  }
  return values;
}

/// Replace any AffineMap-backed array dimensions nested within `type` with wildcard `?` dims.
///
/// This preserves the overall array nesting while erasing only the affine-map dimensions that
/// cannot always be witnessed after flattening a POD leaf array into a split array value.
static Type replaceAffineMapArrayDimsWithWildcards(Type type) {
  auto arrTy = llvm::dyn_cast<ArrayType>(type);
  if (!arrTy) {
    return type;
  }

  Builder builder(arrTy.getContext());
  SmallVector<Attribute> dims;
  dims.reserve(arrTy.getDimensionSizes().size());
  for (Attribute dimSize : arrTy.getDimensionSizes()) {
    if (llvm::isa<AffineMapAttr>(dimSize)) {
      dims.push_back(builder.getIndexAttr(ShapedType::kDynamic));
    } else {
      dims.push_back(dimSize);
    }
  }

  return arrTy.cloneWith(replaceAffineMapArrayDimsWithWildcards(arrTy.getElementType()), dims);
}

/// Return the wildcard-backed storage split type for one flattened POD leaf.
///
/// The precise split type preserves the original affine maps in the flattened leaf array. The
/// storage split type uses the same outer shape but replaces hidden leaf-array affine dims with
/// `?` until a matching instantiation can be recovered from concrete leaf-array values.
static ArrayType getSplitPodArrayStorageType(ArrayType arrTy, ArrayRef<StringAttr> recordChain) {
  auto elemPodTy = llvm::cast<PodType>(arrTy.getElementType());
  Type leafType = getFlattenedTypeAlongPath(elemPodTy, recordChain);
  return flattenArrayElementType(arrTy, replaceAffineMapArrayDimsWithWildcards(leafType));
}

/// Create an array value that callers can fully initialize via explicit writes or inserts.
///
/// Use `llzk.nondet` as the base when affine-map dimensions are present because `array.new`
/// cannot carry both inline elements and affine-map instantiation operands.
inline static Value createWritableArrayValue(OpBuilder &bldr, Location loc, ArrayType arrTy) {
  if (hasAffineMapAttr(arrTy)) {
    return bldr.create<NonDetOp>(loc, arrTy);
  } else {
    return bldr.create<CreateArrayOp>(loc, arrTy);
  }
}

/// Store the affine-map operand groups needed to rebuild one concrete array instantiation.
///
/// The layout mirrors `array.new`: `mapOperandStorage` keeps each instantiation group separately,
/// and `numDimsPerMap` records how many values in each group are dimensional arguments.
struct ArrayInstantiationInfo {
  SmallVector<SmallVector<Value>> mapOperandStorage;
  SmallVector<int32_t> numDimsPerMap;
};

/// Return `true` iff two recovered array instantiations can be rebuilt identically.
static bool equivalentArrayInstantiationInfo(
    const ArrayInstantiationInfo &lhs, const ArrayInstantiationInfo &rhs
) {
  if (lhs.numDimsPerMap != rhs.numDimsPerMap ||
      lhs.mapOperandStorage.size() != rhs.mapOperandStorage.size()) {
    return false;
  }

  for (auto [lhsGroup, rhsGroup] : llvm::zip_equal(lhs.mapOperandStorage, rhs.mapOperandStorage)) {
    if (lhsGroup != rhsGroup) {
      return false;
    }
  }

  return true;
}

/// Try to recover affine-map instantiation operands from a concrete array-producing value.
///
/// This peels compatibility casts, follows simple `pod.read` to dominating `pod.write`
/// forwarding, and succeeds only when the value ultimately traces back to a concrete
/// `array.new` carrying the instantiation groups.
static std::optional<ArrayInstantiationInfo> tryGetArrayInstantiationInfo(Value value) {
  while (auto cast = value.getDefiningOp<UnifiableCastOp>()) {
    value = cast.getInput();
  }

  if (ReadPodOp read = value.getDefiningOp<ReadPodOp>()) {
    if (WritePodOp write = findNearestForwardableWriteInBlock(read)) {
      return tryGetArrayInstantiationInfo(write.getValue());
    }
    return std::nullopt;
  }

  auto create = value.getDefiningOp<CreateArrayOp>();
  if (!create) {
    return std::nullopt;
  }

  ArrayInstantiationInfo info;
  info.mapOperandStorage.reserve(create.getMapOperands().size());
  for (OperandRange group : create.getMapOperands()) {
    info.mapOperandStorage.emplace_back(group.begin(), group.end());
  }

  if (DenseI32ArrayAttr numDimsPerMap = create.getNumDimsPerMapAttr()) {
    llvm::append_range(info.numDimsPerMap, numDimsPerMap.asArrayRef());
  }

  return info;
}

/// Describe whether a set of leaf arrays shares one recoverable instantiation.
enum class CommonArrayInstantiationStatus : std::uint8_t {
  unavailable,
  inferred,
  conflict,
};

/// Recover a single shared affine-map instantiation from all of `values`, if one exists.
///
/// Returns `inferred` when every value resolves to the same concrete `array.new`
/// instantiation, `unavailable` when any value has no recoverable witness, and `conflict`
/// when the recovered instantiations disagree.
static CommonArrayInstantiationStatus
inferCommonArrayInstantiation(ArrayRef<Value> values, ArrayInstantiationInfo &result) {
  bool initialized = false;
  for (Value value : values) {
    std::optional<ArrayInstantiationInfo> info = tryGetArrayInstantiationInfo(value);
    if (!info) {
      return CommonArrayInstantiationStatus::unavailable;
    }

    if (!initialized) {
      result = std::move(*info);
      initialized = true;
      continue;
    }

    if (!equivalentArrayInstantiationInfo(result, *info)) {
      return CommonArrayInstantiationStatus::conflict;
    }
  }

  return initialized ? CommonArrayInstantiationStatus::inferred
                     : CommonArrayInstantiationStatus::unavailable;
}

/// Generate `arith.constant` indices for one static array element position.
static SmallVector<Value> genArrayIndexConstants(OpBuilder &bldr, Location loc, ArrayAttr index) {
  SmallVector<Value> indices;
  for (Attribute attr : index) {
    assert(llvm::isa<IntegerAttr>(attr) && "array index must be an integer attribute");
    indices.push_back(bldr.create<arith::ConstantOp>(loc, llvm::cast<IntegerAttr>(attr)));
  }
  return indices;
}

/// Return the type produced by selecting `numIndices` leading dimensions from `arrTy`.
static Type getArraySelectionType(ArrayType arrTy, size_t numIndices) {
  assert(numIndices <= arrTy.getDimensionSizes().size() && "cannot select past the array rank");
  if (numIndices == arrTy.getDimensionSizes().size()) {
    return arrTy.getElementType();
  }
  return ArrayType::get(arrTy.getElementType(), arrTy.getDimensionSizes().drop_front(numIndices));
}

/// Create an `array.read` or `array.extract` for one concrete element or subarray.
static Value genArrayRead(OpBuilder &bldr, Location loc, Value arrayRef, ArrayRef<Value> indices) {
  Type t = arrayRef.getType();
  assert(llvm::isa<ArrayType>(t) && "array access must target an array type");
  ArrayType arrTy = llvm::cast<ArrayType>(t);
  if (indices.size() == arrTy.getDimensionSizes().size()) {
    return bldr.create<ReadArrayOp>(loc, arrTy.getElementType(), arrayRef, indices);
  }
  return bldr.create<ExtractArrayOp>(
      loc, llvm::cast<ArrayType>(getArraySelectionType(arrTy, indices.size())), arrayRef, indices
  );
}

inline static Value genArrayRead(OpBuilder &bldr, Location loc, Value arrayRef, ArrayAttr index) {
  SmallVector<Value> indices = genArrayIndexConstants(bldr, loc, index);
  return genArrayRead(bldr, loc, arrayRef, indices);
}

/// Create an `array.write` or `array.insert` for one concrete element or subarray.
static void
genArrayWrite(OpBuilder &bldr, Location loc, Value arrayRef, ArrayRef<Value> indices, Value value) {
  Type t = arrayRef.getType();
  assert(llvm::isa<ArrayType>(t) && "array access must target an array type");
  ArrayType arrTy = llvm::cast<ArrayType>(t);
  value = castValueToTypeIfNeeded(bldr, loc, value, getArraySelectionType(arrTy, indices.size()));
  if (indices.size() == arrTy.getDimensionSizes().size()) {
    bldr.create<WriteArrayOp>(loc, arrayRef, indices, value);
    return;
  }
  assert(llvm::isa<ArrayType>(value.getType()) && "subarray insertion requires an array value");
  bldr.create<InsertArrayOp>(loc, arrayRef, indices, value);
}

inline static void
genArrayWrite(OpBuilder &bldr, Location loc, Value arrayRef, ArrayAttr index, Value value) {
  SmallVector<Value> indices = genArrayIndexConstants(bldr, loc, index);
  genArrayWrite(bldr, loc, arrayRef, indices, value);
}

/// Strip compatibility casts introduced while threading POD-derived array values through rewrites.
static Value peelUnifiableCasts(Value value) {
  while (auto cast = value.getDefiningOp<UnifiableCastOp>()) {
    value = cast.getInput();
  }
  return value;
}

/// Collect split leaf arrays that are already available for an aggregate array-of-POD value.
///
/// This peels compatibility casts and forwards through a dominating same-record `pod.write` so
/// nested POD scalarization can reuse the split-array representation already produced elsewhere in
/// the pass instead of re-materializing dynamic arrays element-by-element.
static bool tryCollectDirectSplitPodArrayLeafValues(
    Value arrayValue, ArrayType arrTy, ArrayRef<Type> splitTypes, SmallVectorImpl<Value> &leafArrays
) {
  arrayValue = peelUnifiableCasts(arrayValue);

  if (auto cast = arrayValue.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast->getNumResults() != 1 || cast.getResult(0).getType() != arrTy ||
        cast->getNumOperands() != splitTypes.size()) {
      return false;
    }

    leafArrays.reserve(splitTypes.size());
    for (auto [operand, splitType] : llvm::zip_equal(cast.getOperands(), splitTypes)) {
      if (operand.getType() != splitType) {
        return false;
      }
      leafArrays.push_back(operand);
    }
    return true;
  }

  if (ReadPodOp readOp = arrayValue.getDefiningOp<ReadPodOp>()) {
    if (WritePodOp writeOp = findNearestForwardableWriteInBlock(readOp)) {
      return tryCollectDirectSplitPodArrayLeafValues(
          writeOp.getValue(), arrTy, splitTypes, leafArrays
      );
    }
  }

  return false;
}

/// Return whether `op` is preceded in its block by a write to `podRef.recordName`.
static bool hasEarlierWriteToRecordInBlock(Operation *op, Value podRef, StringAttr recordName) {
  for (Operation &candidate : *op->getBlock()) {
    if (&candidate == op) {
      return false;
    }
    if (auto writeOp = dyn_cast<WritePodOp>(&candidate)) {
      if (isSamePodRecord(writeOp, podRef, recordName)) {
        return true;
      }
    } else if (hasNestedWriteToRecord(candidate, podRef, recordName)) {
      return true;
    }
  }
  return false;
}

/// Return whether the read is preceded by a write to the same pod record within its block.
static bool hasEarlierWriteInBlock(ReadPodOp readOp) {
  return hasEarlierWriteToRecordInBlock(
      readOp.getOperation(), readOp.getPodRef(), readOp.getRecordNameAttr()
  );
}

/// Return whether `op` is preceded in its block by any write to `podRef`.
static bool hasEarlierWriteToPodInBlock(Operation *op, Value podRef) {
  for (Operation &candidate : *op->getBlock()) {
    if (&candidate == op) {
      return false;
    }
    if (auto writeOp = dyn_cast<WritePodOp>(&candidate)) {
      if (writeOp.getPodRef() == podRef) {
        return true;
      }
    } else if (hasNestedWriteToPod(candidate, podRef)) {
      return true;
    }
  }
  return false;
}

/// Return `true` iff `readOp` names a fresh pod record that has not been initialized or written.
static bool isFreshUnwrittenPodRead(ReadPodOp readOp) {
  NewPodOp newPod = readOp.getPodRef().getDefiningOp<NewPodOp>();
  if (!newPod) {
    return false;
  }
  auto isReadOpRecordName = [&readOp](Attribute attr) {
    return attr == readOp.getRecordNameAttr();
  };
  return llvm::none_of(newPod.getInitializedRecords(), isReadOpRecordName) &&
         !hasEarlierWriteInBlock(readOp);
}

/// Return `true` iff `value` is an unwritten array-of-POD field read from a fresh `pod.new`.
static bool isFreshUnwrittenPodArrayRead(Value value) {
  value = peelUnifiableCasts(value);
  ReadPodOp readOp = value.getDefiningOp<ReadPodOp>();
  return readOp && splittablePodArray(readOp.getType()) && isFreshUnwrittenPodRead(readOp);
}

/// Read one flattened POD leaf, including leaves that live inside an array-of-POD record.
static Value
genReadAlongPath(OpBuilder &bldr, Location loc, Value value, ArrayRef<StringAttr> recordChain) {
  if (recordChain.empty()) {
    return value;
  }

  Type valueType = value.getType();
  if (llvm::isa<PodType>(valueType)) {
    Value nextValue = genRead(bldr, loc, value, recordChain.front());
    return genReadAlongPath(bldr, loc, nextValue, recordChain.drop_front());
  }

  if (ArrayType arrTy = splittablePodArray(valueType)) {
    auto splitArrTy = llvm::cast<ArrayType>(getFlattenedTypeAlongPath(valueType, recordChain));

    if (isFreshUnwrittenPodArrayRead(value)) {
      return createWritableArrayValue(bldr, loc, splitArrTy);
    }

    if (!arrTy.hasStaticShape()) {
      SmallVector<RecordChain> splitIds;
      SmallVector<Type> splitTypes;
      splitPodArrayTypeTo(arrTy, splitTypes, &splitIds);

      SmallVector<Value> leafArrays;
      if (tryCollectDirectSplitPodArrayLeafValues(value, arrTy, splitTypes, leafArrays)) {
        auto *it = llvm::find(splitIds, RecordChain(recordChain));
        assert(it != splitIds.end() && "record path must name a flattened POD array leaf");
        return leafArrays[std::distance(splitIds.begin(), it)];
      }

      Value strippedValue = peelUnifiableCasts(value);
      if (strippedValue.getDefiningOp<ReadPodOp>()) {
        auto splitLeafReads =
            bldr.create<UnrealizedConversionCastOp>(loc, TypeRange(splitTypes), strippedValue);
        auto *it = llvm::find(splitIds, RecordChain(recordChain));
        assert(it != splitIds.end() && "record path must name a flattened POD array leaf");
        return splitLeafReads.getResult(std::distance(splitIds.begin(), it));
      }

      llvm_unreachable(
          "non-static nested array-of-POD scalarization requires split-array backing or an "
          "uninitialized pod field"
      );
    }

    auto subIndices = arrTy.getSubelementIndices();
    assert(subIndices && "static-shape arrays must provide subelement indices");

    Value splitArray = bldr.create<CreateArrayOp>(loc, splitArrTy);
    for (ArrayAttr index : *subIndices) {
      Value element = genArrayRead(bldr, loc, value, index);
      Value leafValue = genReadAlongPath(bldr, loc, element, recordChain);
      genArrayWrite(bldr, loc, splitArray, index, leafValue);
    }
    return splitArray;
  }

  llvm_unreachable("record path cannot continue through a non-POD leaf");
}

/// Read a flattened POD leaf by following each record name in `recordChain`.
inline static Value
genReadAlongPath(OpBuilder &bldr, Location loc, Value podRef, const RecordChain &recordChain) {
  return genReadAlongPath(bldr, loc, podRef, ArrayRef(recordChain.nameList));
}

/// Reconstruct a POD record from the leaf values collected while splitting nested accesses.
static Value rebuildFlattenedPodRecord(
    OpBuilder &bldr, Location loc, Type recordType, SmallVectorImpl<StringAttr> &recordChain,
    const DenseMap<RecordChain, Value> &leafValues
) {
  if (PodType nestedPodTy = dyn_cast<PodType>(recordType)) {
    NewPodOp nestedPod = bldr.create<NewPodOp>(loc, nestedPodTy);
    for (RecordAttr record : nestedPodTy.getRecords()) {
      recordChain.push_back(record.getName());
      Value recordValue =
          rebuildFlattenedPodRecord(bldr, loc, record.getType(), recordChain, leafValues);
      genWrite(bldr, loc, nestedPod, record.getName(), recordValue);
      recordChain.pop_back();
    }
    return nestedPod;
  }

  if (ArrayType arrTy = splittablePodArray(recordType)) {
    if (!arrTy.hasStaticShape()) {
      SmallVector<RecordChain> splitIds;
      SmallVector<Type> splitTypes;
      splitPodArrayTypeTo(arrTy, splitTypes, &splitIds);

      SmallVector<Value> leafArrays;
      leafArrays.reserve(splitIds.size());
      for (auto [id, splitType] : llvm::zip_equal(splitIds, splitTypes)) {
        SmallVector<StringAttr> fullChain(recordChain.begin(), recordChain.end());
        llvm::append_range(fullChain, id.nameList);
        auto it = leafValues.find(RecordChain(fullChain));
        assert(it != leafValues.end() && "missing flattened POD array leaf value");
        leafArrays.push_back(castValueToTypeIfNeeded(bldr, loc, it->second, splitType));
      }

      return bldr.create<UnrealizedConversionCastOp>(loc, TypeRange {arrTy}, leafArrays)
          .getResult(0);
    }

    auto elemPodTy = llvm::cast<PodType>(arrTy.getElementType());
    auto subIndices = arrTy.getSubelementIndices();
    assert(subIndices && "static-shape arrays must provide subelement indices");

    Value rebuiltArray = bldr.create<CreateArrayOp>(loc, arrTy);
    for (ArrayAttr index : *subIndices) {
      DenseMap<RecordChain, Value> elementLeafValues;
      SmallVector<StringAttr> elementRecordChain;
      forEachPodLeaf(elemPodTy, elementRecordChain, [&](RecordChain id, Type) {
        SmallVector<StringAttr> fullChain(recordChain.begin(), recordChain.end());
        llvm::append_range(fullChain, id.nameList);
        auto it = leafValues.find(RecordChain(fullChain));
        assert(it != leafValues.end() && "missing flattened POD array leaf value");
        elementLeafValues[id] = genArrayRead(bldr, loc, it->second, index);
      });

      NewPodOp elementPod = bldr.create<NewPodOp>(loc, elemPodTy);
      SmallVector<StringAttr> nestedChain;
      for (RecordAttr record : elemPodTy.getRecords()) {
        nestedChain.push_back(record.getName());
        Value recordValue =
            rebuildFlattenedPodRecord(bldr, loc, record.getType(), nestedChain, elementLeafValues);
        genWrite(bldr, loc, elementPod, record.getName(), recordValue);
        nestedChain.pop_back();
      }
      genArrayWrite(bldr, loc, rebuiltArray, index, elementPod);
    }
    return rebuiltArray;
  }

  auto it = leafValues.find(RecordChain(recordChain));
  assert(it != leafValues.end() && "missing flattened POD leaf value");
  return it->second;
}

using VirtualPodLeafMap = DenseMap<RecordChain, Value>;
using VirtualPodValueMap = DenseMap<Value, VirtualPodLeafMap>;
using DeferredPodArrayLeafMap = DenseMap<Value, SmallVector<Value>>;

/// Return the flattened leaf values for `podValue` when it is tracked as a virtual POD.
static const VirtualPodLeafMap *
lookupVirtualPodLeafMap(Value podValue, const VirtualPodValueMap &virtualPods) {
  auto it = virtualPods.find(podValue);
  return it != virtualPods.end() ? &it->second : nullptr;
}

/// Collect flattened POD leaf values in canonical traversal order.
static SmallVector<Value> orderedVirtualPodLeafValues(
    PodType podTy, Location loc, OpBuilder &bldr, const VirtualPodLeafMap &leafValues
) {
  SmallVector<Value> orderedValues;
  SmallVector<StringAttr> recordChain;
  forEachPodLeaf(
      podTy, recordChain,
      [&leafValues, &orderedValues, &bldr, loc](const RecordChain &id, Type leafType) {
    auto it = leafValues.find(id);
    assert(it != leafValues.end() && "missing virtual POD leaf value");
    orderedValues.push_back(castValueToTypeIfNeeded(bldr, loc, it->second, leafType));
  }
  );
  return orderedValues;
}

/// Create a POD-typed placeholder for virtual leaf storage tracked in `leafValues`.
///
/// PODs that embed affine-map-parameterized arrays cannot always be represented by a bare
/// `pod.new` at this stage because there may be no op-local instantiation operands available.
/// Use an unrealized cast from the ordered leaf values for those cases; later rewrites consult
/// `virtualPods` directly, and only concrete `pod.new` placeholders require materialization.
static Value createVirtualPodPlaceholder(
    OpBuilder &bldr, Location loc, PodType podTy, const VirtualPodLeafMap &leafValues
) {
  if (!hasAffineMapAttr(podTy)) {
    return bldr.create<NewPodOp>(loc, podTy);
  }

  SmallVector<Value> orderedValues = orderedVirtualPodLeafValues(podTy, loc, bldr, leafValues);
  return bldr.create<UnrealizedConversionCastOp>(loc, TypeRange {podTy}, orderedValues)
      .getResult(0);
}

/// Materialize the tracked contents of a virtual POD into concrete `pod.write` operations.
inline static void
materializeVirtualPod(OpBuilder &bldr, NewPodOp pod, const VirtualPodLeafMap &leafValues) {
  Location loc = pod.getLoc();
  PodType podTy = pod.getType();
  SmallVector<StringAttr> recordChain;
  for (RecordAttr record : podTy.getRecords()) {
    recordChain.push_back(record.getName());
    Value recordValue =
        rebuildFlattenedPodRecord(bldr, loc, record.getType(), recordChain, leafValues);
    genWrite(bldr, loc, pod, record.getName(), recordValue);
    recordChain.pop_back();
  }
}

/// Return `true` iff a read from a virtual POD can be resolved without materializing it.
static bool canResolveVirtualPodRead(ReadPodOp op, const VirtualPodValueMap &virtualPods) {
  if (!lookupVirtualPodLeafMap(op.getPodRef(), virtualPods) || hasEarlierWriteInBlock(op) ||
      findNearestForwardableWriteInBlock(op)) {
    return false;
  }
  Type recType = llvm::cast<PodType>(op.getPodRefType()).getRecordMap().lookup(op.getRecordName());
  return llvm::isa<PodType>(recType) || !splittablePodArray(recType);
}

/// Return `true` iff step 2 should defer splitting this array read until POD-aware rewriting.
static bool shouldDeferPodArrayReadToStep3(ReadArrayOp op) {
  return splittablePodArray(op.getArrRefType()) &&
         llvm::isa_and_present<ReadPodOp>(op.getArrRef().getDefiningOp());
}

/// Return the suffixes to append to a function arg/result name when splitting the given type.
static SmallVector<std::string> getSplitRecordNameSuffixes(Type type) {
  SmallVector<std::string> suffixes;
  if (PodType pt = splittablePod(type)) {
    SmallVector<StringAttr> recordChain;
    forEachPodLeaf(pt, recordChain, [&suffixes](const RecordChain &id, Type) {
      std::string suffix;
      llvm::raw_string_ostream os(suffix);
      for (StringAttr recordName : id.nameList) {
        os << '.' << recordName.getValue();
      }
      suffixes.push_back(std::move(suffix));
    });
  }
  return suffixes;
}

// If the operand has PodType, add reads from all pod records to the `newOperands` list otherwise
// add the original operand to the list.
static void processInputOperand(
    Location loc, Value operand, SmallVector<Value> &newOperands,
    ConversionPatternRewriter &rewriter, Operation *userOp = nullptr,
    const VirtualPodValueMap *virtualPods = nullptr
) {
  if (PodType pt = splittablePod(operand.getType())) {
    if (virtualPods) {
      if (const VirtualPodLeafMap *leafValues = lookupVirtualPodLeafMap(operand, *virtualPods);
          leafValues && (!userOp || !hasEarlierWriteToPodInBlock(userOp, operand))) {
        llvm::append_range(
            newOperands, orderedVirtualPodLeafValues(pt, loc, rewriter, *leafValues)
        );
        return;
      }
    }
    SmallVector<StringAttr> recordChain;
    forEachPodLeaf(pt, recordChain, [&](const RecordChain &id, Type) {
      newOperands.push_back(genReadAlongPath(rewriter, loc, operand, id));
    });
  } else {
    newOperands.push_back(operand);
  }
}

/// For each operand with PodType, add reads from all pod records in place of the original operand
/// and update the op to use the new operands.
static void processInputOperands(
    ValueRange operands, MutableOperandRange outputOpRef, Operation *op,
    ConversionPatternRewriter &rewriter, const VirtualPodValueMap *virtualPods = nullptr
) {
  SmallVector<Value> newOperands;
  for (Value v : operands) {
    processInputOperand(op->getLoc(), v, newOperands, rewriter, op, virtualPods);
  }
  rewriter.modifyOpInPlace(op, [&outputOpRef, &newOperands]() {
    outputOpRef.assign(ValueRange(newOperands));
  });
}

/// Update the tracked leaf values for one top-level POD record after a virtual `pod.write`.
static void updateVirtualPodRecordLeafValues(
    Location loc, StringAttr recordName, Type recordType, Value recordValue,
    const VirtualPodValueMap &virtualPods, ConversionPatternRewriter &rewriter,
    VirtualPodLeafMap &leafValues
) {
  SmallVector<StringAttr> prefix {recordName};

  if (PodType nestedPodTy = llvm::dyn_cast<PodType>(recordType)) {
    if (const VirtualPodLeafMap *nestedLeafValues =
            lookupVirtualPodLeafMap(recordValue, virtualPods)) {
      SmallVector<StringAttr> nestedRecordChain;
      forEachPodLeaf(nestedPodTy, nestedRecordChain, [&](const RecordChain &id, Type) {
        SmallVector<StringAttr> fullChain(prefix);
        llvm::append_range(fullChain, id.nameList);
        leafValues[RecordChain(fullChain)] = nestedLeafValues->at(id);
      });
      return;
    }

    SmallVector<StringAttr> nestedRecordChain;
    forEachPodLeaf(nestedPodTy, nestedRecordChain, [&](const RecordChain &id, Type) {
      SmallVector<StringAttr> fullChain(prefix);
      llvm::append_range(fullChain, id.nameList);
      leafValues[RecordChain(fullChain)] = genReadAlongPath(rewriter, loc, recordValue, id);
    });
    return;
  }

  if (ArrayType arrTy = splittablePodArray(recordType)) {
    auto elemPodTy = llvm::cast<PodType>(arrTy.getElementType());
    SmallVector<StringAttr> nestedRecordChain;
    forEachPodLeaf(elemPodTy, nestedRecordChain, [&](const RecordChain &id, Type) {
      SmallVector<StringAttr> fullChain(prefix);
      llvm::append_range(fullChain, id.nameList);
      leafValues[RecordChain(fullChain)] = genReadAlongPath(rewriter, loc, recordValue, id);
    });
    return;
  }

  leafValues[RecordChain(prefix)] = castValueToTypeIfNeeded(rewriter, loc, recordValue, recordType);
}

/// Register the dialects and operations that remain legal across the conversion-based stages.
inline static void baseTargetSetup(ConversionTarget &target) {
  target.addLegalDialect<
      LLZKDialect, array::ArrayDialect, boolean::BoolDialect, cast::CastDialect,
      constrain::ConstrainDialect, component::StructDialect, felt::FeltDialect,
      function::FunctionDialect, global::GlobalDialect, include::IncludeDialect, pod::PODDialect,
      polymorphic::PolymorphicDialect, ram::RAMDialect, string::StringDialect, arith::ArithDialect,
      scf::SCFDialect>();
  target.addLegalOp<ModuleOp>();
}

/// Rewrite pod-typed `llzk.nondet` allocations into explicit `pod.new` allocations so the rest of
/// the pass only needs to reason about POD storage through POD dialect operations.
class NondetToNewPod : public OpConversionPattern<NonDetOp> {
  using OpConversionPattern<NonDetOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      NonDetOp nondetOp, OpAdaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (auto pt = dyn_cast<PodType>(nondetOp.getType())) {
      rewriter.replaceOpWithNewOp<NewPodOp>(nondetOp, pt);
      return success();
    }
    return failure();
  }
};

/// Prepare the module by replacing `llzk.nondet` pod allocation ops with `pod.new`.
static LogicalResult step0(ModuleOp modOp) {
  MLIRContext *ctx = modOp.getContext();
  RewritePatternSet patterns {ctx};
  patterns.add<NondetToNewPod>(ctx);
  ConversionTarget target {*ctx};

  baseTargetSetup(target);
  target.addDynamicallyLegalOp<NonDetOp>([](NonDetOp op) { return !isa<PodType>(op.getType()); });

  return applyFullConversion(modOp, target, std::move(patterns));
}

/// new member name and type
using MemberInfo = std::pair<StringAttr, Type>;
/// original nested pod record name chain -> split scalar member info
using LocalMemberReplacementMap = DenseMap<RecordChain, MemberInfo>;
/// struct -> original pod-type member name -> LocalMemberReplacementMap
using MemberReplacementMap = DenseMap<StructDefOp, DenseMap<StringAttr, LocalMemberReplacementMap>>;

/// Build a flattened struct-member name like `member_outer_inner_leaf`.
static StringAttr
getFlattenedMemberName(MLIRContext *ctx, StringAttr memberName, ArrayRef<StringAttr> recordChain) {
  std::string flatName;
  llvm::raw_string_ostream os(flatName);
  os << memberName.getValue();
  for (StringAttr recordName : recordChain) {
    os << '_' << recordName.getValue();
  }
  return StringAttr::get(ctx, flatName);
}

/// Recursively create scalar leaf members for a POD-typed struct member.
static void flattenPodMemberIntoLeaves(
    MemberDefOp originalMember, PodType podTy, SmallVectorImpl<StringAttr> &recordChain,
    LocalMemberReplacementMap &localRepMapRef, SymbolTable &structSymbolTable,
    ConversionPatternRewriter &rewriter
) {
  forEachPodLeaf(podTy, recordChain, [&](const RecordChain &id, Type ty) {
    StringAttr name = getFlattenedMemberName(
        originalMember.getContext(), originalMember.getSymNameAttr(), id.nameList
    );
    MemberDefOp newMember = rewriter.create<MemberDefOp>(
        originalMember.getLoc(), name, ty, originalMember.getSignal(), originalMember.getColumn()
    );
    newMember.setPublicAttr(originalMember.hasPublicAttr());
    localRepMapRef[id] = std::make_pair(structSymbolTable.insert(newMember), ty);
  });
}

/// Split a pod-typed struct member definition into one scalar member definition per POD record.
///
/// The replacement map records the fresh member symbols so later rewrites can retarget
/// `struct.readm` and `struct.writem` operations to the split members.
class SplitPodInMemberDefOp : public OpConversionPattern<MemberDefOp> {
  SymbolTableCollection &tables;
  MemberReplacementMap &repMapRef;

public:
  SplitPodInMemberDefOp(
      MLIRContext *ctx, SymbolTableCollection &symTables, MemberReplacementMap &memberRepMap
  )
      : OpConversionPattern<MemberDefOp>(ctx), tables(symTables), repMapRef(memberRepMap) {}

  inline static bool legal(MemberDefOp op) { return !splittablePod(op.getType()); }

  LogicalResult match(MemberDefOp op) const override { return failure(legal(op)); }

  void
  rewrite(MemberDefOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    StructDefOp inStruct = op->getParentOfType<StructDefOp>();
    assert(inStruct);
    LocalMemberReplacementMap &localRepMapRef = repMapRef[inStruct][op.getSymNameAttr()];

    PodType podTy = llvm::cast<PodType>(adaptor.getType()); // safe per legal() check

    SymbolTable &structSymbolTable = tables.getSymbolTable(inStruct);
    SmallVector<StringAttr> recordChain;
    flattenPodMemberIntoLeaves(op, podTy, recordChain, localRepMapRef, structSymbolTable, rewriter);
    rewriter.eraseOp(op);
  }
};

/// Split an array-of-POD struct member definition into one parallel array member per POD leaf.
class SplitPodArrayInMemberDefOp : public OpConversionPattern<MemberDefOp> {
  SymbolTableCollection &tables;
  MemberReplacementMap &repMapRef;

public:
  SplitPodArrayInMemberDefOp(
      MLIRContext *ctx, SymbolTableCollection &symTables, MemberReplacementMap &memberRepMap
  )
      : OpConversionPattern<MemberDefOp>(ctx), tables(symTables), repMapRef(memberRepMap) {}

  inline static bool legal(MemberDefOp op) { return !splittablePodArray(op.getType()); }

  LogicalResult match(MemberDefOp op) const override { return failure(legal(op)); }

  void
  rewrite(MemberDefOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    StructDefOp inStruct = op->getParentOfType<StructDefOp>();
    assert(inStruct);
    LocalMemberReplacementMap &localRepMapRef = repMapRef[inStruct][op.getSymNameAttr()];

    ArrayType arrTy = llvm::cast<ArrayType>(adaptor.getType());
    SmallVector<RecordChain> splitIds;
    SmallVector<Type> splitTypes;
    splitPodArrayTypeTo(arrTy, splitTypes, &splitIds);

    SymbolTable &structSymbolTable = tables.getSymbolTable(inStruct);
    for (auto [id, splitType] : llvm::zip_equal(splitIds, splitTypes)) {
      StringAttr name = getFlattenedMemberName(op.getContext(), op.getSymNameAttr(), id.nameList);
      MemberDefOp newMember = rewriter.create<MemberDefOp>(
          op.getLoc(), name, splitType, op.getSignal(), op.getColumn()
      );
      newMember.setPublicAttr(op.hasPublicAttr());
      localRepMapRef[id] = std::make_pair(structSymbolTable.insert(newMember), splitType);
    }
    rewriter.eraseOp(op);
  }
};

/// Replace direct `PodType` struct members with scalar members and arrays-of-POD with parallel
/// array members named after the corresponding POD leaf.
static LogicalResult
step1(ModuleOp modOp, SymbolTableCollection &symTables, MemberReplacementMap &memberRepMap) {
  MLIRContext *ctx = modOp.getContext();

  RewritePatternSet patterns(ctx);

  patterns.add<SplitPodInMemberDefOp, SplitPodArrayInMemberDefOp>(ctx, symTables, memberRepMap);

  ConversionTarget target(*ctx);
  baseTargetSetup(target);
  target.addDynamicallyLegalOp<MemberDefOp>([](MemberDefOp op) {
    return SplitPodInMemberDefOp::legal(op) && SplitPodArrayInMemberDefOp::legal(op);
  });

  LLVM_DEBUG(llvm::dbgs() << "Begin step 1: split pod-type and array-of-pod members\n";);
  return applyFullConversion(modOp, target, std::move(patterns));
}

/// Type converter that replaces each array-of-POD type with one parallel array type per POD leaf.
///
/// Besides splitting result types, this also materializes compatibility casts between precise
/// split array types and wildcard-backed storage split types when target or block-argument
/// conversion needs to cross that boundary.
class PodArrayTypeConverter : public TypeConverter {
public:
  PodArrayTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(
        [](ArrayType arrTy, SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
      if (!splittablePodArray(arrTy)) {
        return std::nullopt;
      }
      splitPodArrayTypeTo(arrTy, results);
      return success();
    }
    );

    auto materializeCast = [](OpBuilder &bldr, Type targetType, ValueRange inputs,
                              Location loc) -> Value {
      if (inputs.size() != 1 || !typesUnify(inputs.front().getType(), targetType)) {
        return {};
      }
      return castValueToTypeIfNeeded(bldr, loc, inputs.front(), targetType);
    };
    addTargetMaterialization(materializeCast);
    addArgumentMaterialization(materializeCast);
  }
};

/// Split `llzk.nondet` of array-of-POD type into one `llzk.nondet` per parallel leaf array.
class SplitPodArrayNonDetOp : public OpConversionPattern<NonDetOp> {
public:
  using OpConversionPattern<NonDetOp>::OpConversionPattern;

  static bool legal(NonDetOp op) { return !splittablePodArray(op.getType()); }

  LogicalResult match(NonDetOp op) const override { return failure(legal(op)); }

  void rewrite(NonDetOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> splitTypes;
    splitPodArrayTypeTo(op.getType(), splitTypes);
    SmallVector<Value> replacements;
    replacements.reserve(splitTypes.size());
    for (Type splitType : splitTypes) {
      replacements.push_back(rewriter.create<NonDetOp>(op.getLoc(), splitType));
    }
    rewriter.replaceOpWithMultiple(op, {ValueRange(replacements)});
  }
};

/// Split `array.new` of array-of-POD type into one `array.new` per parallel leaf array.
///
/// For each leaf, the precise split type preserves the original affine maps in the flattened leaf
/// array. When hidden leaf-array affine dims have no direct witness, the rewrite may first build a
/// wildcard-backed storage split type with the same outer shape and cast back to the precise type.
///
/// Uninitialized `array.new` uses that storage fallback directly when needed. Explicit-element
/// `array.new` tries to infer one shared affine-map instantiation from all leaf arrays so it can
/// materialize the precise split type immediately. If different elements imply conflicting
/// instantiations, the rewrite remains a hard failure.
class SplitPodArrayCreateArrayOp : public OpConversionPattern<CreateArrayOp> {
public:
  using OpConversionPattern<CreateArrayOp>::OpConversionPattern;

  static bool legal(CreateArrayOp op) { return !splittablePodArray(op.getType()); }

  LogicalResult matchAndRewrite(
      CreateArrayOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (legal(op)) {
      return failure();
    }
    ArrayType arrTy = llvm::cast<ArrayType>(op.getType());
    SmallVector<RecordChain> splitIds;
    SmallVector<Type> splitTypes;
    splitPodArrayTypeTo(arrTy, splitTypes, &splitIds);

    SmallVector<Value> replacements;
    replacements.reserve(splitTypes.size());
    DenseI32ArrayAttr numDimsPerMap = op.getNumDimsPerMapAttr();
    if (isNullOrEmpty(numDimsPerMap)) {
      if (adaptor.getElements().empty()) {
        for (auto [id, splitType] : llvm::zip_equal(splitIds, splitTypes)) {
          ArrayType preciseSplitType = llvm::cast<ArrayType>(splitType);
          ArrayType storageSplitType = getSplitPodArrayStorageType(arrTy, id.nameList);
          Value splitArray = rewriter.create<CreateArrayOp>(op.getLoc(), storageSplitType);
          replacements.push_back(
              castValueToTypeIfNeeded(rewriter, op.getLoc(), splitArray, preciseSplitType)
          );
        }
        rewriter.replaceOpWithMultiple(op, {ValueRange(replacements)});
        return success();
      }

      auto elementIndices = arrTy.getSubelementIndices();
      assert(elementIndices && "array.new with explicit elements requires a static array shape");
      assert(
          elementIndices->size() == adaptor.getElements().size() &&
          "array.new element count must match the outer array cardinality"
      );

      // Inline initializers are linearized only across the original outer array dimensions. When
      // a flattened POD leaf is itself an array, populate the rewritten split array one outer
      // element at a time so each leaf array becomes a subarray insert rather than a malformed
      // inline operand to the flattened `array.new`.
      for (auto [id, splitType] : llvm::zip_equal(splitIds, splitTypes)) {
        ArrayType preciseSplitType = llvm::cast<ArrayType>(splitType);
        ArrayType storageSplitType = getSplitPodArrayStorageType(arrTy, id.nameList);

        SmallVector<Value> leafValues;
        leafValues.reserve(adaptor.getElements().size());
        for (ValueRange elementRange : adaptor.getElements()) {
          Value element = getSingleConvertedValue(elementRange);
          leafValues.push_back(genReadAlongPath(rewriter, op.getLoc(), element, id));
        }

        ArrayType materializedType = storageSplitType;
        Value splitArray;
        if (storageSplitType != preciseSplitType) {
          ArrayInstantiationInfo instantiationInfo;
          switch (inferCommonArrayInstantiation(leafValues, instantiationInfo)) {
          case CommonArrayInstantiationStatus::conflict:
            // TODO: this POD could be promoted to a complete `struct.def` but that's not easy.
            op.emitOpError(
                "with POD elements having conflicting affine map instantiations cannot be promoted "
                "to higher dimensional array"
            );
            return failure();
          case CommonArrayInstantiationStatus::inferred: {
            materializedType = preciseSplitType;
            SmallVector<ValueRange> mapOperands;
            mapOperands.reserve(instantiationInfo.mapOperandStorage.size());
            for (const SmallVector<Value> &values : instantiationInfo.mapOperandStorage) {
              mapOperands.push_back(values);
            }
            splitArray = rewriter.create<CreateArrayOp>(
                op.getLoc(), materializedType, mapOperands, instantiationInfo.numDimsPerMap
            );
            break;
          }
          case CommonArrayInstantiationStatus::unavailable:
            break;
          }
        }

        if (!splitArray) {
          splitArray = createWritableArrayValue(rewriter, op.getLoc(), materializedType);
        }

        for (auto [index, leafValue] : llvm::zip_equal(*elementIndices, leafValues)) {
          genArrayWrite(rewriter, op.getLoc(), splitArray, index, leafValue);
        }
        replacements.push_back(
            castValueToTypeIfNeeded(rewriter, op.getLoc(), splitArray, preciseSplitType)
        );
      }
    } else {
      SmallVector<SmallVector<Value>> mapOperandStorage;
      SmallVector<ValueRange> mapOperands;
      mapOperandStorage.reserve(adaptor.getMapOperands().size());
      mapOperands.reserve(adaptor.getMapOperands().size());
      for (ArrayRef<ValueRange> mapOperandGroup : adaptor.getMapOperands()) {
        mapOperandStorage.push_back(flattenConvertedValues(mapOperandGroup));
      }
      for (const SmallVector<Value> &values : mapOperandStorage) {
        mapOperands.push_back(values);
      }
      for (auto [id, splitType] : llvm::zip_equal(splitIds, splitTypes)) {
        ArrayType preciseSplitType = llvm::cast<ArrayType>(splitType);
        ArrayType storageSplitType = getSplitPodArrayStorageType(arrTy, id.nameList);
        Value splitArray = rewriter.create<CreateArrayOp>(
            op.getLoc(), storageSplitType, mapOperands, numDimsPerMap
        );
        replacements.push_back(
            castValueToTypeIfNeeded(rewriter, op.getLoc(), splitArray, preciseSplitType)
        );
      }
    }

    rewriter.replaceOpWithMultiple(op, {ValueRange(replacements)});
    return success();
  }
};

/// Split `array.read` from an array-of-POD into leaf reads plus local POD reconstruction.
class SplitPodArrayReadArrayOp : public OpConversionPattern<ReadArrayOp> {
public:
  using OpConversionPattern<ReadArrayOp>::OpConversionPattern;

  static bool legal(ReadArrayOp op) {
    return !splittablePodArray(op.getArrRefType()) || shouldDeferPodArrayReadToStep3(op);
  }

  LogicalResult matchAndRewrite(
      ReadArrayOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (legal(op)) {
      return failure();
    }
    ArrayType arrTy = op.getArrRefType();
    PodType podTy = llvm::cast<PodType>(arrTy.getElementType());
    SmallVector<RecordChain> splitIds;
    SmallVector<Type> splitTypes;
    splitPodArrayTypeTo(arrTy, splitTypes, &splitIds);

    SmallVector<Value> indices = flattenConvertedValues(adaptor.getIndices());
    NewPodOp pod = rewriter.create<NewPodOp>(op.getLoc(), podTy);
    DenseMap<RecordChain, Value> leafValues;
    for (auto [id, splitArrRange] : llvm::zip_equal(splitIds, adaptor.getArrRef())) {
      leafValues[id] =
          genArrayRead(rewriter, op.getLoc(), getSingleConvertedValue(splitArrRange), indices);
    }

    SmallVector<StringAttr> recordChain;
    for (RecordAttr record : podTy.getRecords()) {
      recordChain.push_back(record.getName());
      Value recordValue = rebuildFlattenedPodRecord(
          rewriter, op.getLoc(), record.getType(), recordChain, leafValues
      );
      genWrite(rewriter, op.getLoc(), pod, record.getName(), recordValue);
      recordChain.pop_back();
    }
    rewriter.replaceOp(op, pod);
    return success();
  }
};

/// Split `array.write` to an array-of-POD into one write per parallel leaf array.
class SplitPodArrayWriteArrayOp : public OpConversionPattern<WriteArrayOp> {
public:
  using OpConversionPattern<WriteArrayOp>::OpConversionPattern;

  static bool legal(WriteArrayOp op) { return !splittablePodArray(op.getArrRefType()); }

  LogicalResult matchAndRewrite(
      WriteArrayOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (legal(op)) {
      return failure();
    }
    ArrayType arrTy = op.getArrRefType();
    SmallVector<RecordChain> splitIds;
    SmallVector<Type> splitTypes;
    splitPodArrayTypeTo(arrTy, splitTypes, &splitIds);

    SmallVector<Value> indices = flattenConvertedValues(adaptor.getIndices());
    Value podValue = getSingleConvertedValue(adaptor.getRvalue());
    for (auto [id, splitArrRange, splitType] :
         llvm::zip_equal(splitIds, adaptor.getArrRef(), splitTypes)) {
      Value leafValue = genReadAlongPath(rewriter, op.getLoc(), podValue, id);
      genArrayWrite(
          rewriter, op.getLoc(), getSingleConvertedValue(splitArrRange), indices, leafValue
      );
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// Rewrite array-of-POD function signatures to use one parallel array per POD leaf.
class SplitPodArrayInFuncDefOp : public OpConversionPattern<FuncDefOp> {
public:
  using OpConversionPattern<FuncDefOp>::OpConversionPattern;

  static bool legal(FuncDefOp op) {
    return !containsSplittablePodArrayType(op.getArgumentTypes()) &&
           !containsSplittablePodArrayType(op.getResultTypes());
  }

  LogicalResult match(FuncDefOp op) const override { return failure(legal(op)); }

  LogicalResult
  matchAndRewrite(FuncDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    const auto *tyConv = getTypeConverter();
    assert(tyConv && "expected pod-array type converter");

    FunctionType oldTy = op.getFunctionType();
    TypeConverter::SignatureConversion inputConversion(oldTy.getNumInputs());
    if (failed(tyConv->convertSignatureArgs(oldTy.getInputs(), inputConversion))) {
      return rewriter.notifyMatchFailure(op, "failed to convert array-of-pod inputs");
    }

    SmallVector<Type> newResults;
    if (failed(tyConv->convertTypes(oldTy.getResults(), newResults))) {
      return rewriter.notifyMatchFailure(op, "failed to convert array-of-pod results");
    }

    if (!op.getBody().empty() &&
        failed(rewriter.convertRegionTypes(&op.getBody(), *tyConv, &inputConversion))) {
      return rewriter.notifyMatchFailure(op, "failed to convert function body block arguments");
    }

    SmallVector<size_t> originalInputIdxToSize, originalResultIdxToSize;
    SmallVector<Type> newInputs = splitPodArrayType(oldTy.getInputs(), &originalInputIdxToSize);
    SmallVector<Type> newResultsWithSizeInfo =
        splitPodArrayType(oldTy.getResults(), &originalResultIdxToSize);
    assert(
        newResultsWithSizeInfo == newResults &&
        "expected array-of-pod type conversion to match function result attr replication"
    );
    SplitFunctionNameInfo inputNameInfo =
        collectSplitFunctionNameInfo(op.getArgumentTypes(), [&](unsigned i) {
      return op.getArgNameAttr(i);
    }, getSplitPodArrayRecordNameSuffixes);
    ArrayAttr resultAttrs = op.getAllResultAttrs();
    SplitFunctionNameInfo resultNameInfo =
        collectSplitFunctionNameInfo(op.getResultTypes(), [resultAttrs](unsigned i) {
      return getAttrAtIndexWithName(resultAttrs, i, RES_NAME_ATTR_NAME);
    }, getSplitPodArrayRecordNameSuffixes);

    rewriter.modifyOpInPlace(op, [&]() {
      op.setFunctionType(FunctionType::get(op.getContext(), newInputs, newResults));
      if (ArrayAttr newArgAttrs = replicateFunctionNameAttrsAsNeeded(
              op.getArgAttrsAttr(), originalInputIdxToSize, newInputs, ARG_NAME_ATTR_NAME,
              inputNameInfo.originalNames, inputNameInfo.existingNames,
              inputNameInfo.splitNameSuffixes
          )) {
        op.setArgAttrsAttr(newArgAttrs);
      }
      if (ArrayAttr newResAttrs = replicateFunctionNameAttrsAsNeeded(
              op.getResAttrsAttr(), originalResultIdxToSize, newResults, RES_NAME_ATTR_NAME,
              resultNameInfo.originalNames, resultNameInfo.existingNames,
              resultNameInfo.splitNameSuffixes
          )) {
        op.setResAttrsAttr(newResAttrs);
      }
    });
    return success();
  }
};

/// Append the split leaf-array values for one step-2 operand.
///
/// When dialect conversion has already produced the parallel leaf arrays, reuse those converted
/// values directly. Otherwise derive the split arrays from the original aggregate operand so users
/// like `poly.unifiable_cast` and `function.return` can still flatten a raw `pod.read` of an array
/// field.
static void collectSplitPodArrayOperandValues(
    Location loc, Value originalOperand, ValueRange convertedValues,
    SmallVectorImpl<Value> &newOperands, ConversionPatternRewriter &rewriter
) {
  ArrayType arrTy = splittablePodArray(originalOperand.getType());
  if (!arrTy) {
    llvm::append_range(newOperands, convertedValues);
    return;
  }

  SmallVector<RecordChain> splitIds;
  SmallVector<Type> splitTypes;
  splitPodArrayTypeTo(arrTy, splitTypes, &splitIds);

  auto isDirectAggregateToSplitCast = [&convertedValues, &originalOperand]() {
    if (convertedValues.empty()) {
      return false;
    }
    auto castOp = convertedValues.front().getDefiningOp<UnrealizedConversionCastOp>();
    if (!castOp || castOp->getNumOperands() != 1 || castOp.getOperand(0) != originalOperand) {
      return false;
    }
    return llvm::all_of(convertedValues, [&castOp](Value value) {
      return value.getDefiningOp<UnrealizedConversionCastOp>() == castOp;
    });
  };

  if (!isDirectAggregateToSplitCast() && convertedValues.size() == splitTypes.size() &&
      llvm::all_of(llvm::zip_equal(convertedValues, splitTypes), [](auto pair) {
    return typesUnify(std::get<0>(pair).getType(), std::get<1>(pair));
  })) {
    llvm::append_range(newOperands, convertedValues);
    return;
  }

  for (auto [id, splitType] : llvm::zip_equal(splitIds, splitTypes)) {
    Value splitValue = genReadAlongPath(rewriter, loc, originalOperand, id);
    newOperands.push_back(castValueToTypeIfNeeded(rewriter, loc, splitValue, splitType));
  }
}

/// Rewrite array-of-POD `poly.unifiable_cast` into one leaf-array cast per split array.
class SplitPodArrayInUnifiableCastOp : public OpConversionPattern<UnifiableCastOp> {
public:
  using OpConversionPattern<UnifiableCastOp>::OpConversionPattern;

  static bool legal(UnifiableCastOp op) { return !splittablePodArray(op.getType()); }

  LogicalResult matchAndRewrite(
      UnifiableCastOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (legal(op)) {
      return failure();
    }

    ArrayType inputArrTy = splittablePodArray(op.getInput().getType());
    if (!inputArrTy) {
      return rewriter.notifyMatchFailure(
          op, "expected array-of-pod cast input when rewriting array-of-pod cast result"
      );
    }

    SmallVector<RecordChain> inputSplitIds;
    SmallVector<Type> inputSplitTypes;
    splitPodArrayTypeTo(inputArrTy, inputSplitTypes, &inputSplitIds);

    ArrayType resultArrTy = llvm::cast<ArrayType>(op.getType());
    SmallVector<RecordChain> resultSplitIds;
    SmallVector<Type> resultSplitTypes;
    splitPodArrayTypeTo(resultArrTy, resultSplitTypes, &resultSplitIds);

    if (inputSplitIds != resultSplitIds) {
      return rewriter.notifyMatchFailure(
          op, "array-of-pod cast changed POD leaf structure unexpectedly"
      );
    }

    SmallVector<Value> splitInputs;
    collectSplitPodArrayOperandValues(
        op.getLoc(), op.getInput(), adaptor.getInput(), splitInputs, rewriter
    );
    if (splitInputs.size() != resultSplitTypes.size()) {
      return rewriter.notifyMatchFailure(
          op, "failed to collect one split input per array-of-pod cast leaf"
      );
    }

    SmallVector<Value> replacements;
    replacements.reserve(resultSplitTypes.size());
    for (auto [splitInput, resultSplitType] : llvm::zip_equal(splitInputs, resultSplitTypes)) {
      replacements.push_back(
          castValueToTypeIfNeeded(rewriter, op.getLoc(), splitInput, resultSplitType)
      );
    }

    rewriter.replaceOpWithMultiple(op, {ValueRange(replacements)});
    return success();
  }
};

/// Rewrite `function.return` to flatten any array-of-POD operands into their parallel arrays.
class SplitPodArrayInReturnOp : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  static bool legal(ReturnOp op) {
    return !containsSplittablePodArrayType(op.getOperands().getTypes());
  }

  LogicalResult matchAndRewrite(
      ReturnOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (legal(op)) {
      return failure();
    }
    SmallVector<Value> newOperands;
    for (auto [operand, convertedValues] :
         llvm::zip_equal(op.getOperands(), adaptor.getOperands())) {
      collectSplitPodArrayOperandValues(
          op.getLoc(), operand, convertedValues, newOperands, rewriter
      );
    }
    rewriter.replaceOpWithNewOp<ReturnOp>(op, ValueRange(newOperands));
    return success();
  }
};

/// Rewrite calls whose arguments or results contain arrays-of-POD to use the split signature.
class SplitPodArrayInCallOp : public OpConversionPattern<CallOp> {
public:
  using OpConversionPattern<CallOp>::OpConversionPattern;

  static bool legal(CallOp op) {
    return !containsSplittablePodArrayType(op.getArgOperands().getTypes()) &&
           !containsSplittablePodArrayType(op.getResultTypes());
  }

  LogicalResult match(CallOp op) const override { return failure(legal(op)); }

  LogicalResult matchAndRewrite(
      CallOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    const auto *tyConv = getTypeConverter();
    assert(tyConv && "expected pod-array type converter");

    SmallVector<Type> newResultTypes;
    if (failed(tyConv->convertTypes(op.getResultTypes(), newResultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert array-of-pod call results");
    }

    SmallVector<SmallVector<Value>> mapOperandStorage;
    SmallVector<ValueRange> mapOperands;
    mapOperandStorage.reserve(adaptor.getMapOperands().size());
    mapOperands.reserve(adaptor.getMapOperands().size());
    for (ArrayRef<ValueRange> mapOperandGroup : adaptor.getMapOperands()) {
      mapOperandStorage.push_back(flattenConvertedValues(mapOperandGroup));
    }
    for (const SmallVector<Value> &values : mapOperandStorage) {
      mapOperands.push_back(values);
    }

    SmallVector<Value> newArgOperands = flattenConvertedValues(adaptor.getArgOperands());
    CallOp newCall = createCallPreservingInstantiationOperands(
        op.getLoc(), newResultTypes, op, mapOperands, newArgOperands, rewriter
    );

    SmallVector<SmallVector<Value>> replacementStorage;
    replacementStorage.reserve(op.getNumResults());
    auto newResultIt = newCall.getResults().begin();
    for (Type oldResultType : op.getResultTypes()) {
      SmallVector<Type> convertedTypes;
      (void)splitPodArrayTypeTo(oldResultType, convertedTypes);
      SmallVector<Value> replacementsForResult;
      replacementsForResult.reserve(convertedTypes.size());
      for (size_t i = 0; i < convertedTypes.size(); ++i) {
        replacementsForResult.push_back(*newResultIt);
        ++newResultIt;
      }
      replacementStorage.push_back(std::move(replacementsForResult));
    }

    SmallVector<ValueRange> replacements;
    replacements.reserve(replacementStorage.size());
    for (const SmallVector<Value> &values : replacementStorage) {
      replacements.push_back(values);
    }
    rewriter.replaceOpWithMultiple(op, replacements);
    return success();
  }
};

/// Rewrite `constrain.eq` over arrays-of-POD into one equality per parallel leaf array.
class SplitPodArrayInEmitEqualityOp : public OpConversionPattern<constrain::EmitEqualityOp> {
public:
  using OpConversionPattern<constrain::EmitEqualityOp>::OpConversionPattern;

  static bool legal(constrain::EmitEqualityOp op) {
    return !containsSplittablePodArrayType(op->getOperandTypes());
  }

  LogicalResult matchAndRewrite(
      constrain::EmitEqualityOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (legal(op)) {
      return failure();
    }

    if (adaptor.getLhs().size() != adaptor.getRhs().size()) {
      return rewriter.notifyMatchFailure(
          op, "expected array-of-pod equality operands to expand to the same number of leaves"
      );
    }

    for (auto [lhs, rhs] : llvm::zip_equal(adaptor.getLhs(), adaptor.getRhs())) {
      rewriter.create<constrain::EmitEqualityOp>(op.getLoc(), lhs, rhs);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// Rewrite `constrain.in` over arrays-of-POD into a shared-slice witness plus leaf equalities.
///
/// After step 2 converts an array-of-POD into parallel leaf arrays, `constrain.in` can no longer be
/// left in place because it has no built-in 1:N operand rewrite. This pattern preserves the
/// original containment semantics by:
///
/// 1. Expanding both operands into matching POD leaves.
/// 2. Computing how many leading lhs dimensions must be selected to match the rhs rank.
/// 3. Creating one nondeterministic index per selected dimension and constraining each index to be
///    in bounds using `array.len` and `constrain.eq` on the comparison results.
/// 4. Using that same index tuple for every leaf, reading a scalar leaf with `array.read` or
///    extracting an array leaf with `array.extract`.
/// 5. Emitting one `constrain.eq` per selected lhs leaf and rhs leaf, then erasing the original
///    `constrain.in`.
///
/// Reusing the same nondeterministic indices across all leaves is essential: it guarantees that all
/// field equalities refer to the same POD element or subarray, rather than allowing different
/// leaves to match at different positions.
class SplitPodArrayInEmitContainmentOp : public OpConversionPattern<constrain::EmitContainmentOp> {
public:
  using OpConversionPattern<constrain::EmitContainmentOp>::OpConversionPattern;

  static bool legal(constrain::EmitContainmentOp op) {
    return !containsSplittablePodArrayType(op->getOperandTypes());
  }

  /// Return the split scalar or leaf-array values representing one containment operand.
  static SmallVector<Value> collectContainmentLeaves(
      Location loc, Value originalOperand, ValueRange convertedValues,
      ConversionPatternRewriter &rewriter
  ) {
    if (splittablePod(originalOperand.getType())) {
      SmallVector<Value> podLeaves;
      processInputOperand(loc, getSingleConvertedValue(convertedValues), podLeaves, rewriter);
      return podLeaves;
    }

    return SmallVector<Value>(convertedValues.begin(), convertedValues.end());
  }

  LogicalResult matchAndRewrite(
      constrain::EmitContainmentOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (legal(op)) {
      return failure();
    }

    Location loc = op.getLoc();
    ArrayType lhsTy = op.getLhs().getType();
    Type rhsTy = op.getRhs().getType();

    size_t lhsRank = lhsTy.getDimensionSizes().size();
    size_t rhsRank = 0;
    if (auto rhsArrTy = llvm::dyn_cast<ArrayType>(rhsTy)) {
      rhsRank = rhsArrTy.getDimensionSizes().size();
    }
    assert(lhsRank >= rhsRank && "constrain.in verifier should reject higher-rank rhs arrays");
    size_t selectedDims = lhsRank - rhsRank;

    SmallVector<Value> lhsLeaves(adaptor.getLhs().begin(), adaptor.getLhs().end());
    SmallVector<Value> rhsLeaves =
        collectContainmentLeaves(loc, op.getRhs(), adaptor.getRhs(), rewriter);
    if (lhsLeaves.size() != rhsLeaves.size()) {
      return rewriter.notifyMatchFailure(
          op, "expected array-of-pod containment operands to expand to the same number of leaves"
      );
    }

    Value shapeCarrier = adaptor.getLhs().empty()
                             ? materializeArrayLengthCarrier(op.getLhs(), lhsTy, loc, rewriter)
                             : adaptor.getLhs().front();
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    Value trueVal = rewriter.create<arith::ConstantOp>(
        loc, IntegerAttr::get(IntegerType::get(rewriter.getContext(), 1), 1)
    );

    SmallVector<Value> selectedIndices;
    selectedIndices.reserve(selectedDims);
    for (size_t dim = 0; dim < selectedDims; ++dim) {
      Value idx = rewriter.create<NonDetOp>(loc, IndexType::get(rewriter.getContext()));
      Value dimVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexAttr(llzk::checkedCast<int64_t>(dim))
      );
      Value dimLen = rewriter.create<ArrayLengthOp>(loc, shapeCarrier, dimVal);

      Value nonNegative = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, idx, zero);
      rewriter.create<constrain::EmitEqualityOp>(loc, nonNegative, trueVal);

      Value inRange = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, idx, dimLen);
      rewriter.create<constrain::EmitEqualityOp>(loc, inRange, trueVal);

      selectedIndices.push_back(idx);
    }

    for (auto [lhsLeaf, rhsLeaf] : llvm::zip_equal(lhsLeaves, rhsLeaves)) {
      Value selectedLhs = lhsLeaf;
      if (auto rhsLeafArrTy = llvm::dyn_cast<ArrayType>(rhsLeaf.getType())) {
        if (!selectedIndices.empty()) {
          selectedLhs =
              rewriter.create<ExtractArrayOp>(loc, rhsLeafArrTy, lhsLeaf, selectedIndices);
        }
      } else {
        selectedLhs =
            rewriter.create<ReadArrayOp>(loc, rhsLeaf.getType(), lhsLeaf, selectedIndices);
      }
      rewriter.create<constrain::EmitEqualityOp>(loc, selectedLhs, rhsLeaf);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Return an array value whose visible rank still matches the original `array.len` source.
///
/// Split POD leaves always preserve the original outer dimensions, but array-valued leaves append
/// their own inner dimensions. Dynamic dimension indices must not be able to observe those extra
/// leaf-only dimensions, so when every converted leaf has higher rank this synthesizes a shape-only
/// carrier with the original rank instead.
static Value selectArrayLengthShapeSource(
    ArrayLengthOp op, ValueRange convertedArrRefs, ConversionPatternRewriter &rewriter
) {
  size_t originalRank = op.getArrRefType().getDimensionSizes().size();
  for (Value arrRef : convertedArrRefs) {
    auto arrTy = llvm::dyn_cast<ArrayType>(arrRef.getType());
    assert(arrTy && "converted array-of-POD operand must stay an array");
    if (arrTy.getDimensionSizes().size() == originalRank) {
      return arrRef;
    }
  }

  return materializeArrayLengthCarrier(op.getArrRef(), op.getArrRefType(), op.getLoc(), rewriter);
}

/// Replace `array.length` on an array-of-POD with an equivalent rank-preserving array value.
class SplitPodArrayLengthOp : public OpConversionPattern<ArrayLengthOp> {
public:
  using OpConversionPattern<ArrayLengthOp>::OpConversionPattern;

  static bool legal(ArrayLengthOp op) { return !splittablePodArray(op.getArrRefType()); }

  LogicalResult matchAndRewrite(
      ArrayLengthOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (legal(op)) {
      return failure();
    }
    Value arrRef = selectArrayLengthShapeSource(op, adaptor.getArrRef(), rewriter);
    rewriter.replaceOpWithNewOp<ArrayLengthOp>(
        op, arrRef, getSingleConvertedValue(adaptor.getDim())
    );
    return success();
  }
};

/// Rebuild the current quantifier iterand from one read or extract per split POD-array leaf.
static Value rebuildSplitPodArrayQuantifierIterValue(
    OpBuilder &bldr, Location loc, Type iterType, Value index, ArrayType sortType,
    ValueRange convertedSort
) {
  SmallVector<RecordChain> splitIds;
  SmallVector<Type> splitTypes;
  splitPodArrayTypeTo(sortType, splitTypes, &splitIds);
  assert(
      convertedSort.size() == splitIds.size() &&
      "converted quantifier sort must provide one value per POD-array leaf"
  );

  DenseMap<RecordChain, Value> leafValues;
  for (auto [id, leafArray] : llvm::zip_equal(splitIds, convertedSort)) {
    SmallVector<Value> indices {index};
    leafValues[id] = genArrayRead(bldr, loc, leafArray, indices);
  }

  SmallVector<StringAttr> recordChain;
  return rebuildFlattenedPodRecord(bldr, loc, iterType, recordChain, leafValues);
}

/// Lower a bool quantifier over an array-of-POD to an `scf.for` over the split leaf arrays.
template <typename QuantifierOp, typename CombineOp>
static LogicalResult rewriteSplitPodArrayQuantifier(
    QuantifierOp op, ValueRange convertedSort, ConversionPatternRewriter &rewriter,
    bool initialValue
) {
  ArrayType sortType = llvm::cast<ArrayType>(op.getSort().getType());
  Location loc = op.getLoc();

  Value shapeCarrier = convertedSort.empty()
                           ? materializeArrayLengthCarrier(op.getSort(), sortType, loc, rewriter)
                           : convertedSort.front();
  Value lowerBound = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
  Value upperBound = rewriter.create<ArrayLengthOp>(loc, shapeCarrier, lowerBound);
  Value step = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
  Value init = rewriter.create<arith::ConstantOp>(
      loc, IntegerAttr::get(IntegerType::get(rewriter.getContext(), 1), initialValue ? 1 : 0)
  );

  auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step, ValueRange {init});
  loop->setDiscardableAttrs(op->getDiscardableAttrDictionary());

  Block &loopBody = *loop.getBody();
  if (!loopBody.empty()) {
    rewriter.eraseOp(&loopBody.back());
  }

  rewriter.setInsertionPointToStart(&loopBody);
  Value iterValue = rebuildSplitPodArrayQuantifierIterValue(
      rewriter, loc, op.getBody()->getArgument(0).getType(), loop.getInductionVar(), sortType,
      convertedSort
  );

  IRMapping mapping;
  mapping.map(op.getBody()->getArgument(0), iterValue);

  for (Operation &nestedOp : op.getBody()->without_terminator()) {
    rewriter.clone(nestedOp, mapping);
  }

  auto yieldOp = llvm::cast<boolean::YieldOp>(op.getBody()->getTerminator());
  Value predicate = mapping.lookupOrDefault(yieldOp.getValue());
  Value combined = rewriter.create<CombineOp>(loc, loop.getRegionIterArg(0), predicate);
  rewriter.create<scf::YieldOp>(loc, combined);

  rewriter.replaceOp(op, loop.getResults());
  return success();
}

/// Rewrite `bool.forall` over an array-of-POD to iterate over the split leaf arrays directly.
class SplitPodArrayForAllOp : public OpConversionPattern<boolean::ForAllOp> {
public:
  using OpConversionPattern<boolean::ForAllOp>::OpConversionPattern;

  static bool legal(boolean::ForAllOp op) { return !splittablePodArray(op.getSort().getType()); }

  LogicalResult matchAndRewrite(
      boolean::ForAllOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (legal(op)) {
      return failure();
    }
    return rewriteSplitPodArrayQuantifier<boolean::ForAllOp, boolean::AndBoolOp>(
        op, adaptor.getSort(), rewriter, /*initialValue=*/true
    );
  }
};

/// Rewrite `bool.exists` over an array-of-POD to iterate over the split leaf arrays directly.
class SplitPodArrayExistsOp : public OpConversionPattern<boolean::ExistsOp> {
public:
  using OpConversionPattern<boolean::ExistsOp>::OpConversionPattern;

  static bool legal(boolean::ExistsOp op) { return !splittablePodArray(op.getSort().getType()); }

  LogicalResult matchAndRewrite(
      boolean::ExistsOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (legal(op)) {
      return failure();
    }
    return rewriteSplitPodArrayQuantifier<boolean::ExistsOp, boolean::OrBoolOp>(
        op, adaptor.getSort(), rewriter, /*initialValue=*/false
    );
  }
};

/// Rewrite `array.extract` of an array-of-POD subarray into one extract per parallel leaf array.
class SplitPodArrayExtractArrayOp : public OpConversionPattern<ExtractArrayOp> {
public:
  using OpConversionPattern<ExtractArrayOp>::OpConversionPattern;

  static bool legal(ExtractArrayOp op) { return !splittablePodArray(op.getResult().getType()); }

  LogicalResult matchAndRewrite(
      ExtractArrayOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (legal(op)) {
      return failure();
    }

    SmallVector<Type> splitResultTypes;
    splitPodArrayTypeTo(op.getResult().getType(), splitResultTypes);

    SmallVector<Value> indices = flattenConvertedValues(adaptor.getIndices());
    SmallVector<Value> replacements;
    replacements.reserve(splitResultTypes.size());
    for (auto [splitArrRange, splitResultType] :
         llvm::zip_equal(adaptor.getArrRef(), splitResultTypes)) {
      replacements.push_back(rewriter.create<ExtractArrayOp>(
          op.getLoc(), llvm::cast<ArrayType>(splitResultType),
          getSingleConvertedValue(splitArrRange), indices
      ));
    }

    rewriter.replaceOpWithMultiple(op, {ValueRange(replacements)});
    return success();
  }
};

/// Rewrite `array.insert` of an array-of-POD subarray into one insert per parallel leaf array.
class SplitPodArrayInsertArrayOp : public OpConversionPattern<InsertArrayOp> {
public:
  using OpConversionPattern<InsertArrayOp>::OpConversionPattern;

  static bool legal(InsertArrayOp op) { return !splittablePodArray(op.getRvalue().getType()); }

  LogicalResult matchAndRewrite(
      InsertArrayOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (legal(op)) {
      return failure();
    }

    SmallVector<Value> indices = flattenConvertedValues(adaptor.getIndices());
    for (auto [splitArrRange, splitRvalueRange] :
         llvm::zip_equal(adaptor.getArrRef(), adaptor.getRvalue())) {
      rewriter.create<InsertArrayOp>(
          op.getLoc(), getSingleConvertedValue(splitArrRange), indices,
          getSingleConvertedValue(splitRvalueRange)
      );
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Rewrite a write to a split array-of-POD struct member into writes to each parallel array member.
class SplitPodArrayInMemberWriteOp : public OpConversionPattern<MemberWriteOp> {
  SymbolTableCollection &tables;
  const MemberReplacementMap &repMapRef;

public:
  SplitPodArrayInMemberWriteOp(
      const TypeConverter &converter, MLIRContext *ctx, SymbolTableCollection &symTables,
      const MemberReplacementMap &memberRepMap
  )
      : OpConversionPattern<MemberWriteOp>(converter, ctx), tables(symTables),
        repMapRef(memberRepMap) {}

  static bool legal(MemberWriteOp op) { return !splittablePodArray(op.getVal().getType()); }

  LogicalResult matchAndRewrite(
      MemberWriteOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (legal(op)) {
      return failure();
    }
    StructType tgtStructTy = llvm::cast<MemberRefOpInterface>(op.getOperation()).getStructType();
    auto tgtStructDef = tgtStructTy.getDefinition(tables, op);
    assert(succeeded(tgtStructDef));

    const LocalMemberReplacementMap &idToMember =
        repMapRef.at(tgtStructDef->get()).at(op.getMemberNameAttr().getAttr());
    ArrayType arrTy = llvm::cast<ArrayType>(op.getVal().getType());
    SmallVector<RecordChain> splitIds;
    SmallVector<Type> splitTypes;
    splitPodArrayTypeTo(arrTy, splitTypes, &splitIds);

    for (auto [id, splitValRange] : llvm::zip_equal(splitIds, adaptor.getVal())) {
      const MemberInfo &newMember = idToMember.at(id);
      rewriter.create<MemberWriteOp>(
          op.getLoc(), getSingleConvertedValue(adaptor.getComponent()),
          FlatSymbolRefAttr::get(newMember.first), getSingleConvertedValue(splitValRange)
      );
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// Rewrite a read from a split array-of-POD struct member into reads of each parallel array member.
class SplitPodArrayInMemberReadOp : public OpConversionPattern<MemberReadOp> {
  SymbolTableCollection &tables;
  const MemberReplacementMap &repMapRef;

public:
  SplitPodArrayInMemberReadOp(
      const TypeConverter &converter, MLIRContext *ctx, SymbolTableCollection &symTables,
      const MemberReplacementMap &memberRepMap
  )
      : OpConversionPattern<MemberReadOp>(converter, ctx), tables(symTables),
        repMapRef(memberRepMap) {}

  static bool legal(MemberReadOp op) { return !splittablePodArray(op.getResult().getType()); }

  LogicalResult matchAndRewrite(
      MemberReadOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (legal(op)) {
      return failure();
    }
    StructType tgtStructTy = llvm::cast<MemberRefOpInterface>(op.getOperation()).getStructType();
    auto tgtStructDef = tgtStructTy.getDefinition(tables, op);
    assert(succeeded(tgtStructDef));

    const LocalMemberReplacementMap &idToMember =
        repMapRef.at(tgtStructDef->get()).at(op.getMemberNameAttr().getAttr());
    ArrayType arrTy = llvm::cast<ArrayType>(op.getType());
    SmallVector<RecordChain> splitIds;
    SmallVector<Type> splitTypes;
    splitPodArrayTypeTo(arrTy, splitTypes, &splitIds);

    SmallVector<Value> replacements;
    replacements.reserve(splitIds.size());
    for (auto [id, splitType] : llvm::zip_equal(splitIds, splitTypes)) {
      const MemberInfo &newMember = idToMember.at(id);
      replacements.push_back(rewriter.create<MemberReadOp>(
          op.getLoc(), splitType, getSingleConvertedValue(adaptor.getComponent()), newMember.first
      ));
    }
    rewriter.replaceOpWithMultiple(op, {ValueRange(replacements)});
    return success();
  }
};

/// Split arrays-of-POD into parallel arrays before direct pod scalarization.
static LogicalResult
step2(ModuleOp modOp, SymbolTableCollection &symTables, const MemberReplacementMap &memberRepMap) {
  MLIRContext *ctx = modOp.getContext();
  PodArrayTypeConverter typeConverter;

  RewritePatternSet patterns(ctx);
  patterns.add<
      SplitPodArrayNonDetOp, SplitPodArrayCreateArrayOp, SplitPodArrayReadArrayOp,
      SplitPodArrayWriteArrayOp, SplitPodArrayExtractArrayOp, SplitPodArrayInsertArrayOp,
      SplitPodArrayInFuncDefOp, SplitPodArrayInUnifiableCastOp, SplitPodArrayInReturnOp,
      SplitPodArrayInCallOp, SplitPodArrayInEmitEqualityOp, SplitPodArrayInEmitContainmentOp,
      SplitPodArrayLengthOp, SplitPodArrayForAllOp, SplitPodArrayExistsOp>(typeConverter, ctx);
  patterns.add<SplitPodArrayInMemberWriteOp, SplitPodArrayInMemberReadOp>(
      typeConverter, ctx, symTables, memberRepMap
  );

  ConversionTarget target(*ctx);
  baseTargetSetup(target);
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addDynamicallyLegalOp<NonDetOp>(SplitPodArrayNonDetOp::legal);
  target.addDynamicallyLegalOp<CreateArrayOp>(SplitPodArrayCreateArrayOp::legal);
  target.addDynamicallyLegalOp<ReadArrayOp>(SplitPodArrayReadArrayOp::legal);
  target.addDynamicallyLegalOp<WriteArrayOp>(SplitPodArrayWriteArrayOp::legal);
  target.addDynamicallyLegalOp<ExtractArrayOp>(SplitPodArrayExtractArrayOp::legal);
  target.addDynamicallyLegalOp<InsertArrayOp>(SplitPodArrayInsertArrayOp::legal);
  target.addDynamicallyLegalOp<FuncDefOp>(SplitPodArrayInFuncDefOp::legal);
  target.addDynamicallyLegalOp<UnifiableCastOp>(SplitPodArrayInUnifiableCastOp::legal);
  target.addDynamicallyLegalOp<ReturnOp>(SplitPodArrayInReturnOp::legal);
  target.addDynamicallyLegalOp<CallOp>(SplitPodArrayInCallOp::legal);
  target.addDynamicallyLegalOp<constrain::EmitEqualityOp>(SplitPodArrayInEmitEqualityOp::legal);
  target.addDynamicallyLegalOp<constrain::EmitContainmentOp>(
      SplitPodArrayInEmitContainmentOp::legal
  );
  target.addDynamicallyLegalOp<ArrayLengthOp>(SplitPodArrayLengthOp::legal);
  target.addDynamicallyLegalOp<boolean::ForAllOp>(SplitPodArrayForAllOp::legal);
  target.addDynamicallyLegalOp<boolean::ExistsOp>(SplitPodArrayExistsOp::legal);
  target.addDynamicallyLegalOp<MemberWriteOp>(SplitPodArrayInMemberWriteOp::legal);
  target.addDynamicallyLegalOp<MemberReadOp>(SplitPodArrayInMemberReadOp::legal);

  mlir::scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns, target);

  LLVM_DEBUG(llvm::dbgs() << "Begin step 2: split arrays with POD element type\n";);
  return applyFullConversion(modOp, target, std::move(patterns));
}

/// Split inline `pod.new` initializers into explicit `pod.write` operations.
class SplitInitFromNewPodOp : public OpConversionPattern<NewPodOp> {
public:
  using OpConversionPattern<NewPodOp>::OpConversionPattern;

  static bool legal(NewPodOp op) { return op.getInitialValues().empty(); }

  LogicalResult match(NewPodOp op) const override { return failure(legal(op)); }

  void rewrite(NewPodOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    // Generate an individual write for each initialization
    rewriter.setInsertionPointAfter(op);
    Location loc = op.getLoc();
    for (auto [name, init] :
         llvm::zip_equal(adaptor.getInitializedRecords(), adaptor.getInitialValues())) {
      // Create the write
      rewriter.create<WritePodOp>(loc, op.getResult(), llvm::cast<StringAttr>(name), init);
    }
    // Remove initializations from `op`
    rewriter.modifyOpInPlace(op, [&op]() {
      op.getInitialValuesMutable().clear();
      op.setInitializedRecordsAttr(ArrayAttr::get(op.getContext(), {})); // DefaultValuedAttr:{}
    });
  }
};

/// Rewrite `array.new` when explicit elements are PODs or flattened leaf arrays.
///
/// This occurs after the array-of-POD stage has already converted the result type away from
/// `!array.type<... x !pod.type<...>>`, but before the POD operands themselves have been fully
/// scalarized. Rebuild the destination array explicitly so leaf arrays become subarray inserts
/// rather than invalid inline operands to the flattened `array.new`.
class SplitPodElementCreateArrayOp : public OpConversionPattern<CreateArrayOp> {
  const VirtualPodValueMap &virtualPods;

public:
  SplitPodElementCreateArrayOp(MLIRContext *ctx, const VirtualPodValueMap &virtualPodMap)
      : OpConversionPattern<CreateArrayOp>(ctx), virtualPods(virtualPodMap) {}

  static bool legal(CreateArrayOp op) {
    return !llvm::any_of(op.getElements().getTypes(), [](Type type) {
      return splittablePod(type) || llvm::isa<ArrayType>(type);
    });
  }

  LogicalResult match(CreateArrayOp op) const override { return failure(legal(op)); }

  void
  rewrite(CreateArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> leafElements;
    leafElements.reserve(adaptor.getElements().size());

    Type leafType;
    for (Value element : adaptor.getElements()) {
      SmallVector<Value> flattenedValues;
      if (splittablePod(element.getType())) {
        processInputOperand(
            op.getLoc(), element, flattenedValues, rewriter, op.getOperation(), &virtualPods
        );
      } else {
        flattenedValues.push_back(element);
      }

      assert(
          flattenedValues.size() == 1 &&
          "array.new elements should already have been split to a single flattened leaf"
      );
      if (!leafType) {
        leafType = flattenedValues.front().getType();
      } else {
        assert(
            leafType == flattenedValues.front().getType() && "array.new elements must stay uniform"
        );
      }
      leafElements.push_back(flattenedValues.front());
    }

    size_t leafRank = 0;
    if (auto leafArrTy = llvm::dyn_cast_if_present<ArrayType>(leafType)) {
      leafRank = leafArrTy.getDimensionSizes().size();
    }
    ArrayType arrTy = op.getType();
    assert(
        arrTy.getDimensionSizes().size() >= leafRank && "flattened leaf rank exceeds array rank"
    );
    size_t outerRank = arrTy.getDimensionSizes().size() - leafRank;
    assert(outerRank > 0 && "array.new elements must populate at least one outer array dimension");

    ArrayType outerIndexTy =
        ArrayType::get(arrTy.getElementType(), arrTy.getDimensionSizes().take_front(outerRank));
    auto elementIndices = outerIndexTy.getSubelementIndices();
    assert(
        elementIndices && "array.new with explicit POD elements requires static outer dimensions"
    );
    assert(
        elementIndices->size() == leafElements.size() &&
        "array.new element count must match the outer array cardinality"
    );

    Value rebuiltArray = createWritableArrayValue(rewriter, op.getLoc(), arrTy);
    for (auto [index, leafValue] : llvm::zip_equal(*elementIndices, leafElements)) {
      genArrayWrite(rewriter, op.getLoc(), rebuiltArray, index, leafValue);
    }
    rewriter.replaceOp(op, rebuiltArray);
  }
};

/// Rewrite pod-typed function signatures to pass one scalar per POD record instead.
///
/// Each pod argument is expanded into one scalar argument per record, and each pod result is
/// expanded into one scalar result per record. Inside the rewritten function body, the original
/// POD-typed block arguments are reconstructed locally with `pod.new` plus `pod.write` so the
/// rest of the function can continue to use POD values until later cleanup passes scalarize those
/// local temporaries away.
class SplitPodInFuncDefOp : public OpConversionPattern<FuncDefOp> {
  VirtualPodValueMap &virtualPods;

public:
  SplitPodInFuncDefOp(MLIRContext *ctx, VirtualPodValueMap &virtualPodMap)
      : OpConversionPattern<FuncDefOp>(ctx), virtualPods(virtualPodMap) {}

  inline static bool legal(FuncDefOp op) {
    return !containsSplittablePodType(op.getArgumentTypes()) &&
           !containsSplittablePodType(op.getResultTypes());
  }

  LogicalResult match(FuncDefOp op) const override { return failure(legal(op)); }

  void rewrite(FuncDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    // Update in/out types of the function to replace pods with scalars
    class Impl : public FunctionTypeConverter {
      SmallVector<size_t> originalInputIdxToSize, originalResultIdxToSize;
      SplitFunctionNameInfo inputNameInfo;
      SplitFunctionNameInfo resultNameInfo;
      VirtualPodValueMap &virtualPods;

    protected:
      SmallVector<Type> convertInputs(ArrayRef<Type> origTypes) override {
        return splitPodType(origTypes, &originalInputIdxToSize);
      }
      SmallVector<Type> convertResults(ArrayRef<Type> origTypes) override {
        return splitPodType(origTypes, &originalResultIdxToSize);
      }
      ArrayAttr convertInputAttrs(ArrayAttr origAttrs, SmallVector<Type> newTypes) override {
        return replicateFunctionNameAttrsAsNeeded(
            origAttrs, originalInputIdxToSize, newTypes, ARG_NAME_ATTR_NAME,
            inputNameInfo.originalNames, inputNameInfo.existingNames,
            inputNameInfo.splitNameSuffixes
        );
      }
      ArrayAttr convertResultAttrs(ArrayAttr origAttrs, SmallVector<Type> newTypes) override {
        return replicateFunctionNameAttrsAsNeeded(
            origAttrs, originalResultIdxToSize, newTypes, RES_NAME_ATTR_NAME,
            resultNameInfo.originalNames, resultNameInfo.existingNames,
            resultNameInfo.splitNameSuffixes
        );
      }

      /// For each argument to the Block that has a splittable PodType, replace it with the
      /// necessary number of scalar arguments, generate a NewPodOp, and generate writes from
      /// the new block scalar arguments to the new pod. All users of the original block
      /// argument are updated to target the result of the NewPodOp.
      void processBlockArgs(Block &entryBlock, RewriterBase &rewriter) override {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&entryBlock);

        for (unsigned i = 0; i < entryBlock.getNumArguments();) {
          Value oldV = entryBlock.getArgument(i);
          if (PodType pt = splittablePod(oldV.getType())) {
            Location loc = oldV.getLoc();
            VirtualPodLeafMap leafValues;
            SmallVector<StringAttr> recordChain;
            unsigned nextArgIdx = i + 1;
            forEachPodLeaf(pt, recordChain, [&](const RecordChain &id, Type leafType) {
              BlockArgument newArg = entryBlock.insertArgument(nextArgIdx, leafType, loc);
              leafValues[id] = newArg;
              ++nextArgIdx;
            });

            Value virtualPod = createVirtualPodPlaceholder(rewriter, loc, pt, leafValues);
            rewriter.replaceAllUsesWith(oldV, virtualPod);
            entryBlock.eraseArgument(i);

            i += leafValues.size();
            virtualPods[virtualPod] = std::move(leafValues);
          } else {
            ++i;
          }
        }
      }

    public:
      Impl(FuncDefOp op, VirtualPodValueMap &virtualPodMap) : virtualPods(virtualPodMap) {
        inputNameInfo = collectSplitFunctionNameInfo(op.getArgumentTypes(), [&op](unsigned i) {
          return op.getArgNameAttr(i);
        }, getSplitRecordNameSuffixes);
        resultNameInfo = collectSplitFunctionNameInfo(
            op.getResultTypes(), [resultAttrs = op.getAllResultAttrs()](unsigned i) {
          return getAttrAtIndexWithName(resultAttrs, i, RES_NAME_ATTR_NAME);
        }, getSplitRecordNameSuffixes
        );
      }
    };
    Impl(op, virtualPods).convert(op, rewriter);
  }
};

/// Rewrite `function.return` to flatten any POD operands into their scalar record values.
///
/// This mirrors the function-signature conversion performed by `SplitPodInFuncDefOp`: POD results
/// are returned as one SSA value per record, using local `pod.read` operations to extract the
/// scalar pieces immediately before the return.
class SplitPodInReturnOp : public OpConversionPattern<ReturnOp> {
  const VirtualPodValueMap &virtualPods;

public:
  SplitPodInReturnOp(MLIRContext *ctx, const VirtualPodValueMap &virtualPodMap)
      : OpConversionPattern<ReturnOp>(ctx), virtualPods(virtualPodMap) {}

  inline static bool legal(ReturnOp op) {
    return !containsSplittablePodType(op.getOperands().getTypes());
  }

  LogicalResult match(ReturnOp op) const override { return failure(legal(op)); }

  void rewrite(ReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    processInputOperands(
        adaptor.getOperands(), op.getOperandsMutable(), op, rewriter, &virtualPods
    );
  }
};

/// Rebuild a call with split scalar results, then reconstruct POD-typed results locally.
static CallOp newCallOpWithSplitResults(
    CallOp oldCall, CallOp::Adaptor adaptor, ConversionPatternRewriter &rewriter,
    VirtualPodValueMap &virtualPods
) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(oldCall);

  Operation::result_range oldResults = oldCall.getResults();
  CallOp newCall = createCallPreservingInstantiationOperands(
      oldCall.getLoc(), splitPodType(oldResults.getTypes()), oldCall, adaptor.getMapOperands(),
      adaptor.getArgOperands(), rewriter
  );

  auto newResults = newCall.getResults().begin();
  for (Value oldVal : oldResults) {
    if (PodType pt = splittablePod(oldVal.getType())) {
      Location loc = oldVal.getLoc();
      VirtualPodLeafMap leafValues;
      SmallVector<StringAttr> recordChain;
      forEachPodLeaf(pt, recordChain, [&leafValues, &newResults](const RecordChain &id, Type) {
        leafValues[id] = *newResults;
        ++newResults;
      });
      Value virtualPod = createVirtualPodPlaceholder(rewriter, loc, pt, leafValues);
      virtualPods[virtualPod] = std::move(leafValues);
      rewriter.replaceAllUsesWith(oldVal, virtualPod);
    } else {
      rewriter.replaceAllUsesWith(oldVal, *newResults);
      newResults++;
    }
  }
  // erase the original CallOp
  rewriter.eraseOp(oldCall);

  return newCall;
}

/// Rewrite calls whose arguments or results contain PODs to use flattened scalar signatures.
///
/// POD arguments are decomposed into scalar record operands before the new call is formed. POD
/// results are reconstructed locally after the call with `pod.new` plus `pod.write`, preserving
/// the original POD-typed uses in the caller until later optimization passes remove the temporary
/// POD allocations.
class SplitPodInCallOp : public OpConversionPattern<CallOp> {
  VirtualPodValueMap &virtualPods;

public:
  SplitPodInCallOp(MLIRContext *ctx, VirtualPodValueMap &virtualPodMap)
      : OpConversionPattern<CallOp>(ctx), virtualPods(virtualPodMap) {}

  inline static bool legal(CallOp op) {
    return !containsSplittablePodType(op.getArgOperands().getTypes()) &&
           !containsSplittablePodType(op.getResultTypes());
  }

  LogicalResult match(CallOp op) const override { return failure(legal(op)); }

  void rewrite(CallOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    // Create new CallOp with split results first so, then process its inputs to split types
    CallOp newCall = newCallOpWithSplitResults(op, adaptor, rewriter, virtualPods);
    processInputOperands(
        newCall.getArgOperands(), newCall.getArgOperandsMutable(), newCall, rewriter, &virtualPods
    );
  }
};

/// Rewrite a write to a pod-typed struct member into writes to the corresponding scalar leaves.
class SplitPodInMemberWriteOp : public OpConversionPattern<MemberWriteOp> {
  SymbolTableCollection &tables;
  const MemberReplacementMap &repMapRef;
  const VirtualPodValueMap &virtualPods;

public:
  SplitPodInMemberWriteOp(
      MLIRContext *ctx, SymbolTableCollection &symTables, const MemberReplacementMap &memberRepMap,
      const VirtualPodValueMap &virtualPodMap
  )
      : OpConversionPattern<MemberWriteOp>(ctx), tables(symTables), repMapRef(memberRepMap),
        virtualPods(virtualPodMap) {}

  static bool legal(MemberWriteOp op) { return !containsSplittablePodType(op.getVal().getType()); }

  LogicalResult match(MemberWriteOp op) const override { return failure(legal(op)); }

  void
  rewrite(MemberWriteOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    StructType tgtStructTy = llvm::cast<MemberRefOpInterface>(op.getOperation()).getStructType();
    auto tgtStructDef = tgtStructTy.getDefinition(tables, op);
    assert(succeeded(tgtStructDef));

    const LocalMemberReplacementMap &idToMember =
        repMapRef.at(tgtStructDef->get()).at(op.getMemberNameAttr().getAttr());
    const VirtualPodLeafMap *virtualLeafValues =
        !hasEarlierWriteToPodInBlock(op.getOperation(), adaptor.getVal())
            ? lookupVirtualPodLeafMap(adaptor.getVal(), virtualPods)
            : nullptr;

    for (const auto &[id, newMember] : idToMember) {
      Value scalarValue = virtualLeafValues
                              ? virtualLeafValues->at(id)
                              : genReadAlongPath(rewriter, op.getLoc(), adaptor.getVal(), id);
      rewriter.create<MemberWriteOp>(
          op.getLoc(), adaptor.getComponent(), FlatSymbolRefAttr::get(newMember.first), scalarValue
      );
    }
    rewriter.eraseOp(op);
  }
};

/// Rewrite a read from a pod-typed struct member into reads from the corresponding scalar leaves.
class SplitPodInMemberReadOp : public OpConversionPattern<MemberReadOp> {
  SymbolTableCollection &tables;
  const MemberReplacementMap &repMapRef;
  VirtualPodValueMap &virtualPods;

public:
  SplitPodInMemberReadOp(
      MLIRContext *ctx, SymbolTableCollection &symTables, const MemberReplacementMap &memberRepMap,
      VirtualPodValueMap &virtualPodMap
  )
      : OpConversionPattern<MemberReadOp>(ctx), tables(symTables), repMapRef(memberRepMap),
        virtualPods(virtualPodMap) {}

  static bool legal(MemberReadOp op) {
    return !containsSplittablePodType(op.getResult().getType());
  }

  LogicalResult match(MemberReadOp op) const override { return failure(legal(op)); }

  void
  rewrite(MemberReadOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    StructType tgtStructTy = llvm::cast<MemberRefOpInterface>(op.getOperation()).getStructType();
    auto tgtStructDef = tgtStructTy.getDefinition(tables, op);
    assert(succeeded(tgtStructDef));

    const LocalMemberReplacementMap &idToMember =
        repMapRef.at(tgtStructDef->get()).at(op.getMemberNameAttr().getAttr());

    VirtualPodLeafMap leafValues;
    for (const auto &[id, newMember] : idToMember) {
      leafValues[id] = rewriter.create<MemberReadOp>(
          op.getLoc(), newMember.second, adaptor.getComponent(), newMember.first
      );
    }

    PodType podTy = llvm::cast<PodType>(op.getType());
    Value virtualPod = createVirtualPodPlaceholder(rewriter, op.getLoc(), podTy, leafValues);
    virtualPods[virtualPod] = std::move(leafValues);
    rewriter.replaceOp(op, virtualPod);
  }
};

/// Collect precise split leaf arrays from a value re-materialized as an aggregate array-of-POD.
///
/// This recognizes the temporary aggregate form produced by dialect conversion casts and unwraps
/// it back into the parallel split arrays expected by the late pod-array read resolvers.
static bool tryCollectMaterializedSplitPodArrayLeafValues(
    Value arrayValue, ArrayType arrTy, ArrayRef<Type> splitTypes, SmallVectorImpl<Value> &leafArrays
) {
  auto cast = arrayValue.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast->getNumResults() != 1 || cast.getResult(0).getType() != arrTy ||
      cast->getNumOperands() != splitTypes.size()) {
    return false;
  }

  for (auto [operand, splitType] : llvm::zip_equal(cast.getOperands(), splitTypes)) {
    if (operand.getType() != splitType) {
      return false;
    }
    leafArrays.push_back(operand);
  }
  return true;
}

/// Collect precise split leaf arrays for an array-of-POD value backed by a direct `pod.read`.
///
/// This first consults virtual POD leaf storage and, if unavailable, falls back to forwarding
/// through a dominating same-record `pod.write` whose value was previously materialized as split
/// arrays.
static bool tryCollectReadPodSplitPodArrayLeafValues(
    ReadPodOp readOp, ArrayType arrTy, ArrayRef<RecordChain> splitIds, ArrayRef<Type> splitTypes,
    const VirtualPodValueMap &virtualPods, SmallVectorImpl<Value> &leafArrays
) {
  if (WritePodOp writeOp = findNearestForwardableWriteInBlock(readOp)) {
    return tryCollectMaterializedSplitPodArrayLeafValues(
        writeOp.getValue(), arrTy, splitTypes, leafArrays
    );
  }

  if (!hasEarlierWriteInBlock(readOp)) {
    if (const VirtualPodLeafMap *podLeafValues =
            lookupVirtualPodLeafMap(readOp.getPodRef(), virtualPods)) {
      leafArrays.reserve(splitIds.size());
      for (const RecordChain &id : splitIds) {
        SmallVector<StringAttr> fullChain {readOp.getRecordNameAttr()};
        llvm::append_range(fullChain, id.nameList);
        auto it = podLeafValues->find(RecordChain(fullChain));
        if (it == podLeafValues->end() ||
            !typesUnify(it->second.getType(), getFlattenedTypeAlongPath(arrTy, id.nameList))) {
          return false;
        }
        leafArrays.push_back(it->second);
      }
      return true;
    }
  }

  return false;
}

/// Materialize or recover split leaf arrays for a dynamic array-of-POD produced by `pod.read`.
static bool resolveReadPodSplitPodArrayLeafValues(
    ReadPodOp readOp, ArrayType arrTy, ArrayRef<RecordChain> splitIds, ArrayRef<Type> splitTypes,
    const VirtualPodValueMap &virtualPods, DeferredPodArrayLeafMap &deferredPodArrays, Location loc,
    OpBuilder &bldr, SmallVectorImpl<Value> &leafArrays
) {
  if (tryCollectReadPodSplitPodArrayLeafValues(
          readOp, arrTy, splitIds, splitTypes, virtualPods, leafArrays
      )) {
    return true;
  }

  if (!isFreshUnwrittenPodRead(readOp)) {
    return false;
  }

  // Reuse one synthetic split-array backing per deferred field read so repeated users of the same
  // aggregate value continue to observe the same unwritten leaf storage.
  auto [it, inserted] = deferredPodArrays.try_emplace(readOp.getResult());
  leafArrays.assign(it->second.begin(), it->second.end());
  if (inserted) {
    OpBuilder::InsertionGuard guard(bldr);
    bldr.setInsertionPointAfter(readOp);
    leafArrays.reserve(splitTypes.size());
    for (Type splitType : splitTypes) {
      leafArrays.push_back(createWritableArrayValue(bldr, loc, llvm::cast<ArrayType>(splitType)));
    }
    it->second.assign(leafArrays.begin(), leafArrays.end());
  } else {
    assert(
        leafArrays.size() == splitTypes.size() &&
        "cached split POD arrays must match the rewritten read arity"
    );
  }

  return true;
}

/// Erase a resolved deferred field-read chain once both the read and its placeholder pod vanish.
static void eraseDeadDeferredFieldReadChain(ReadPodOp readOp, PatternRewriter &rewriter) {
  if (!readOp.getResult().use_empty()) {
    return;
  }

  Value podRef = readOp.getPodRef();
  rewriter.eraseOp(readOp);
  if (podRef.use_empty()) {
    if (auto cast = podRef.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (cast->getNumResults() == 1 && cast.getResult(0) == podRef) {
        rewriter.eraseOp(cast);
      }
    }
  }
}

/// Return `true` iff `op` is a deferred split placeholder for one array-of-POD aggregate value.
static bool getDeferredSplitPodArrayCastInfo(
    UnrealizedConversionCastOp op, ArrayType &arrTy, SmallVector<RecordChain> &splitIds,
    SmallVectorImpl<Type> &splitTypes
) {
  if (op->getNumOperands() != 1) {
    return false;
  }

  arrTy = splittablePodArray(op.getOperand(0).getType());
  if (!arrTy) {
    return false;
  }

  splitPodArrayTypeTo(arrTy, splitTypes, &splitIds);
  if (op->getNumResults() != splitTypes.size()) {
    return false;
  }

  for (auto [result, splitType] : llvm::zip_equal(op.getResults(), splitTypes)) {
    if (result.getType() != splitType) {
      return false;
    }
  }
  return true;
}

/// Resolve deferred `array.read` from `pod.read`-produced array-of-POD values.
///
/// When step 2 defers a read because the array-of-POD came from a POD record, this pattern
/// reconstructs the per-leaf split arrays, performs the array read on each leaf array, and then
/// rebuilds the element POD virtually instead of materializing the whole aggregate array first.
class ResolvePodReadBackedArrayReadOp : public OpConversionPattern<ReadArrayOp> {
  VirtualPodValueMap &virtualPods;
  DeferredPodArrayLeafMap &deferredPodArrays;

public:
  ResolvePodReadBackedArrayReadOp(
      MLIRContext *ctx, VirtualPodValueMap &virtualPodMap,
      DeferredPodArrayLeafMap &deferredPodArrayMap
  )
      : OpConversionPattern<ReadArrayOp>(ctx), virtualPods(virtualPodMap),
        deferredPodArrays(deferredPodArrayMap) {}

  static bool canResolve(ReadArrayOp op, const VirtualPodValueMap &virtualPods) {
    if (!shouldDeferPodArrayReadToStep3(op)) {
      return false;
    }

    ArrayType arrTy = op.getArrRefType();
    auto fieldRead = llvm::cast<ReadPodOp>(op.getArrRef().getDefiningOp());
    SmallVector<RecordChain> splitIds;
    SmallVector<Type> splitTypes;
    splitPodArrayTypeTo(arrTy, splitTypes, &splitIds);

    SmallVector<Value> ignoredLeafArrays;
    return tryCollectReadPodSplitPodArrayLeafValues(
               fieldRead, arrTy, splitIds, splitTypes, virtualPods, ignoredLeafArrays
           ) ||
           isFreshUnwrittenPodRead(fieldRead);
  }

  LogicalResult matchAndRewrite(
      ReadArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    auto fieldRead = op.getArrRef().getDefiningOp<ReadPodOp>();
    if (!fieldRead) {
      return failure();
    }

    ArrayType arrTy = op.getArrRefType();
    PodType podTy = llvm::cast<PodType>(arrTy.getElementType());
    SmallVector<RecordChain> splitIds;
    SmallVector<Type> splitTypes;
    splitPodArrayTypeTo(arrTy, splitTypes, &splitIds);

    SmallVector<Value> splitLeafArrays;
    if (!resolveReadPodSplitPodArrayLeafValues(
            fieldRead, arrTy, splitIds, splitTypes, virtualPods, deferredPodArrays, op.getLoc(),
            rewriter, splitLeafArrays
        )) {
      return failure();
    }

    SmallVector<Value> indices(adaptor.getIndices().begin(), adaptor.getIndices().end());
    VirtualPodLeafMap leafValues;
    for (auto [id, leafArray] : llvm::zip_equal(splitIds, splitLeafArrays)) {
      leafValues[id] = genArrayRead(rewriter, op.getLoc(), leafArray, indices);
    }

    Value virtualPod = createVirtualPodPlaceholder(rewriter, op.getLoc(), podTy, leafValues);
    virtualPods[virtualPod] = std::move(leafValues);
    rewriter.replaceOp(op, virtualPod);
    eraseDeadDeferredFieldReadChain(fieldRead, rewriter);
    return success();
  }
};

/// Resolve deferred split-array placeholders created while flattening direct POD operands.
///
/// Step 2 may need one specific split leaf array from a dynamic array-of-POD field before step 3
/// has converted the surrounding POD value into virtual leaf storage. In that case
/// `genReadAlongPath` leaves behind a `builtin.unrealized_conversion_cast` from the aggregate field
/// read to all split leaf arrays, and this pattern resolves that placeholder once the backing leaf
/// arrays become available.
class ResolveDeferredSplitPodArrayCastOp : public OpConversionPattern<UnrealizedConversionCastOp> {
  VirtualPodValueMap &virtualPods;
  DeferredPodArrayLeafMap &deferredPodArrays;

public:
  ResolveDeferredSplitPodArrayCastOp(
      MLIRContext *ctx, VirtualPodValueMap &virtualPodMap,
      DeferredPodArrayLeafMap &deferredPodArrayMap
  )
      : OpConversionPattern<UnrealizedConversionCastOp>(ctx), virtualPods(virtualPodMap),
        deferredPodArrays(deferredPodArrayMap) {}

  static bool canResolve(UnrealizedConversionCastOp op, const VirtualPodValueMap &virtualPods) {
    ArrayType arrTy;
    SmallVector<RecordChain> splitIds;
    SmallVector<Type> splitTypes;
    if (!getDeferredSplitPodArrayCastInfo(op, arrTy, splitIds, splitTypes)) {
      return false;
    }

    ReadPodOp fieldRead = peelUnifiableCasts(op.getOperand(0)).getDefiningOp<ReadPodOp>();
    if (!fieldRead) {
      return false;
    }

    SmallVector<Value> ignoredLeafArrays;
    return tryCollectReadPodSplitPodArrayLeafValues(
               fieldRead, arrTy, splitIds, splitTypes, virtualPods, ignoredLeafArrays
           ) ||
           isFreshUnwrittenPodRead(fieldRead);
  }

  LogicalResult matchAndRewrite(
      UnrealizedConversionCastOp op, OpAdaptor, ConversionPatternRewriter &rewriter
  ) const override {
    ArrayType arrTy;
    SmallVector<RecordChain> splitIds;
    SmallVector<Type> splitTypes;
    if (!getDeferredSplitPodArrayCastInfo(op, arrTy, splitIds, splitTypes)) {
      return failure();
    }

    ReadPodOp fieldRead = peelUnifiableCasts(op.getOperand(0)).getDefiningOp<ReadPodOp>();
    if (!fieldRead) {
      return failure();
    }

    SmallVector<Value> splitLeafArrays;
    if (!resolveReadPodSplitPodArrayLeafValues(
            fieldRead, arrTy, splitIds, splitTypes, virtualPods, deferredPodArrays, op.getLoc(),
            rewriter, splitLeafArrays
        )) {
      return failure();
    }

    rewriter.replaceOp(op, splitLeafArrays);
    eraseDeadDeferredFieldReadChain(fieldRead, rewriter);
    return success();
  }
};

/// Update virtual POD leaf storage in response to `pod.write` without materializing the aggregate.
class ResolveVirtualPodWriteOp : public OpConversionPattern<WritePodOp> {
  VirtualPodValueMap &virtualPods;

public:
  ResolveVirtualPodWriteOp(MLIRContext *ctx, VirtualPodValueMap &virtualPodMap)
      : OpConversionPattern<WritePodOp>(ctx), virtualPods(virtualPodMap) {}

  LogicalResult matchAndRewrite(
      WritePodOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    auto it = virtualPods.find(adaptor.getPodRef());
    if (it == virtualPods.end()) {
      return failure();
    }

    Type recordType =
        llvm::cast<PodType>(op.getPodRefType()).getRecordMap().lookup(op.getRecordName());
    assert(recordType && "record must exist in POD type");
    updateVirtualPodRecordLeafValues(
        op.getLoc(), op.getRecordNameAttr(), recordType, adaptor.getValue(), virtualPods, rewriter,
        it->second
    );
    rewriter.eraseOp(op);
    return success();
  }
};

/// Resolve reads from a virtual POD placeholder without materializing the whole aggregate.
///
/// This pattern answers `pod.read` directly from virtual leaf storage, rebuilding nested POD
/// subrecords on demand and casting scalar leaves back to the precise record type when needed.
class ResolveVirtualPodReadOp : public OpConversionPattern<ReadPodOp> {
  VirtualPodValueMap &virtualPods;

public:
  ResolveVirtualPodReadOp(MLIRContext *ctx, VirtualPodValueMap &virtualPodMap)
      : OpConversionPattern<ReadPodOp>(ctx), virtualPods(virtualPodMap) {}

  LogicalResult matchAndRewrite(
      ReadPodOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (hasEarlierWriteInBlock(op) || findNearestForwardableWriteInBlock(op)) {
      return failure();
    }

    const VirtualPodLeafMap *leafValues = lookupVirtualPodLeafMap(adaptor.getPodRef(), virtualPods);
    if (!leafValues) {
      return failure();
    }

    SmallVector<StringAttr> prefix {op.getRecordNameAttr()};
    Type recordType =
        llvm::cast<PodType>(op.getPodRefType()).getRecordMap().lookup(op.getRecordName());
    assert(recordType && "record must exist in POD type");

    if (PodType nestedPodTy = llvm::dyn_cast<PodType>(recordType)) {
      VirtualPodLeafMap nestedLeafValues;
      SmallVector<StringAttr> nestedRecordChain;
      forEachPodLeaf(nestedPodTy, nestedRecordChain, [&](RecordChain id, Type) {
        SmallVector<StringAttr> fullChain(prefix);
        llvm::append_range(fullChain, id.nameList);
        nestedLeafValues[id] = leafValues->at(RecordChain(fullChain));
      });
      Value virtualPod =
          createVirtualPodPlaceholder(rewriter, op.getLoc(), nestedPodTy, nestedLeafValues);
      virtualPods[virtualPod] = std::move(nestedLeafValues);
      rewriter.replaceOp(op, virtualPod);
      return success();
    }

    if (splittablePodArray(recordType)) {
      return failure();
    }

    rewriter.replaceOp(
        op, castValueToTypeIfNeeded(
                rewriter, op.getLoc(), leafValues->at(RecordChain(prefix)), recordType
            )
    );
    return success();
  }
};

/// Special handling to split pods in struct member refs and function signatures and desugar
/// initializations on pod.new into pod writes.
static LogicalResult
step3(ModuleOp modOp, SymbolTableCollection &symTables, const MemberReplacementMap &memberRepMap) {
  MLIRContext *ctx = modOp.getContext();
  VirtualPodValueMap virtualPods;
  DeferredPodArrayLeafMap deferredPodArrays;

  RewritePatternSet patterns(ctx);
  patterns.add<SplitInitFromNewPodOp>(ctx);
  patterns.add<SplitPodElementCreateArrayOp>(ctx, virtualPods);
  patterns.add<SplitPodInFuncDefOp, SplitPodInReturnOp, SplitPodInCallOp>(ctx, virtualPods);
  patterns.add<SplitPodInMemberWriteOp, SplitPodInMemberReadOp>(
      ctx, symTables, memberRepMap, virtualPods
  );
  patterns.add<ResolvePodReadBackedArrayReadOp>(ctx, virtualPods, deferredPodArrays);
  patterns.add<ResolveDeferredSplitPodArrayCastOp>(ctx, virtualPods, deferredPodArrays);
  patterns.add<ResolveVirtualPodWriteOp, ResolveVirtualPodReadOp>(ctx, virtualPods);

  ConversionTarget target(*ctx);
  baseTargetSetup(target);
  target.addDynamicallyLegalOp<NewPodOp>(SplitInitFromNewPodOp::legal);
  target.addDynamicallyLegalOp<CreateArrayOp>(SplitPodElementCreateArrayOp::legal);
  target.addDynamicallyLegalOp<FuncDefOp>(SplitPodInFuncDefOp::legal);
  target.addDynamicallyLegalOp<ReturnOp>(SplitPodInReturnOp::legal);
  target.addDynamicallyLegalOp<CallOp>(SplitPodInCallOp::legal);
  target.addDynamicallyLegalOp<MemberWriteOp>(SplitPodInMemberWriteOp::legal);
  target.addDynamicallyLegalOp<MemberReadOp>(SplitPodInMemberReadOp::legal);
  target.addDynamicallyLegalOp<WritePodOp>([&virtualPods](WritePodOp op) {
    return !lookupVirtualPodLeafMap(op.getPodRef(), virtualPods);
  });
  target.addDynamicallyLegalOp<ReadArrayOp>([&virtualPods](ReadArrayOp op) {
    return !ResolvePodReadBackedArrayReadOp::canResolve(op, virtualPods);
  });
  target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
      [&virtualPods](UnrealizedConversionCastOp op) {
    return !ResolveDeferredSplitPodArrayCastOp::canResolve(op, virtualPods);
  }
  );
  target.addDynamicallyLegalOp<ReadPodOp>([&virtualPods](ReadPodOp op) {
    return !canResolveVirtualPodRead(op, virtualPods);
  });

  LLVM_DEBUG(llvm::dbgs() << "Begin step 3: update/split other pod ops\n";);
  if (failed(applyFullConversion(modOp, target, std::move(patterns)))) {
    return failure();
  }

  OpBuilder builder(ctx);
  for (auto &[podValue, leafValues] : virtualPods) {
    if (podValue.use_empty()) {
      continue;
    }
    if (auto newPod = llvm::dyn_cast<NewPodOp>(podValue.getDefiningOp())) {
      builder.setInsertionPointAfter(newPod);
      materializeVirtualPod(builder, newPod, leafValues);
    }
  }

  bool erasedDeadPlaceholderOps = false;
  do {
    SmallVector<Operation *> deadPlaceholderOps;
    modOp->walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (auto readOp = llvm::dyn_cast<ReadPodOp>(op)) {
        if (readOp.getResult().use_empty()) {
          deadPlaceholderOps.push_back(op);
        }
        return;
      }

      if (auto castOp = llvm::dyn_cast<UnrealizedConversionCastOp>(op)) {
        if (llvm::all_of(castOp.getResults(), [](Value result) { return result.use_empty(); })) {
          deadPlaceholderOps.push_back(op);
        }
      }
    });
    for (Operation *op : deadPlaceholderOps) {
      op->erase();
    }
    erasedDeadPlaceholderOps = !deadPlaceholderOps.empty();
  } while (erasedDeadPlaceholderOps);

  SmallVector<Operation *> deadOps;
  modOp->walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (op != modOp.getOperation() && isOpTriviallyDead(op)) {
      deadOps.push_back(op);
    }
  });
  for (Operation *op : deadOps) {
    op->erase();
  }
  return success();
}

/// Return whether `value` is defined within `ancestor` or one of its nested regions.
///
/// Values defined inside a control-flow operation cannot be hoisted across that operation without
/// introducing an explicit region result or loop-carried value.
static bool isValueDefinedInside(Operation *ancestor, Value value) {
  if (Operation *defOp = value.getDefiningOp()) {
    return ancestor->isAncestor(defOp);
  }

  auto blockArg = llvm::dyn_cast<BlockArgument>(value);
  Operation *parentOp = blockArg.getOwner()->getParentOp();
  return parentOp && ancestor->isAncestor(parentOp);
}

/// Find a write before the parent `scf.if` that can be forwarded to `readOp`.
static WritePodOp findPrecedingWriteForIfRead(ReadPodOp readOp) {
  auto ifOp = readOp->getParentOfType<scf::IfOp>();
  if (!ifOp || readOp->getBlock()->getParentOp() != ifOp.getOperation()) {
    return nullptr;
  }
  if (hasEarlierWriteInBlock(readOp)) {
    return nullptr;
  }

  Block *ifBlock = ifOp->getBlock();
  if (!ifBlock) {
    return nullptr;
  }

  Value podRef = readOp.getPodRef();
  StringAttr recordName = readOp.getRecordNameAttr();
  WritePodOp replacement = nullptr;
  for (Operation &op : *ifBlock) {
    if (&op == ifOp.getOperation()) {
      break;
    }

    if (auto writeOp = dyn_cast<WritePodOp>(&op)) {
      if (isSamePodRecord(writeOp, podRef, recordName)) {
        replacement = writeOp;
      }
      continue;
    }

    if (hasNestedWriteToRecord(op, podRef, recordName)) {
      replacement = nullptr;
    }
  }

  return replacement;
}

/// Replace a read with the value from the nearest preceding same-record write in the block.
class FoldReadAfterWriteInBlockPattern final : public OpRewritePattern<ReadPodOp> {
public:
  using OpRewritePattern<ReadPodOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReadPodOp readOp, PatternRewriter &rewriter) const override {
    if (WritePodOp writeOp = findNearestForwardableWriteInBlock(readOp)) {
      rewriter.replaceOp(readOp, writeOp.getValue());
      return success();
    }
    return failure();
  }
};

/// Replace a branch-local read with a value available in the parent block.
class ReplaceIfReadPattern final : public OpRewritePattern<ReadPodOp> {
public:
  using OpRewritePattern<ReadPodOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReadPodOp readOp, PatternRewriter &rewriter) const override {
    auto ifOp = readOp->getParentOfType<scf::IfOp>();
    if (!ifOp || readOp->getBlock()->getParentOp() != ifOp.getOperation()) {
      return failure();
    }
    if (isValueDefinedInside(ifOp, readOp.getPodRef()) || hasEarlierWriteInBlock(readOp)) {
      return failure();
    }

    if (WritePodOp writeOp = findPrecedingWriteForIfRead(readOp)) {
      rewriter.replaceOp(readOp, writeOp.getValue());
      return success();
    }

    rewriter.setInsertionPoint(ifOp);
    rewriter.replaceOp(
        readOp, genRead(rewriter, readOp.getLoc(), readOp.getPodRef(), readOp.getRecordNameAttr())
                    .getResult()
    );
    return success();
  }
};

/// Fold reads from an `scf.if`-carried POD result when the same record was just written from
/// another result of that same `scf.if`.
///
/// Pattern:
///   %if:2 = scf.if ... -> (!pod<...>, T) {
///     scf.yield %pod, %v0
///   } else {
///     scf.yield %pod, %v1
///   }
///   pod.write %pod[@r] = %if#1
///   %x = pod.read %if#0[@r]
///
/// Rewritten to `%x = %if#1` when all yielded values for `%if#0` are the same `%pod`.
class FoldIfCarriedPodReadAfterWritePattern final : public OpRewritePattern<ReadPodOp> {
public:
  using OpRewritePattern<ReadPodOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReadPodOp readOp, PatternRewriter &rewriter) const override {
    auto podRes = dyn_cast<OpResult>(readOp.getPodRef());
    if (!podRes) {
      return failure();
    }

    auto ifOp = dyn_cast<scf::IfOp>(podRes.getOwner());
    if (!ifOp) {
      return failure();
    }

    auto writeOp = dyn_cast_or_null<WritePodOp>(readOp->getPrevNode());
    if (!writeOp || writeOp.getRecordNameAttr() != readOp.getRecordNameAttr()) {
      return failure();
    }

    auto valueRes = dyn_cast<OpResult>(writeOp.getValue());
    if (!valueRes || valueRes.getOwner() != ifOp.getOperation()) {
      return failure();
    }

    Value carriedPod = writeOp.getPodRef();
    unsigned podResultIndex = podRes.getResultNumber();

    auto thenYield = dyn_cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    if (!thenYield || thenYield.getOperand(podResultIndex) != carriedPod) {
      return failure();
    }

    Region &elseRegion = ifOp.getElseRegion();
    if (Block *elseBlock = elseRegion.empty() ? nullptr : &elseRegion.front()) {
      auto elseYield = dyn_cast<scf::YieldOp>(elseBlock->getTerminator());
      if (!elseYield || elseYield.getOperand(podResultIndex) != carriedPod) {
        return failure();
      }
    }

    rewriter.replaceOp(readOp, valueRes);
    return success();
  }
};

/// State for a POD record written directly in one or both branches of an `scf.if`.
///
/// When the write can be lifted, the branch-local `pod.write` ops are replaced by yielded scalar
/// values and a single parent-block write reconstructed from this tracked information.
struct IfWriteSlot {
  Value podRef;
  StringAttr recordName;
  Type type;
  WritePodOp thenWrite;
  WritePodOp elseWrite;
  Value incomingValue;
};

/// Find the tracked branch-write slot for `podRef.recordName`.
static IfWriteSlot *
lookupSlot(SmallVectorImpl<IfWriteSlot> &slots, Value podRef, StringAttr recordName) {
  for (IfWriteSlot &slot : slots) {
    if (slot.podRef == podRef && slot.recordName == recordName) {
      return &slot;
    }
  }
  return nullptr;
}

/// Return the tracked branch-write slot for `podRef.recordName`, creating it on first use.
static IfWriteSlot &getOrCreateSlot(
    SmallVectorImpl<IfWriteSlot> &slots, Value podRef, StringAttr recordName, Type type
) {
  if (IfWriteSlot *slot = lookupSlot(slots, podRef, recordName)) {
    return *slot;
  }
  slots.push_back(IfWriteSlot {podRef, recordName, type, nullptr, nullptr, Value()});
  return slots.back();
}

/// Return the else block when present; otherwise null.
static Block *getElseBlockOrNull(scf::IfOp ifOp) {
  return ifOp.getElseRegion().empty() ? nullptr : &ifOp.getElseRegion().front();
}

/// Collect direct `pod.write` operations from a branch, grouped by POD reference and record name.
static void
collectDirectWrites(Block *block, bool isThenBlock, SmallVectorImpl<IfWriteSlot> &slots) {
  if (!block) {
    return;
  }

  for (Operation &op : *block) {
    if (op.hasTrait<OpTrait::IsTerminator>()) {
      break;
    }

    auto writeOp = dyn_cast<WritePodOp>(&op);
    if (!writeOp) {
      continue;
    }

    IfWriteSlot &slot = getOrCreateSlot(
        slots, writeOp.getPodRef(), writeOp.getRecordNameAttr(), writeOp.getValue().getType()
    );
    if (isThenBlock) {
      slot.thenWrite = writeOp;
    } else {
      slot.elseWrite = writeOp;
    }
  }
}

/// Return whether writes to `podRef.recordName` can be lifted out of the branch as an SSA result.
///
/// Lifting is rejected when nested writes may alias the same record or when a later read in the
/// branch would observe branch-local mutation ordering that the lifted form would not preserve.
static bool branchSlotCanBeLifted(Block *block, Value podRef, StringAttr recordName) {
  if (!block) {
    return true;
  }

  bool seenDirectWrite = false;
  for (Operation &op : *block) {
    if (op.hasTrait<OpTrait::IsTerminator>()) {
      return true;
    }

    if (auto writeOp = dyn_cast<WritePodOp>(&op)) {
      if (isSamePodRecord(writeOp, podRef, recordName)) {
        seenDirectWrite = true;
        continue;
      }
    }

    if (hasNestedWriteToRecord(op, podRef, recordName)) {
      return false;
    }
    if (seenDirectWrite && (hasReadFromRecord(op, podRef, recordName) || hasValueUse(op, podRef))) {
      return false;
    }
  }
  return true;
}

/// Return whether `op` is one of the branch writes that will be recreated after the lifted `if`.
static bool isLiftedWrite(Operation &op, ArrayRef<IfWriteSlot> slots) {
  auto writeOp = dyn_cast<WritePodOp>(&op);
  return writeOp && llvm::any_of(slots, [&writeOp](const IfWriteSlot &slot) {
    return isSamePodRecord(writeOp, slot.podRef, slot.recordName);
  });
}

/// Return the `scf.yield` terminator from `block`.
static scf::YieldOp getYieldOp(Block &block) {
  auto yieldOp = dyn_cast<scf::YieldOp>(block.getTerminator());
  assert(yieldOp && "expected scf.if branch to terminate with scf.yield");
  return yieldOp;
}

/// Remove the default terminator from a freshly created SCF block before cloning contents into it.
static void dropTerminatorIfPresent(Block &block) {
  if (!block.empty() && block.back().hasTrait<OpTrait::IsTerminator>()) {
    block.back().erase();
  }
}

/// Move non-lifted branch operations into the replacement branch block.
static void
moveBranchWithoutLiftedWrites(Block *srcBlock, Block &destBlock, ArrayRef<IfWriteSlot> slots) {
  if (srcBlock) {
    for (auto it = srcBlock->begin(), end = srcBlock->end(); it != end;) {
      Operation &op = *it++;
      if (op.hasTrait<OpTrait::IsTerminator>() || isLiftedWrite(op, slots)) {
        continue;
      }
      op.moveBefore(&destBlock, destBlock.end());
    }
  }
}

/// Finish a lifted branch by yielding the original branch results followed by one lifted POD value
/// per tracked record.
static void appendYield(
    OpBuilder &bldr, Location loc, Block &block, ValueRange priorYieldValues,
    ArrayRef<IfWriteSlot> slots, bool isThenBlock
) {
  SmallVector<Value> yieldValues = llvm::to_vector(priorYieldValues);
  llvm::append_range(yieldValues, llvm::map_range(slots, [isThenBlock](const IfWriteSlot &slot) {
    WritePodOp writeOp = isThenBlock ? slot.thenWrite : slot.elseWrite;
    return writeOp ? writeOp.getValue() : slot.incomingValue;
  }));

  bldr.setInsertionPointToEnd(&block);
  bldr.create<scf::YieldOp>(loc, yieldValues);
}

/// One POD record whose value is carried across an SCF loop boundary as an SSA scalar.
///
/// These slots are populated for direct `pod.read` and `pod.write` accesses that refer to POD
/// values defined outside the loop and therefore need explicit iter args/block arguments/results
/// after lifting.
struct LoopPodSlot {
  Value podRef;
  StringAttr recordName;
  Type type;

  /// Return whether this slot is `findPodRef.findRecordName`.
  bool matches(Value findPodRef, StringAttr findRecordName) const {
    return this->podRef == findPodRef && this->recordName == findRecordName;
  }
};

/// Return the tracked loop slot for `podRef.recordName`, or null if not found.
static LoopPodSlot *
lookupLoopSlot(SmallVectorImpl<LoopPodSlot> &slots, Value podRef, StringAttr recordName) {
  auto *it = llvm::find_if(slots, [&podRef, &recordName](const LoopPodSlot &slot) {
    return slot.matches(podRef, recordName);
  });
  return it == slots.end() ? nullptr : &*it;
}

/// Return whether a loop slot is tracked for `podRef.recordName`.
static bool hasLoopSlot(ArrayRef<LoopPodSlot> slots, Value podRef, StringAttr recordName) {
  const auto *it = llvm::find_if(slots, [&podRef, &recordName](const LoopPodSlot &slot) {
    return slot.matches(podRef, recordName);
  });
  return it != slots.end();
}

/// Return the tracked loop slot for `podRef.recordName`, creating it on first use.
static LoopPodSlot &getOrCreateLoopSlot(
    SmallVectorImpl<LoopPodSlot> &slots, Value podRef, StringAttr recordName, Type type
) {
  if (LoopPodSlot *slot = lookupLoopSlot(slots, podRef, recordName)) {
    return *slot;
  }
  slots.push_back(LoopPodSlot {podRef, recordName, type});
  return slots.back();
}

/// Find the stable index of the tracked loop slot for `podRef.recordName`.
static std::optional<size_t>
findLoopSlotIndex(ArrayRef<LoopPodSlot> slots, Value podRef, StringAttr recordName) {
  for (auto [idx, slot] : llvm::enumerate(slots)) {
    if (slot.podRef == podRef && slot.recordName == recordName) {
      return idx;
    }
  }
  return std::nullopt;
}

/// Collect direct POD reads and writes that cross the loop boundary and therefore need an
/// explicit loop-carried scalar value when lifted.
static void
collectDirectLoopPodSlots(Block &block, Operation *ancestor, SmallVectorImpl<LoopPodSlot> &slots) {
  for (Operation &op : block) {
    if (auto readOp = dyn_cast<ReadPodOp>(&op)) {
      if (!isValueDefinedInside(ancestor, readOp.getPodRef())) {
        getOrCreateLoopSlot(
            slots, readOp.getPodRef(), readOp.getRecordNameAttr(), readOp.getType()
        );
      }
      continue;
    }

    if (auto writeOp = dyn_cast<WritePodOp>(&op)) {
      if (!isValueDefinedInside(ancestor, writeOp.getPodRef())) {
        getOrCreateLoopSlot(
            slots, writeOp.getPodRef(), writeOp.getRecordNameAttr(), writeOp.getValue().getType()
        );
      }
    }
  }
}

/// Return whether `op` directly uses a POD reference tracked for loop lifting.
static bool opUsesTrackedPodRefDirectly(Operation &op, ArrayRef<LoopPodSlot> slots) {
  return llvm::any_of(op.getOperands(), [&slots](Value operand) {
    return llvm::any_of(slots, [&operand](const LoopPodSlot &slot) {
      return slot.podRef == operand;
    });
  });
}

/// Return whether `op` contains nested POD accesses tracked for loop lifting.
static bool hasNestedTrackedPodAccess(Operation &op, ArrayRef<LoopPodSlot> slots) {
  return op
      .walk([&op, &slots](Operation *nestedOp) {
    if (nestedOp == &op) {
      return WalkResult::advance();
    }

    if (auto readOp = dyn_cast<ReadPodOp>(nestedOp)) {
      if (hasLoopSlot(slots, readOp.getPodRef(), readOp.getRecordNameAttr())) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }

    if (auto writeOp = dyn_cast<WritePodOp>(nestedOp)) {
      if (hasLoopSlot(slots, writeOp.getPodRef(), writeOp.getRecordNameAttr())) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  }).wasInterrupted();
}

/// Return whether the loop body contains non-POD operations that still observe the tracked POD
/// references directly, which would make the simple lifting rewrite invalid.
static bool hasUnliftableLoopPodUses(Block &block, ArrayRef<LoopPodSlot> slots) {
  for (Operation &op : block) {
    if (isa<ReadPodOp, WritePodOp>(op)) {
      continue;
    }
    if (opUsesTrackedPodRefDirectly(op, slots) || hasNestedTrackedPodAccess(op, slots)) {
      return true;
    }
  }
  return false;
}

/// Lift direct branch-local writes out of `scf.if` as yielded values, then write those values in
/// the parent block. Existing `scf.if` results are preserved as a prefix of the new result list,
/// which gives mem2reg parent-block pod writes instead of nested-region writes.
class LiftPodWritesFromIfBlocksPattern final : public OpRewritePattern<scf::IfOp> {
public:
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp, PatternRewriter &rewriter) const override {
    SmallVector<IfWriteSlot> slots;
    Block &thenBlock = *ifOp.thenBlock();
    Block *elseBlock = getElseBlockOrNull(ifOp);
    collectDirectWrites(&thenBlock, true, slots);
    collectDirectWrites(elseBlock, false, slots);
    if (slots.empty()) {
      return failure();
    }

    llvm::erase_if(slots, [&](const IfWriteSlot &slot) {
      return isValueDefinedInside(ifOp, slot.podRef) ||
             !branchSlotCanBeLifted(&thenBlock, slot.podRef, slot.recordName) ||
             !branchSlotCanBeLifted(elseBlock, slot.podRef, slot.recordName);
    });
    if (slots.empty()) {
      return failure();
    }

    for (IfWriteSlot &slot : slots) {
      if (slot.thenWrite && slot.elseWrite) {
        continue;
      }
      rewriter.setInsertionPoint(ifOp);
      slot.incomingValue =
          genRead(rewriter, ifOp.getLoc(), slot.podRef, slot.recordName).getResult();
    }

    SmallVector<Type> resultTypes = llvm::to_vector(ifOp.getResultTypes());
    llvm::append_range(resultTypes, llvm::map_range(slots, [](auto slot) { return slot.type; }));

    SmallVector<Value> originalThenYields;
    if (!ifOp.getResults().empty()) {
      scf::YieldOp thenYieldOp = getYieldOp(thenBlock);
      originalThenYields.append(thenYieldOp.getOperands().begin(), thenYieldOp.getOperands().end());
    }

    SmallVector<Value> originalElseYields;
    if (elseBlock && !ifOp.getResults().empty()) {
      scf::YieldOp elseYieldOp = getYieldOp(*elseBlock);
      originalElseYields.append(elseYieldOp.getOperands().begin(), elseYieldOp.getOperands().end());
    }

    rewriter.setInsertionPoint(ifOp);
    auto newIf = rewriter.create<scf::IfOp>(ifOp.getLoc(), resultTypes, ifOp.getCondition(), true);
    Block &newThenBlock = *newIf.thenBlock();
    Block &newElseBlock = *newIf.elseBlock();
    dropTerminatorIfPresent(newThenBlock);
    dropTerminatorIfPresent(newElseBlock);

    moveBranchWithoutLiftedWrites(&thenBlock, newThenBlock, slots);
    moveBranchWithoutLiftedWrites(elseBlock, newElseBlock, slots);
    appendYield(rewriter, ifOp.getLoc(), newThenBlock, originalThenYields, slots, true);
    appendYield(rewriter, ifOp.getLoc(), newElseBlock, originalElseYields, slots, false);

    rewriter.setInsertionPointAfter(newIf);
    unsigned originalResultCount = ifOp.getNumResults();
    for (auto [idx, slot] : llvm::enumerate(slots)) {
      genWrite(
          rewriter, ifOp.getLoc(), slot.podRef, slot.recordName,
          newIf.getResult(originalResultCount + idx)
      );
    }

    rewriter.replaceOp(ifOp, newIf.getResults().take_front(originalResultCount));
    return success();
  }
};

/// Rewrite loop-local POD reads and writes in an `scf.for` into extra iter args/results carrying
/// one SSA value per touched POD record.
class LiftPodAccessesFromForLoopPattern final : public OpRewritePattern<scf::ForOp> {
public:
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp, PatternRewriter &rewriter) const override {
    Block &body = *forOp.getBody();
    SmallVector<LoopPodSlot> slots;
    collectDirectLoopPodSlots(body, forOp.getOperation(), slots);
    if (slots.empty() || hasUnliftableLoopPodUses(body, slots)) {
      return failure();
    }

    Location loc = forOp.getLoc();

    SmallVector<Value> newInitArgs = llvm::to_vector(forOp.getInitArgs());
    rewriter.setInsertionPoint(forOp);
    for (const LoopPodSlot &slot : slots) {
      newInitArgs.push_back(genRead(rewriter, loc, slot.podRef, slot.recordName).getResult());
    }

    auto newFor = rewriter.create<scf::ForOp>(
        loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(), newInitArgs
    );
    newFor->setAttrs(forOp->getAttrs());

    Block &newBody = *newFor.getBody();
    dropTerminatorIfPresent(newBody);

    IRMapping mapping;
    mapping.map(forOp.getInductionVar(), newFor.getInductionVar());
    for (auto [idx, oldArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
      mapping.map(oldArg, newFor.getRegionIterArg(idx));
    }

    SmallVector<Value> slotValues = llvm::map_to_vector(
        llvm::seq<size_t>(0, slots.size()),
        [base = static_cast<size_t>(forOp.getNumRegionIterArgs()), &newFor](size_t idx) -> Value {
      return newFor.getRegionIterArg(llzk::checkedCast<unsigned>(base + idx));
    }
    );

    rewriter.setInsertionPointToEnd(&newBody);
    for (Operation &op : body) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(&op)) {
        auto yieldValues = llvm::map_to_vector(yieldOp.getOperands(), [&mapping](Value operand) {
          return mapping.lookupOrDefault(operand);
        });
        llvm::append_range(yieldValues, slotValues);
        rewriter.create<scf::YieldOp>(yieldOp.getLoc(), yieldValues);
        continue;
      }

      if (auto readOp = dyn_cast<ReadPodOp>(&op)) {
        if (std::optional<size_t> slotIdx =
                findLoopSlotIndex(slots, readOp.getPodRef(), readOp.getRecordNameAttr())) {
          mapping.map(readOp.getResult(), slotValues[*slotIdx]);
          continue;
        }
      }

      if (auto writeOp = dyn_cast<WritePodOp>(&op)) {
        if (std::optional<size_t> slotIdx =
                findLoopSlotIndex(slots, writeOp.getPodRef(), writeOp.getRecordNameAttr())) {
          slotValues[*slotIdx] = mapping.lookupOrDefault(writeOp.getValue());
          continue;
        }
      }

      rewriter.clone(op, mapping);
    }

    rewriter.setInsertionPointAfter(newFor);
    for (auto [idx, slot] : llvm::enumerate(slots)) {
      genWrite(
          rewriter, loc, slot.podRef, slot.recordName, newFor.getResult(forOp.getNumResults() + idx)
      );
    }

    rewriter.replaceOp(forOp, newFor.getResults().take_front(forOp.getNumResults()));
    return success();
  }
};

/// Rewrite loop-local POD reads and writes in an `scf.while` into extra block arguments/results
/// carrying one SSA value per touched POD record.
class LiftPodAccessesFromWhileLoopPattern final : public OpRewritePattern<scf::WhileOp> {
public:
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp, PatternRewriter &rewriter) const override {
    Block &beforeBody = *whileOp.getBeforeBody();
    Block &afterBody = *whileOp.getAfterBody();

    SmallVector<LoopPodSlot> slots;
    collectDirectLoopPodSlots(beforeBody, whileOp.getOperation(), slots);
    collectDirectLoopPodSlots(afterBody, whileOp.getOperation(), slots);
    if (slots.empty() || hasUnliftableLoopPodUses(beforeBody, slots) ||
        hasUnliftableLoopPodUses(afterBody, slots)) {
      return failure();
    }

    Location loc = whileOp.getLoc();

    SmallVector<Value> newInits = llvm::to_vector(whileOp.getInits());
    SmallVector<Type> newResultTypes = llvm::to_vector(whileOp.getResultTypes());
    rewriter.setInsertionPoint(whileOp);
    for (const LoopPodSlot &slot : slots) {
      newInits.push_back(genRead(rewriter, loc, slot.podRef, slot.recordName).getResult());
      newResultTypes.push_back(slot.type);
    }

    auto newWhile = rewriter.create<scf::WhileOp>(loc, newResultTypes, newInits, nullptr, nullptr);
    newWhile->setAttrs(whileOp->getAttrs());

    Block &newBeforeBody = *newWhile.getBeforeBody();
    Block &newAfterBody = *newWhile.getAfterBody();
    dropTerminatorIfPresent(newBeforeBody);
    dropTerminatorIfPresent(newAfterBody);

    IRMapping beforeMapping;
    for (auto [oldArg, newArg] : llvm::zip_equal(
             whileOp.getBeforeArguments(),
             newWhile.getBeforeArguments().take_front(whileOp.getBeforeArguments().size())
         )) {
      beforeMapping.map(oldArg, newArg);
    }

    SmallVector<Value> beforeSlotValues = llvm::map_to_vector(
        llvm::seq<size_t>(0, slots.size()),
        [base = whileOp.getBeforeArguments().size(), &newWhile](size_t idx) -> Value {
      return newWhile.getBeforeArguments()[llzk::checkedCast<unsigned>(base + idx)];
    }
    );

    rewriter.setInsertionPointToEnd(&newBeforeBody);
    for (Operation &op : beforeBody) {
      if (auto conditionOp = dyn_cast<scf::ConditionOp>(&op)) {
        SmallVector<Value> conditionArgs =
            llvm::map_to_vector(conditionOp.getArgs(), [&beforeMapping](Value a) {
          return beforeMapping.lookupOrDefault(a);
        });
        llvm::append_range(conditionArgs, beforeSlotValues);
        rewriter.create<scf::ConditionOp>(
            conditionOp.getLoc(), beforeMapping.lookupOrDefault(conditionOp.getCondition()),
            conditionArgs
        );
        continue;
      }

      if (auto readOp = dyn_cast<ReadPodOp>(&op)) {
        if (std::optional<size_t> slotIdx =
                findLoopSlotIndex(slots, readOp.getPodRef(), readOp.getRecordNameAttr())) {
          beforeMapping.map(readOp.getResult(), beforeSlotValues[*slotIdx]);
          continue;
        }
      }

      if (auto writeOp = dyn_cast<WritePodOp>(&op)) {
        if (std::optional<size_t> slotIdx =
                findLoopSlotIndex(slots, writeOp.getPodRef(), writeOp.getRecordNameAttr())) {
          beforeSlotValues[*slotIdx] = beforeMapping.lookupOrDefault(writeOp.getValue());
          continue;
        }
      }

      rewriter.clone(op, beforeMapping);
    }

    IRMapping afterMapping;
    for (auto [oldArg, newArg] : llvm::zip_equal(
             whileOp.getAfterArguments(),
             newWhile.getAfterArguments().take_front(whileOp.getAfterArguments().size())
         )) {
      afterMapping.map(oldArg, newArg);
    }

    SmallVector<Value> afterSlotValues = llvm::map_to_vector(
        llvm::seq<size_t>(0, slots.size()),
        [base = whileOp.getAfterArguments().size(), &newWhile](size_t idx) -> Value {
      return newWhile.getAfterArguments()[llzk::checkedCast<unsigned>(base + idx)];
    }
    );

    rewriter.setInsertionPointToEnd(&newAfterBody);
    for (Operation &op : afterBody) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(&op)) {
        SmallVector<Value> yieldValues =
            llvm::map_to_vector(yieldOp.getOperands(), [&afterMapping](Value v) {
          return afterMapping.lookupOrDefault(v);
        });
        llvm::append_range(yieldValues, afterSlotValues);
        rewriter.create<scf::YieldOp>(yieldOp.getLoc(), yieldValues);
        continue;
      }

      if (auto readOp = dyn_cast<ReadPodOp>(&op)) {
        if (std::optional<size_t> slotIdx =
                findLoopSlotIndex(slots, readOp.getPodRef(), readOp.getRecordNameAttr())) {
          afterMapping.map(readOp.getResult(), afterSlotValues[*slotIdx]);
          continue;
        }
      }

      if (auto writeOp = dyn_cast<WritePodOp>(&op)) {
        if (std::optional<size_t> slotIdx =
                findLoopSlotIndex(slots, writeOp.getPodRef(), writeOp.getRecordNameAttr())) {
          afterSlotValues[*slotIdx] = afterMapping.lookupOrDefault(writeOp.getValue());
          continue;
        }
      }

      rewriter.clone(op, afterMapping);
    }

    rewriter.setInsertionPointAfter(newWhile);
    for (auto [idx, slot] : llvm::enumerate(slots)) {
      genWrite(
          rewriter, loc, slot.podRef, slot.recordName,
          newWhile.getResult(whileOp.getNumResults() + idx)
      );
    }

    rewriter.replaceOp(whileOp, newWhile.getResults().take_front(whileOp.getNumResults()));
    return success();
  }
};

/// Apply a greedy rewrite/fold pass over the module body using the provided patterns.
static LogicalResult
applyGreedily(ModuleOp modOp, RewritePatternSet &&patterns, bool *changed = nullptr) {
  return applyPatternsGreedily(
      modOp->getRegion(0), std::move(patterns),
      GreedyRewriteConfig {.fold = false, .cseConstants = false}, changed
  );
}

/// Repeatedly lift pod accesses out of supported SCF regions so SROA + mem2reg can eliminate the
/// remaining POD storage.
static LogicalResult step4(ModuleOp modOp) {
  RewritePatternSet patterns(modOp.getContext());
  patterns.add<
      FoldReadAfterWriteInBlockPattern, ReplaceIfReadPattern, LiftPodWritesFromIfBlocksPattern,
      LiftPodAccessesFromForLoopPattern, LiftPodAccessesFromWhileLoopPattern,
      FoldIfCarriedPodReadAfterWritePattern>(patterns.getContext());

  LLVM_DEBUG(llvm::dbgs() << "Begin step 4: refactor pod ops within SCF regions\n";);
  return applyGreedily(modOp, std::move(patterns));
}

/// Run only the read-after-write fold for `scf.if`-carried POD results and report whether it
/// changed the IR.
static bool applyIfCarriedPodReadAfterWritePatterns(ModuleOp modOp) {
  RewritePatternSet patterns(modOp.getContext());
  patterns.add<FoldIfCarriedPodReadAfterWritePattern>(patterns.getContext());

  bool changed = false;
  if (failed(applyGreedily(modOp, std::move(patterns), &changed))) {
    return false;
  }
  return changed;
}

/// Return a simple measure of how many POD allocation layers are represented by this type.
/// Non-POD records have weight zero; each POD layer contributes one plus its nested POD records.
static size_t podTypeScalarizationWeight(Type type) {
  auto podTy = dyn_cast<PodType>(type);
  if (!podTy) {
    return 0;
  }

  size_t weight = 1;
  for (RecordAttr record : podTy.getRecords()) {
    weight += podTypeScalarizationWeight(record.getType());
  }
  return weight;
}

/// Return the total remaining POD allocation work in the module. This is used to rerun
/// SROA+mem2reg while recursive POD layers are being exposed, and to stop if a pass round cannot
/// make progress.
static size_t podAllocScalarizationWeight(ModuleOp modOp) {
  size_t weight = 0;
  modOp.walk([&weight](NewPodOp newPodOp) {
    weight += podTypeScalarizationWeight(newPodOp.getType());
  });
  return weight;
}

/// Pass driver for the full POD-to-scalar lowering pipeline described above.
class PassImpl : public llzk::pod::impl::PodToScalarPassBase<PassImpl> {
  using Base = PodToScalarPassBase<PassImpl>;
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (failed(step0(module))) {
      return signalPassFailure();
    }
    LLVM_DEBUG({
      llvm::dbgs() << "After step 0:\n";
      module.dump();
    });

    {
      // This is divided into 2 steps to simplify the implementation for member-related ops. The
      // issue is that the conversions for member read/write expect the mapping of record name to
      // member name+type to already be populated for the referenced member (although this could be
      // computed on demand if desired but it complicates the implementation a bit).
      SymbolTableCollection symTables;
      MemberReplacementMap memberRepMap;
      if (failed(step1(module, symTables, memberRepMap))) {
        return signalPassFailure();
      }
      LLVM_DEBUG({
        llvm::dbgs() << "After step 1:\n";
        module.dump();
      });

      if (failed(step2(module, symTables, memberRepMap))) {
        return signalPassFailure();
      }
      LLVM_DEBUG({
        llvm::dbgs() << "After step 2:\n";
        module.dump();
      });

      if (failed(step3(module, symTables, memberRepMap))) {
        return signalPassFailure();
      }
      LLVM_DEBUG({
        llvm::dbgs() << "After step 3:\n";
        module.dump();
      });
    }

    if (failed(step4(module))) {
      return signalPassFailure();
    }
    LLVM_DEBUG({
      llvm::dbgs() << "After step 4:\n";
      module.dump();
    });

    // 1. Use SROA (Destructurable* interfaces) to split each pod with `N` records into `N` pods
    // with 1 record each. This is necessary because the mem2reg pass cannot deal with splitting
    // up memory, i.e., it can only convert scalar memory access into SSA values.
    // 2. The mem2reg pass converts the size 1 pod allocations and accesses into SSA values.
    OpPassManager scalarizePM(ModuleOp::getOperationName());
    scalarizePM.addPass(createSpecializedSROAPass<NewPodOp>());
    scalarizePM.addPass(createSpecializedMem2RegPass<NewPodOp>());

    // Cleanup allocations made dead by memory promotion and other dead SSA values.
    OpPassManager cleanupPM(ModuleOp::getOperationName());
    cleanupPM.addPass(createRemoveUnusedDiscardableAllocationsPass(
        RemoveUnusedDiscardableAllocationsPassOptions {
            .allocatorOpName = CreateArrayOp::getOperationName().str()
        }
    ));
    cleanupPM.addPass(createRemoveUnusedDiscardableAllocationsPass(
        RemoveUnusedDiscardableAllocationsPassOptions {
            .allocatorOpName = NewPodOp::getOperationName().str()
        }
    ));
    cleanupPM.addPass(createRemoveDeadValuesWorkaroundPass());

    size_t podAllocWeight = podAllocScalarizationWeight(module);
    while (podAllocWeight != 0) {
      if (failed(runPipeline(scalarizePM, module))) {
        signalPassFailure();
        return;
      }

      // SROA+mem2reg can expose `scf.if`-carried POD values that become redundant after a
      // same-record write from another `scf.if` result. Fold those reads and clean up before
      // checking convergence.
      bool foldedIfCarriedRead = applyIfCarriedPodReadAfterWritePatterns(module);
      if (failed(runPipeline(cleanupPM, module))) {
        signalPassFailure();
        return;
      }

      // Nested PODs can become visible only after an outer single-record POD has been promoted,
      // and SROA can transiently increase allocation count while splitting aggregates. Keep
      // iterating until the allocation-weight heuristic reaches a fixed point.
      size_t nextPodAllocWeight = podAllocScalarizationWeight(module);
      if (!foldedIfCarriedRead && nextPodAllocWeight == podAllocWeight) {
        break;
      }
      podAllocWeight = nextPodAllocWeight;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "After SROA+Mem2Reg pipeline:\n";
      module.dump();
    });
  }
};

} // namespace
