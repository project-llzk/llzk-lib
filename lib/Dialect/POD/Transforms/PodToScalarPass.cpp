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
///    in `llzk.nondet`, `array.*`, `MemberReadOp`, `MemberWriteOp`, `FuncDefOp`, `CallOp`, and
///    `ReturnOp`.
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
#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
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
#include "llzk/Dialect/RAM/IR/Dialect.h"
#include "llzk/Dialect/String/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKConversionUtils.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Transforms/SpecializedMemoryPasses.h"
#include "llzk/Util/Concepts.h"
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
    return arrTy.cloneWith(getFlattenedTypeAlongPath(nextType, recordChain.drop_front()));
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
        walk(arrTy.cloneWith(record.getType()));
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
    forEachPodLeaf(pt, recordChain, [&collect](RecordChain, Type leafType) {
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
      collect.push_back(at.cloneWith(leafType));
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

/// Create a `pod.read` for one record of `podRef`.
inline static ReadPodOp
genRead(Location loc, Value podRef, StringAttr recordName, OpBuilder &bldr) {
  Type resultType =
      llvm::cast<PodType>(podRef.getType()).getRecordMap().lookup(recordName.getValue());
  return bldr.create<ReadPodOp>(loc, resultType, podRef, recordName);
}

/// Create a `pod.write` for one record of `podRef`.
inline static WritePodOp
genWrite(Location loc, Value podRef, StringAttr recordName, Value value, OpBuilder &bldr) {
  return bldr.create<WritePodOp>(loc, podRef, recordName, value);
}

/// Return the single converted value from a 1:N adaptor range.
inline static Value getSingleConvertedValue(ValueRange values) {
  assert(values.size() == 1 && "expected a 1:1 converted value range");
  return values.front();
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

/// Generate `arith.constant` indices for one static array element position.
static SmallVector<Value> genArrayIndexConstants(ArrayAttr index, Location loc, OpBuilder &bldr) {
  SmallVector<Value> indices;
  for (Attribute attr : index) {
    assert(llvm::isa<IntegerAttr>(attr) && "array index must be an integer attribute");
    indices.push_back(bldr.create<arith::ConstantOp>(loc, llvm::cast<IntegerAttr>(attr)));
  }
  return indices;
}

/// Create an `array.read` for one concrete element or subarray.
inline static ReadArrayOp
genArrayRead(Location loc, Value arrayRef, ArrayAttr index, OpBuilder &bldr) {
  Type t = arrayRef.getType();
  assert(llvm::isa<ArrayType>(t) && "array.read must target an array type");
  return bldr.create<ReadArrayOp>(
      loc, llvm::cast<ArrayType>(t).getElementType(), arrayRef,
      genArrayIndexConstants(index, loc, bldr)
  );
}

/// Create an `array.write` for one concrete element or subarray.
inline static WriteArrayOp
genArrayWrite(Location loc, Value arrayRef, ArrayAttr index, Value value, OpBuilder &bldr) {
  return bldr.create<WriteArrayOp>(loc, arrayRef, genArrayIndexConstants(index, loc, bldr), value);
}

/// Read one flattened POD leaf, including leaves that live inside an array-of-POD record.
static Value
genReadAlongPath(Location loc, Value value, ArrayRef<StringAttr> recordChain, OpBuilder &bldr) {
  if (recordChain.empty()) {
    return value;
  }

  Type valueType = value.getType();
  if (llvm::isa<PodType>(valueType)) {
    Value nextValue = genRead(loc, value, recordChain.front(), bldr);
    return genReadAlongPath(loc, nextValue, recordChain.drop_front(), bldr);
  }

  if (ArrayType arrTy = splittablePodArray(valueType)) {
    assert(arrTy.hasStaticShape() && "nested array-of-POD scalarization requires a static shape");
    auto splitArrTy = llvm::cast<ArrayType>(getFlattenedTypeAlongPath(valueType, recordChain));
    auto subIndices = arrTy.getSubelementIndices();
    assert(subIndices && "static-shape arrays must provide subelement indices");

    Value splitArray = bldr.create<CreateArrayOp>(loc, splitArrTy);
    for (ArrayAttr index : *subIndices) {
      Value element = genArrayRead(loc, value, index, bldr);
      Value leafValue = genReadAlongPath(loc, element, recordChain, bldr);
      genArrayWrite(loc, splitArray, index, leafValue, bldr);
    }
    return splitArray;
  }

  llvm_unreachable("record path cannot continue through a non-POD leaf");
}

/// Read a flattened POD leaf by following each record name in `recordChain`.
inline static Value
genReadAlongPath(Location loc, Value podRef, RecordChain recordChain, OpBuilder &bldr) {
  return genReadAlongPath(loc, podRef, ArrayRef(recordChain.nameList), bldr);
}

/// Reconstruct a POD record from the leaf values collected while splitting nested accesses.
static Value rebuildFlattenedPodRecord(
    Location loc, Type recordType, SmallVectorImpl<StringAttr> &recordChain,
    const DenseMap<RecordChain, Value> &leafValues, OpBuilder &bldr
) {
  if (PodType nestedPodTy = dyn_cast<PodType>(recordType)) {
    NewPodOp nestedPod = bldr.create<NewPodOp>(loc, nestedPodTy);
    for (RecordAttr record : nestedPodTy.getRecords()) {
      recordChain.push_back(record.getName());
      Value recordValue =
          rebuildFlattenedPodRecord(loc, record.getType(), recordChain, leafValues, bldr);
      genWrite(loc, nestedPod, record.getName(), recordValue, bldr);
      recordChain.pop_back();
    }
    return nestedPod;
  }

  if (ArrayType arrTy = splittablePodArray(recordType)) {
    assert(arrTy.hasStaticShape() && "nested array-of-POD scalarization requires a static shape");
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
        elementLeafValues[id] = genArrayRead(loc, it->second, index, bldr);
      });

      NewPodOp elementPod = bldr.create<NewPodOp>(loc, elemPodTy);
      SmallVector<StringAttr> nestedChain;
      for (RecordAttr record : elemPodTy.getRecords()) {
        nestedChain.push_back(record.getName());
        Value recordValue =
            rebuildFlattenedPodRecord(loc, record.getType(), nestedChain, elementLeafValues, bldr);
        genWrite(loc, elementPod, record.getName(), recordValue, bldr);
        nestedChain.pop_back();
      }
      genArrayWrite(loc, rebuiltArray, index, elementPod, bldr);
    }
    return rebuiltArray;
  }

  auto it = leafValues.find(RecordChain(recordChain));
  assert(it != leafValues.end() && "missing flattened POD leaf value");
  return it->second;
}

/// Populate a POD value from its flattened leaf values.
static void populateFlattenedPodValue(
    Location loc, Value podValue, PodType podTy, const DenseMap<RecordChain, Value> &leafValues,
    OpBuilder &bldr
) {
  SmallVector<StringAttr> recordChain;
  for (RecordAttr record : podTy.getRecords()) {
    recordChain.push_back(record.getName());
    Value recordValue =
        rebuildFlattenedPodRecord(loc, record.getType(), recordChain, leafValues, bldr);
    genWrite(loc, podValue, record.getName(), recordValue, bldr);
    recordChain.pop_back();
  }
}

using VirtualPodLeafMap = DenseMap<RecordChain, Value>;
using VirtualPodValueMap = DenseMap<Value, VirtualPodLeafMap>;

/// Return the flattened leaf values for `podValue` when it is tracked as a virtual POD.
static const VirtualPodLeafMap *
lookupVirtualPodLeafMap(Value podValue, const VirtualPodValueMap &virtualPods) {
  auto it = virtualPods.find(podValue);
  return it != virtualPods.end() ? &it->second : nullptr;
}

/// Collect flattened POD leaf values in canonical traversal order.
static SmallVector<Value>
orderedVirtualPodLeafValues(PodType podTy, const VirtualPodLeafMap &leafValues) {
  SmallVector<Value> orderedValues;
  SmallVector<StringAttr> recordChain;
  forEachPodLeaf(podTy, recordChain, [&leafValues, &orderedValues](RecordChain id, Type) {
    auto it = leafValues.find(id);
    assert(it != leafValues.end() && "missing virtual POD leaf value");
    orderedValues.push_back(it->second);
  });
  return orderedValues;
}

/// Materialize the tracked contents of a virtual POD into concrete `pod.write` operations.
inline static void
materializeVirtualPod(NewPodOp pod, const VirtualPodLeafMap &leafValues, OpBuilder &bldr) {
  populateFlattenedPodValue(pod.getLoc(), pod, pod.getType(), leafValues, bldr);
}

/// Return `true` iff a read from a virtual POD can be resolved without materializing it.
static bool canResolveVirtualPodRead(ReadPodOp op, const VirtualPodValueMap &virtualPods) {
  if (!lookupVirtualPodLeafMap(op.getPodRef(), virtualPods)) {
    return false;
  }
  Type recType = llvm::cast<PodType>(op.getPodRefType()).getRecordMap().lookup(op.getRecordName());
  return llvm::isa<PodType>(recType) || !splittablePodArray(recType);
}

/// Return the suffixes to append to a function arg/result name when splitting the given type.
static SmallVector<std::string> getSplitRecordNameSuffixes(Type type) {
  SmallVector<std::string> suffixes;
  if (PodType pt = splittablePod(type)) {
    SmallVector<StringAttr> recordChain;
    forEachPodLeaf(pt, recordChain, [&suffixes](RecordChain id, Type) {
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
    ConversionPatternRewriter &rewriter, const VirtualPodValueMap *virtualPods = nullptr
) {
  if (PodType pt = splittablePod(operand.getType())) {
    if (virtualPods) {
      if (const VirtualPodLeafMap *leafValues = lookupVirtualPodLeafMap(operand, *virtualPods)) {
        llvm::append_range(newOperands, orderedVirtualPodLeafValues(pt, *leafValues));
        return;
      }
    }
    SmallVector<StringAttr> recordChain;
    forEachPodLeaf(pt, recordChain, [&](RecordChain id, Type) {
      newOperands.push_back(genReadAlongPath(loc, operand, id, rewriter));
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
    processInputOperand(op->getLoc(), v, newOperands, rewriter, virtualPods);
  }
  rewriter.modifyOpInPlace(op, [&outputOpRef, &newOperands]() {
    outputOpRef.assign(ValueRange(newOperands));
  });
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
  forEachPodLeaf(podTy, recordChain, [&](RecordChain id, Type ty) {
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
      for (auto [id, splitType] : llvm::zip_equal(splitIds, splitTypes)) {
        SmallVector<Value> splitElements;
        splitElements.reserve(adaptor.getElements().size());
        for (ValueRange elementRange : adaptor.getElements()) {
          Value element = getSingleConvertedValue(elementRange);
          splitElements.push_back(genReadAlongPath(op.getLoc(), element, id, rewriter));
        }
        replacements.push_back(rewriter.create<CreateArrayOp>(
            op.getLoc(), llvm::cast<ArrayType>(splitType), splitElements
        ));
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
      for (Type splitType : splitTypes) {
        replacements.push_back(rewriter.create<CreateArrayOp>(
            op.getLoc(), llvm::cast<ArrayType>(splitType), mapOperands, numDimsPerMap
        ));
      }
    }

    rewriter.replaceOpWithMultiple(op, {ValueRange(replacements)});
    return success();
  }
};

/// Split `array.read` from an array-of-POD into scalar leaf reads plus local POD reconstruction.
class SplitPodArrayReadArrayOp : public OpConversionPattern<ReadArrayOp> {
public:
  using OpConversionPattern<ReadArrayOp>::OpConversionPattern;

  static bool legal(ReadArrayOp op) { return !splittablePodArray(op.getArrRefType()); }

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
    for (auto [id, splitArrRange, splitType] :
         llvm::zip_equal(splitIds, adaptor.getArrRef(), splitTypes)) {
      auto splitArrTy = llvm::cast<ArrayType>(splitType);
      Value scalarRead = rewriter.create<ReadArrayOp>(
          op.getLoc(), splitArrTy.getElementType(), getSingleConvertedValue(splitArrRange), indices
      );
      leafValues[id] = scalarRead;
    }

    SmallVector<StringAttr> recordChain;
    for (RecordAttr record : podTy.getRecords()) {
      recordChain.push_back(record.getName());
      Value recordValue = rebuildFlattenedPodRecord(
          op.getLoc(), record.getType(), recordChain, leafValues, rewriter
      );
      genWrite(op.getLoc(), pod, record.getName(), recordValue, rewriter);
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
      Value leafValue = genReadAlongPath(op.getLoc(), podValue, id, rewriter);
      rewriter.create<WriteArrayOp>(
          op.getLoc(), getSingleConvertedValue(splitArrRange), indices, leafValue
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
    auto *tyConv = getTypeConverter();
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
    SmallVector<Value> newOperands = flattenConvertedValues(adaptor.getOperands());
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
    auto *tyConv = getTypeConverter();
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

/// Replace `array.length` on an array-of-POD with the equivalent length of any split leaf array.
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
    rewriter.replaceOpWithNewOp<ArrayLengthOp>(
        op, getSingleConvertedValue(adaptor.getArrRef()), getSingleConvertedValue(adaptor.getDim())
    );
    return success();
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
      SplitPodArrayInFuncDefOp, SplitPodArrayInReturnOp, SplitPodArrayInCallOp,
      SplitPodArrayLengthOp>(typeConverter, ctx);
  patterns.add<SplitPodArrayInMemberWriteOp, SplitPodArrayInMemberReadOp>(
      typeConverter, ctx, symTables, memberRepMap
  );

  ConversionTarget target(*ctx);
  baseTargetSetup(target);
  target.addDynamicallyLegalOp<NonDetOp>(SplitPodArrayNonDetOp::legal);
  target.addDynamicallyLegalOp<CreateArrayOp>(SplitPodArrayCreateArrayOp::legal);
  target.addDynamicallyLegalOp<ReadArrayOp>(SplitPodArrayReadArrayOp::legal);
  target.addDynamicallyLegalOp<WriteArrayOp>(SplitPodArrayWriteArrayOp::legal);
  target.addDynamicallyLegalOp<ExtractArrayOp>(SplitPodArrayExtractArrayOp::legal);
  target.addDynamicallyLegalOp<InsertArrayOp>(SplitPodArrayInsertArrayOp::legal);
  target.addDynamicallyLegalOp<FuncDefOp>(SplitPodArrayInFuncDefOp::legal);
  target.addDynamicallyLegalOp<ReturnOp>(SplitPodArrayInReturnOp::legal);
  target.addDynamicallyLegalOp<CallOp>(SplitPodArrayInCallOp::legal);
  target.addDynamicallyLegalOp<ArrayLengthOp>(SplitPodArrayLengthOp::legal);
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
            auto newPod = rewriter.create<NewPodOp>(loc, pt);
            rewriter.replaceAllUsesWith(oldV, newPod);
            // Remove the argument from the block
            entryBlock.eraseArgument(i);

            DenseMap<RecordChain, Value> leafValues;
            SmallVector<StringAttr> recordChain;
            forEachPodLeaf(pt, recordChain, [&](RecordChain id, Type leafType) {
              BlockArgument newArg = entryBlock.insertArgument(i, leafType, loc);
              leafValues[id] = newArg;
              ++i;
            });
            virtualPods[newPod] = std::move(leafValues);
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
      DenseMap<RecordChain, Value> leafValues;
      SmallVector<StringAttr> recordChain;
      forEachPodLeaf(pt, recordChain, [&leafValues, &newResults](RecordChain id, Type) {
        leafValues[id] = *newResults;
        ++newResults;
      });
      NewPodOp newPod = rewriter.create<NewPodOp>(loc, pt);
      virtualPods[newPod] = std::move(leafValues);
      rewriter.replaceAllUsesWith(oldVal, newPod);
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
        lookupVirtualPodLeafMap(adaptor.getVal(), virtualPods);

    for (auto [id, newMember] : idToMember) {
      Value scalarValue = virtualLeafValues
                              ? virtualLeafValues->at(id)
                              : genReadAlongPath(op.getLoc(), adaptor.getVal(), id, rewriter);
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
    for (auto [id, newMember] : idToMember) {
      leafValues[id] = rewriter.create<MemberReadOp>(
          op.getLoc(), newMember.second, adaptor.getComponent(), newMember.first
      );
    }

    NewPodOp pod = rewriter.create<NewPodOp>(op.getLoc(), llvm::cast<PodType>(op.getType()));
    virtualPods[pod] = std::move(leafValues);
    rewriter.replaceOp(op, pod);
  }
};

/// Resolve reads from a virtual POD placeholder without materializing the whole aggregate.
class ResolveVirtualPodReadOp : public OpConversionPattern<ReadPodOp> {
  VirtualPodValueMap &virtualPods;

public:
  ResolveVirtualPodReadOp(MLIRContext *ctx, VirtualPodValueMap &virtualPodMap)
      : OpConversionPattern<ReadPodOp>(ctx), virtualPods(virtualPodMap) {}

  LogicalResult matchAndRewrite(
      ReadPodOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
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
      NewPodOp pod = rewriter.create<NewPodOp>(op.getLoc(), nestedPodTy);
      virtualPods[pod] = std::move(nestedLeafValues);
      rewriter.replaceOp(op, pod);
      return success();
    }

    if (splittablePodArray(recordType)) {
      return failure();
    }

    rewriter.replaceOp(op, leafValues->at(RecordChain(prefix)));
    return success();
  }
};

/// Special handling to split pods in struct member refs and function signatures and desugar
/// initializations on pod.new into pod writes.
static LogicalResult
step3(ModuleOp modOp, SymbolTableCollection &symTables, const MemberReplacementMap &memberRepMap) {
  MLIRContext *ctx = modOp.getContext();
  VirtualPodValueMap virtualPods;

  RewritePatternSet patterns(ctx);
  patterns.add<SplitInitFromNewPodOp>(ctx);
  patterns.add<SplitPodInFuncDefOp, SplitPodInReturnOp, SplitPodInCallOp>(ctx, virtualPods);
  patterns.add<SplitPodInMemberWriteOp, SplitPodInMemberReadOp>(
      ctx, symTables, memberRepMap, virtualPods
  );
  patterns.add<ResolveVirtualPodReadOp>(ctx, virtualPods);

  ConversionTarget target(*ctx);
  baseTargetSetup(target);
  target.addDynamicallyLegalOp<NewPodOp>(SplitInitFromNewPodOp::legal);
  target.addDynamicallyLegalOp<FuncDefOp>(SplitPodInFuncDefOp::legal);
  target.addDynamicallyLegalOp<ReturnOp>(SplitPodInReturnOp::legal);
  target.addDynamicallyLegalOp<CallOp>(SplitPodInCallOp::legal);
  target.addDynamicallyLegalOp<MemberWriteOp>(SplitPodInMemberWriteOp::legal);
  target.addDynamicallyLegalOp<MemberReadOp>(SplitPodInMemberReadOp::legal);
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
      materializeVirtualPod(newPod, leafValues, builder);
    }
  }
  return success();
}

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

/// Return whether `op` contains any read from `podRef.recordName`.
static bool hasReadFromRecord(Operation &op, Value podRef, StringAttr recordName) {
  return walkContainsMatch<ReadPodOp>(op, [&](ReadPodOp readOp) {
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

/// Return whether the read is preceded by a write to the same pod record within its block.
static bool hasEarlierWriteInBlock(ReadPodOp readOp) {
  Value podRef = readOp.getPodRef();
  StringAttr recordName = readOp.getRecordNameAttr();

  for (Operation &op : *readOp->getBlock()) {
    if (&op == readOp.getOperation()) {
      return false;
    }

    if (auto writeOp = dyn_cast<WritePodOp>(&op)) {
      if (isSamePodRecord(writeOp, podRef, recordName)) {
        return true;
      }
      continue;
    }

    if (hasNestedWriteToRecord(op, podRef, recordName)) {
      return true;
    }
  }
  return false;
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
        readOp, genRead(readOp.getLoc(), readOp.getPodRef(), readOp.getRecordNameAttr(), rewriter)
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
    Location loc, Block &block, ValueRange priorYieldValues, ArrayRef<IfWriteSlot> slots,
    bool isThenBlock, OpBuilder &builder
) {
  SmallVector<Value> yieldValues = llvm::to_vector(priorYieldValues);
  llvm::append_range(yieldValues, llvm::map_range(slots, [isThenBlock](const IfWriteSlot &slot) {
    WritePodOp writeOp = isThenBlock ? slot.thenWrite : slot.elseWrite;
    return writeOp ? writeOp.getValue() : slot.incomingValue;
  }));

  builder.setInsertionPointToEnd(&block);
  builder.create<scf::YieldOp>(loc, yieldValues);
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
  auto it = llvm::find_if(slots, [&podRef, &recordName](const LoopPodSlot &slot) {
    return slot.matches(podRef, recordName);
  });
  return it == slots.end() ? nullptr : &*it;
}

/// Return whether a loop slot is tracked for `podRef.recordName`.
static bool hasLoopSlot(ArrayRef<LoopPodSlot> slots, Value podRef, StringAttr recordName) {
  auto it = llvm::find_if(slots, [&podRef, &recordName](const LoopPodSlot &slot) {
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
          genRead(ifOp.getLoc(), slot.podRef, slot.recordName, rewriter).getResult();
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
    appendYield(ifOp.getLoc(), newThenBlock, originalThenYields, slots, true, rewriter);
    appendYield(ifOp.getLoc(), newElseBlock, originalElseYields, slots, false, rewriter);

    rewriter.setInsertionPointAfter(newIf);
    unsigned originalResultCount = ifOp.getNumResults();
    for (auto [idx, slot] : llvm::enumerate(slots)) {
      genWrite(
          ifOp.getLoc(), slot.podRef, slot.recordName, newIf.getResult(originalResultCount + idx),
          rewriter
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
      newInitArgs.push_back(genRead(loc, slot.podRef, slot.recordName, rewriter).getResult());
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
          loc, slot.podRef, slot.recordName, newFor.getResult(forOp.getNumResults() + idx), rewriter
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
      newInits.push_back(genRead(loc, slot.podRef, slot.recordName, rewriter).getResult());
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
          loc, slot.podRef, slot.recordName, newWhile.getResult(whileOp.getNumResults() + idx),
          rewriter
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
