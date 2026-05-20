//===-- LLZKWitgenLoweringPass.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "WitgenLowering.h"

#include "WitgenDriver.h"
#include "WitgenUtils.h"
#include "WitnessSelection.h"

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Attrs.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/Compare.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/Field.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/TypeSwitch.h>

#include <limits>

using namespace mlir;

namespace llzk::witgen {
namespace {

/// Hold the flattened lowered SSA values for one original LLZK value.
struct LoweredValue {
  Type sourceType;
  llvm::SmallVector<Value> leaves;
};

/// Return the only field used by the module or emit an error.
static FailureOr<std::reference_wrapper<const Field>> getModuleField(ModuleOp moduleOp) {
  FieldSet fields;
  if (failed(collectFields(moduleOp.getOperation(), fields, false))) {
    moduleOp.emitError("failed to collect fields for llzk-witgen lowering");
    return failure();
  }
  if (fields.size() != 1) {
    moduleOp.emitError("llzk-witgen execution-engine lowering requires exactly one field");
    return failure();
  }
  return *fields.begin();
}

/// Return a sanitized symbol name for one lowered helper or function.
static std::string mangleFunctionName(function::FuncDefOp funcOp) {
  auto symbolRef = funcOp.getFullyQualifiedName(false);
  llvm::SmallString<128> result("__llzk_witgen_");
  for (StringRef piece : getNames(symbolRef)) {
    if (!result.empty() && result.back() != '_') {
      result += "__";
    }
    for (char c : piece) {
      result += llvm::isAlnum(static_cast<unsigned char>(c)) ? c : '_';
    }
  }
  return std::string(result);
}

/// Return a constant index value.
static Value makeIndexConstant(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantIndexOp>(loc, value).getResult();
}

/// Return a one constant of the lowered field integer type.
static Value makeOneFelt(OpBuilder &builder, Location loc, const Field &field) {
  return builder.create<arith::ConstantOp>(
      loc, IntegerAttr::get(IntegerType::get(builder.getContext(), field.bitWidth()), 1)
  );
}

/// Return the lowered scalar type for a non-aggregate LLZK type.
static FailureOr<Type> lowerScalarType(MLIRContext *context, Type type, const Field &field) {
  if (isa<felt::FeltType>(type)) {
    return IntegerType::get(context, field.bitWidth());
  }
  if (isa<IndexType>(type)) {
    return type;
  }
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.getWidth() == 1) {
      return intType;
    }
  }
  return failure();
}

/// Return true iff the type lowers as a scalar SSA value.
static bool isScalarType(Type type) {
  return isa<felt::FeltType, IndexType>(type) ||
         (isa<IntegerType>(type) && mlir::cast<IntegerType>(type).getWidth() == 1);
}

/// Flatten one LLZK type into its ordered lowered leaf types.
static LogicalResult flattenTypeLeaves(
    Type type, SymbolTableCollection &tables, Operation *origin, const Field &field,
    SmallVectorImpl<Type> &out, llvm::ArrayRef<int64_t> prefixShape = {}, bool storage = false
) {
  auto emitScalarLeaf = [&](Type leafType) {
    auto lowered = lowerScalarType(origin->getContext(), leafType, field);
    if (failed(lowered)) {
      return failure();
    }
    if (!storage && prefixShape.empty()) {
      out.push_back(*lowered);
      return success();
    }
    llvm::SmallVector<int64_t> shape(prefixShape.begin(), prefixShape.end());
    if (shape.empty()) {
      shape.push_back(1);
    }
    out.push_back(MemRefType::get(shape, *lowered));
    return success();
  };

  if (isScalarType(type)) {
    return emitScalarLeaf(type);
  }

  if (auto arrayType = dyn_cast<array::ArrayType>(type)) {
    llvm::SmallVector<int64_t> newPrefix(prefixShape.begin(), prefixShape.end());
    newPrefix.append(arrayType.getShape().begin(), arrayType.getShape().end());
    return flattenTypeLeaves(
        arrayType.getElementType(), tables, origin, field, out, newPrefix, true
    );
  }

  if (auto podType = dyn_cast<pod::PodType>(type)) {
    for (pod::RecordAttr record : podType.getRecords()) {
      if (failed(
              flattenTypeLeaves(record.getType(), tables, origin, field, out, prefixShape, true)
          )) {
        return failure();
      }
    }
    return success();
  }

  if (auto structType = dyn_cast<component::StructType>(type)) {
    auto def = structType.getDefinition(tables, origin);
    if (failed(def)) {
      origin->emitError("could not resolve struct type during witgen lowering");
      return failure();
    }
    for (component::MemberDefOp member : def->get().getMemberDefs()) {
      if (failed(
              flattenTypeLeaves(member.getType(), tables, origin, field, out, prefixShape, true)
          )) {
        return failure();
      }
    }
    return success();
  }

  origin->emitError("unsupported type in llzk-witgen lowering: ") << type;
  return failure();
}

/// Return a strided memref type for one internal aggregate ABI leaf.
static MemRefType
getStridedMemRefType(MLIRContext *context, ArrayRef<int64_t> shape, Type elementType) {
  SmallVector<int64_t> strides(shape.size(), ShapedType::kDynamic);
  return MemRefType::get(
      shape, elementType, StridedLayoutAttr::get(context, ShapedType::kDynamic, strides)
  );
}

/// Flatten one LLZK type into its ordered internal aggregate ABI leaf types.
static LogicalResult flattenABILeafTypes(
    Type type, SymbolTableCollection &tables, Operation *origin, const Field &field,
    SmallVectorImpl<Type> &out, size_t prefixRank = 0, bool aggregateStorage = false
) {
  auto emitScalarLeaf = [&](Type leafType) {
    auto lowered = lowerScalarType(origin->getContext(), leafType, field);
    if (failed(lowered)) {
      return failure();
    }
    if (!aggregateStorage && prefixRank == 0) {
      out.push_back(*lowered);
      return success();
    }
    SmallVector<int64_t> shape;
    if (prefixRank == 0) {
      shape.push_back(1);
    } else {
      shape.assign(prefixRank, ShapedType::kDynamic);
    }
    out.push_back(getStridedMemRefType(origin->getContext(), shape, *lowered));
    return success();
  };

  if (isScalarType(type)) {
    return emitScalarLeaf(type);
  }

  if (auto arrayType = dyn_cast<array::ArrayType>(type)) {
    return flattenABILeafTypes(
        arrayType.getElementType(), tables, origin, field, out, prefixRank + arrayType.getRank(),
        true
    );
  }

  if (auto podType = dyn_cast<pod::PodType>(type)) {
    for (pod::RecordAttr record : podType.getRecords()) {
      if (failed(
              flattenABILeafTypes(record.getType(), tables, origin, field, out, prefixRank, true)
          )) {
        return failure();
      }
    }
    return success();
  }

  if (auto structType = dyn_cast<component::StructType>(type)) {
    auto def = structType.getDefinition(tables, origin);
    if (failed(def)) {
      origin->emitError("could not resolve struct type during witgen lowering");
      return failure();
    }
    for (component::MemberDefOp member : def->get().getMemberDefs()) {
      if (failed(
              flattenABILeafTypes(member.getType(), tables, origin, field, out, prefixRank, true)
          )) {
        return failure();
      }
    }
    return success();
  }

  origin->emitError("unsupported type in llzk-witgen lowering: ") << type;
  return failure();
}

/// Return the number of flattened lowered leaves for one LLZK type.
static FailureOr<size_t>
getLeafCount(Type type, SymbolTableCollection &tables, Operation *origin, const Field &field) {
  SmallVector<Type> leaves;
  if (failed(flattenTypeLeaves(type, tables, origin, field, leaves))) {
    return failure();
  }
  return leaves.size();
}

/// Return the lowered leaf types for one LLZK type.
static FailureOr<SmallVector<Type>>
getLeafTypes(Type type, SymbolTableCollection &tables, Operation *origin, const Field &field) {
  SmallVector<Type> leaves;
  if (failed(flattenTypeLeaves(type, tables, origin, field, leaves))) {
    return failure();
  }
  return leaves;
}

/// Return the internal aggregate ABI leaf types for one LLZK type.
static FailureOr<SmallVector<Type>>
getABILeafTypes(Type type, SymbolTableCollection &tables, Operation *origin, const Field &field) {
  SmallVector<Type> leaves;
  if (failed(flattenABILeafTypes(type, tables, origin, field, leaves))) {
    return failure();
  }
  return leaves;
}

/// Return the ordered member span for one pod record or struct member.
static FailureOr<std::pair<size_t, size_t>> getNamedLeafSpan(
    Type ownerType, StringRef name, SymbolTableCollection &tables, Operation *origin,
    const Field &field
) {
  if (auto podType = dyn_cast<pod::PodType>(ownerType)) {
    size_t running = 0;
    for (pod::RecordAttr record : podType.getRecords()) {
      auto count = getLeafCount(record.getType(), tables, origin, field);
      if (failed(count)) {
        return failure();
      }
      if (record.getName().getValue() == name) {
        return std::pair<size_t, size_t> {running, *count};
      }
      running += *count;
    }
  }

  if (auto structType = dyn_cast<component::StructType>(ownerType)) {
    auto def = structType.getDefinition(tables, origin);
    if (failed(def)) {
      origin->emitError("could not resolve struct type during witgen lowering");
      return failure();
    }
    size_t running = 0;
    for (component::MemberDefOp member : def->get().getMemberDefs()) {
      auto count = getLeafCount(member.getType(), tables, origin, field);
      if (failed(count)) {
        return failure();
      }
      if (member.getSymName() == name) {
        return std::pair<size_t, size_t> {running, *count};
      }
      running += *count;
    }
  }

  origin->emitError("could not resolve aggregate member/record @") << name;
  return failure();
}

/// Return the type of one pod record or struct member.
static FailureOr<Type>
getNamedSubType(Type ownerType, StringRef name, SymbolTableCollection &tables, Operation *origin) {
  if (auto podType = dyn_cast<pod::PodType>(ownerType)) {
    for (pod::RecordAttr record : podType.getRecords()) {
      if (record.getName().getValue() == name) {
        return record.getType();
      }
    }
  }
  if (auto structType = dyn_cast<component::StructType>(ownerType)) {
    auto def = structType.getDefinition(tables, origin);
    if (failed(def)) {
      origin->emitError("could not resolve struct type during witgen lowering");
      return failure();
    }
    for (component::MemberDefOp member : def->get().getMemberDefs()) {
      if (member.getSymName() == name) {
        return member.getType();
      }
    }
  }
  origin->emitError("could not resolve aggregate member/record @") << name;
  return failure();
}

/// Create one static memref filled with zeros.
static FailureOr<Value>
createZeroMemRef(OpBuilder &builder, Location loc, MemRefType memrefType, const Field &field) {
  auto elementCount = getStaticElementCount(memrefType, "witgen zero memref");
  if (!elementCount) {
    emitError(loc) << llvm::toString(elementCount.takeError());
    return failure();
  }
  Value alloc = builder.create<memref::AllocOp>(loc, memrefType);
  auto elementType = memrefType.getElementType();
  Value zero;
  if (isa<IndexType>(elementType)) {
    zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  } else {
    zero = builder.create<arith::ConstantOp>(
        loc, IntegerAttr::get(mlir::cast<IntegerType>(elementType), 0)
    );
  }
  auto strides = mlir::computeStrides(memrefType.getShape());
  for (size_t flat = 0; flat < *elementCount; ++flat) {
    SmallVector<Value> indices;
    for (int64_t index : mlir::delinearize(flat, strides)) {
      indices.push_back(makeIndexConstant(builder, loc, index));
    }
    builder.create<memref::StoreOp>(loc, zero, alloc, indices);
  }
  return alloc;
}

/// Create one static memref filled with random in-range values.
static FailureOr<Value> createRandomMemRef(
    OpBuilder &builder, Location loc, MemRefType memrefType, const Field &field,
    std::mt19937_64 &rng
) {
  auto elementCount = getStaticElementCount(memrefType, "witgen random memref");
  if (!elementCount) {
    emitError(loc) << llvm::toString(elementCount.takeError());
    return failure();
  }
  Value alloc = builder.create<memref::AllocOp>(loc, memrefType);
  auto elementType = memrefType.getElementType();
  auto strides = mlir::computeStrides(memrefType.getShape());
  for (size_t flat = 0; flat < *elementCount; ++flat) {
    SmallVector<Value> indices;
    for (int64_t index : mlir::delinearize(flat, strides)) {
      indices.push_back(makeIndexConstant(builder, loc, index));
    }
    if (isa<IndexType>(elementType)) {
      auto value = randomIndexValue(rng);
      builder.create<memref::StoreOp>(
          loc, builder.create<arith::ConstantIndexOp>(loc, value), alloc, indices
      );
      continue;
    }
    auto intType = mlir::cast<IntegerType>(elementType);
    if (intType.getWidth() == 1) {
      builder.create<memref::StoreOp>(
          loc,
          builder.create<arith::ConstantOp>(
              loc, IntegerAttr::get(intType, APInt(1, randomBoolValue(rng)))
          ),
          alloc, indices
      );
      continue;
    }
    auto candidate = randomFieldElement(rng, field);
    builder.create<memref::StoreOp>(
        loc,
        builder.create<arith::ConstantOp>(
            loc, IntegerAttr::get(intType, toAPSInt(candidate).trunc(intType.getWidth()))
        ),
        alloc, indices
    );
  }
  return alloc;
}

/// Build the default lowered value for one LLZK type.
static FailureOr<LoweredValue> createDefaultValue(
    OpBuilder &builder, Location loc, Type type, SymbolTableCollection &tables, Operation *origin,
    const Field &field, UninitializedBehavior behavior, std::mt19937_64 &rng
) {
  LoweredValue lowered {type, {}};
  auto leafTypes = getLeafTypes(type, tables, origin, field);
  if (failed(leafTypes)) {
    return failure();
  }
  for (Type leafType : *leafTypes) {
    if (behavior == UninitializedBehavior::Fail) {
      origin->emitError(
          "fail-mode default materialization is unsupported in witgen lowering because it would "
          "hide uninitialized reads"
      );
      return failure();
    }
    if (behavior == UninitializedBehavior::Random) {
      if (auto memrefType = dyn_cast<MemRefType>(leafType)) {
        auto randomMemRef = createRandomMemRef(builder, loc, memrefType, field, rng);
        if (failed(randomMemRef)) {
          return failure();
        }
        lowered.leaves.push_back(*randomMemRef);
        continue;
      }
      if (isa<IndexType>(leafType)) {
        lowered.leaves.push_back(
            builder.create<arith::ConstantIndexOp>(loc, randomIndexValue(rng))
        );
        continue;
      }
      auto intType = mlir::cast<IntegerType>(leafType);
      if (intType.getWidth() == 1) {
        lowered.leaves.push_back(builder.create<arith::ConstantOp>(
            loc, IntegerAttr::get(intType, APInt(1, randomBoolValue(rng)))
        ));
        continue;
      }
      auto candidate = randomFieldElement(rng, field);
      lowered.leaves.push_back(builder.create<arith::ConstantOp>(
          loc, IntegerAttr::get(intType, toAPSInt(candidate).trunc(intType.getWidth()))
      ));
      continue;
    }
    if (auto memrefType = dyn_cast<MemRefType>(leafType)) {
      auto zeroMemRef = createZeroMemRef(builder, loc, memrefType, field);
      if (failed(zeroMemRef)) {
        return failure();
      }
      lowered.leaves.push_back(*zeroMemRef);
      continue;
    }
    if (isa<IndexType>(leafType)) {
      lowered.leaves.push_back(builder.create<arith::ConstantIndexOp>(loc, 0));
      continue;
    }
    lowered.leaves.push_back(builder.create<arith::ConstantOp>(
        loc, IntegerAttr::get(mlir::cast<IntegerType>(leafType), 0)
    ));
  }
  return lowered;
}

/// Normalize a widened integer back into the field modulus and truncate it.
static Value normalizeWideValue(
    OpBuilder &builder, Location loc, Value wideValue, unsigned dstWidth, const Field &field
) {
  auto wideType = mlir::cast<IntegerType>(wideValue.getType());
  Value modulus = builder.create<arith::ConstantOp>(
      loc, field.getPrimeAttr(builder.getContext(), wideType.getWidth())
  );
  Value reduced = builder.create<arith::RemUIOp>(loc, wideValue, modulus);
  return builder.create<arith::TruncIOp>(
      loc, IntegerType::get(builder.getContext(), dstWidth), reduced
  );
}

/// Lower field addition with explicit modular reduction.
static Value
lowerFeltAdd(OpBuilder &builder, Location loc, Value lhs, Value rhs, const Field &field) {
  unsigned width = field.bitWidth();
  unsigned wideWidth = width + 1;
  auto wideType = IntegerType::get(builder.getContext(), wideWidth);
  Value lhsWide = builder.create<arith::ExtUIOp>(loc, wideType, lhs);
  Value rhsWide = builder.create<arith::ExtUIOp>(loc, wideType, rhs);
  Value sum = builder.create<arith::AddIOp>(loc, lhsWide, rhsWide);
  return normalizeWideValue(builder, loc, sum, width, field);
}

/// Lower field subtraction with explicit modular reduction.
static Value
lowerFeltSub(OpBuilder &builder, Location loc, Value lhs, Value rhs, const Field &field) {
  unsigned width = field.bitWidth();
  unsigned wideWidth = width + 1;
  auto wideType = IntegerType::get(builder.getContext(), wideWidth);
  Value lhsWide = builder.create<arith::ExtUIOp>(loc, wideType, lhs);
  Value rhsWide = builder.create<arith::ExtUIOp>(loc, wideType, rhs);
  Value modulus =
      builder.create<arith::ConstantOp>(loc, field.getPrimeAttr(builder.getContext(), wideWidth));
  Value lhsPlusMod = builder.create<arith::AddIOp>(loc, lhsWide, modulus);
  Value diff = builder.create<arith::SubIOp>(loc, lhsPlusMod, rhsWide);
  return normalizeWideValue(builder, loc, diff, width, field);
}

/// Lower field negation with explicit modular reduction.
static Value lowerFeltNeg(OpBuilder &builder, Location loc, Value operand, const Field &field) {
  unsigned width = field.bitWidth();
  unsigned wideWidth = width + 1;
  auto wideType = IntegerType::get(builder.getContext(), wideWidth);
  Value operandWide = builder.create<arith::ExtUIOp>(loc, wideType, operand);
  Value modulus =
      builder.create<arith::ConstantOp>(loc, field.getPrimeAttr(builder.getContext(), wideWidth));
  Value diff = builder.create<arith::SubIOp>(loc, modulus, operandWide);
  return normalizeWideValue(builder, loc, diff, width, field);
}

/// Lower field multiplication with widening and modular reduction.
static Value
lowerFeltMul(OpBuilder &builder, Location loc, Value lhs, Value rhs, const Field &field) {
  unsigned width = field.bitWidth();
  unsigned wideWidth = width * 2;
  auto wideType = IntegerType::get(builder.getContext(), wideWidth);
  Value lhsWide = builder.create<arith::ExtUIOp>(loc, wideType, lhs);
  Value rhsWide = builder.create<arith::ExtUIOp>(loc, wideType, rhs);
  Value product = builder.create<arith::MulIOp>(loc, lhsWide, rhsWide);
  return normalizeWideValue(builder, loc, product, width, field);
}

/// Lower a field inversion through exponentiation by `p - 2`.
static Value lowerFeltInv(OpBuilder &builder, Location loc, Value operand, const Field &field) {
  llvm::APInt exponent = toExactWidthAPInt(field.prime() - 2, field.bitWidth());
  Value result = makeOneFelt(builder, loc, field);
  Value base = operand;
  for (unsigned bit = 0; bit < exponent.getBitWidth(); ++bit) {
    if (exponent[bit]) {
      result = lowerFeltMul(builder, loc, result, base, field);
    }
    if (bit + 1 < exponent.getBitWidth()) {
      base = lowerFeltMul(builder, loc, base, base, field);
    }
  }
  return result;
}

/// Lower field division as multiplication by the modular inverse.
static Value
lowerFeltDiv(OpBuilder &builder, Location loc, Value lhs, Value rhs, const Field &field) {
  return lowerFeltMul(builder, loc, lhs, lowerFeltInv(builder, loc, rhs, field), field);
}

/// Load one scalar leaf from aggregate storage.
static Value loadStorageScalar(OpBuilder &builder, Location loc, Value storageLeaf) {
  auto memrefType = mlir::cast<MemRefType>(storageLeaf.getType());
  SmallVector<Value> indices;
  indices.reserve(memrefType.getRank());
  for (int64_t dim = 0; dim < memrefType.getRank(); ++dim) {
    indices.push_back(makeIndexConstant(builder, loc, 0));
  }
  return builder.create<memref::LoadOp>(loc, storageLeaf, indices);
}

/// Store one scalar value into aggregate storage.
static void storeStorageScalar(OpBuilder &builder, Location loc, Value scalar, Value storageLeaf) {
  auto memrefType = mlir::cast<MemRefType>(storageLeaf.getType());
  SmallVector<Value> indices;
  indices.reserve(memrefType.getRank());
  for (int64_t dim = 0; dim < memrefType.getRank(); ++dim) {
    indices.push_back(makeIndexConstant(builder, loc, 0));
  }
  builder.create<memref::StoreOp>(loc, scalar, storageLeaf, indices);
}

/// Copy the flattened source value into aggregate storage leaves.
static LogicalResult copyIntoStorage(
    OpBuilder &builder, Location loc, Type sourceType, ArrayRef<Value> destLeaves,
    ArrayRef<Value> sourceLeaves, SymbolTableCollection &tables, Operation *origin,
    const Field &field
) {
  auto leafTypes = getLeafTypes(sourceType, tables, origin, field);
  if (failed(leafTypes)) {
    return failure();
  }
  if (destLeaves.size() != sourceLeaves.size() || destLeaves.size() != leafTypes->size()) {
    origin->emitError("flattened leaf mismatch while copying aggregate storage");
    return failure();
  }
  for (auto [leafType, destLeaf, srcLeaf] : llvm::zip(*leafTypes, destLeaves, sourceLeaves)) {
    if (isa<MemRefType>(leafType)) {
      builder.create<memref::CopyOp>(loc, srcLeaf, destLeaf);
      continue;
    }
    storeStorageScalar(builder, loc, srcLeaf, destLeaf);
  }
  return success();
}

/// Return the flattened lowered sub-value slice for one member or record.
static FailureOr<LoweredValue> readNamedAggregateValue(
    OpBuilder &builder, Location loc, Type ownerType, StringRef name, const LoweredValue &owner,
    SymbolTableCollection &tables, Operation *origin, const Field &field
) {
  auto subType = getNamedSubType(ownerType, name, tables, origin);
  if (failed(subType)) {
    return failure();
  }
  auto span = getNamedLeafSpan(ownerType, name, tables, origin, field);
  if (failed(span)) {
    return failure();
  }
  LoweredValue result {*subType, {}};
  auto leafTypes = getLeafTypes(*subType, tables, origin, field);
  if (failed(leafTypes)) {
    return failure();
  }
  auto leaves = ArrayRef<Value>(owner.leaves).slice(span->first, span->second);
  for (auto [leafType, leafValue] : llvm::zip(*leafTypes, leaves)) {
    if (isa<MemRefType>(leafType)) {
      result.leaves.push_back(leafValue);
    } else {
      result.leaves.push_back(loadStorageScalar(builder, loc, leafValue));
    }
  }
  return result;
}

/// Write one member or record into aggregate storage.
static LogicalResult writeNamedAggregateValue(
    OpBuilder &builder, Location loc, Type ownerType, StringRef name, LoweredValue &owner,
    const LoweredValue &value, SymbolTableCollection &tables, Operation *origin, const Field &field
) {
  auto subType = getNamedSubType(ownerType, name, tables, origin);
  if (failed(subType)) {
    return failure();
  }
  auto span = getNamedLeafSpan(ownerType, name, tables, origin, field);
  if (failed(span)) {
    return failure();
  }
  return copyIntoStorage(
      builder, loc, *subType, ArrayRef<Value>(owner.leaves).slice(span->first, span->second),
      value.leaves, tables, origin, field
  );
}

/// Create a static subview representing one aggregate array element leaf.
static Value
createElementSubview(OpBuilder &builder, Location loc, Value source, ValueRange outerIndices) {
  auto sourceType = mlir::cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> mixedOffsets;
  SmallVector<OpFoldResult> mixedSizes;
  SmallVector<OpFoldResult> mixedStrides;
  const int64_t indexedRank = llzk::checkedCast<int64_t>(outerIndices.size());
  mixedOffsets.reserve(sourceType.getRank());
  mixedSizes.reserve(sourceType.getRank());
  mixedStrides.reserve(sourceType.getRank());
  for (Value index : outerIndices) {
    mixedOffsets.push_back(index);
  }
  for (int64_t dim = indexedRank; dim < sourceType.getRank(); ++dim) {
    mixedOffsets.push_back(builder.getIndexAttr(0));
  }
  for (int64_t dim = 0; dim < indexedRank; ++dim) {
    mixedSizes.push_back(builder.getIndexAttr(1));
  }
  for (int64_t dim = indexedRank; dim < sourceType.getRank(); ++dim) {
    mixedSizes.push_back(memref::getMixedSize(builder, loc, source, dim));
  }
  for (int64_t dim = 0; dim < sourceType.getRank(); ++dim) {
    mixedStrides.push_back(builder.getIndexAttr(1));
  }
  SmallVector<int64_t> desiredShape;
  desiredShape.reserve(llzk::checkedCast<size_t>(sourceType.getRank() - indexedRank));
  for (int64_t dim = indexedRank; dim < sourceType.getRank(); ++dim) {
    if (auto attr = llvm::dyn_cast<Attribute>(mixedSizes[llzk::checkedCast<size_t>(dim)])) {
      desiredShape.push_back(mlir::cast<IntegerAttr>(attr).getInt());
    } else {
      desiredShape.push_back(ShapedType::kDynamic);
    }
  }
  if (desiredShape.empty()) {
    desiredShape.push_back(1);
  }
  auto resultType = mlir::cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
      desiredShape, sourceType, mixedOffsets, mixedSizes, mixedStrides
  ));
  return builder.create<memref::SubViewOp>(
      loc, resultType, source, mixedOffsets, mixedSizes, mixedStrides
  );
}

/// Read one LLZK array element.
static FailureOr<LoweredValue> readArrayElement(
    OpBuilder &builder, Location loc, array::ArrayType arrayType, const LoweredValue &arrayValue,
    ArrayRef<Value> indices, SymbolTableCollection &tables, Operation *origin, const Field &field
) {
  Type elementType = arrayType.getElementType();
  LoweredValue result {elementType, {}};
  if (isScalarType(elementType)) {
    result.leaves.push_back(
        builder.create<memref::LoadOp>(loc, arrayValue.leaves.front(), indices)
    );
    return result;
  }

  for (Value sourceLeaf : arrayValue.leaves) {
    result.leaves.push_back(createElementSubview(builder, loc, sourceLeaf, indices));
  }
  return result;
}

/// Write one LLZK array element.
static LogicalResult writeArrayElement(
    OpBuilder &builder, Location loc, array::ArrayType arrayType, LoweredValue &arrayValue,
    ArrayRef<Value> indices, const LoweredValue &elementValue, SymbolTableCollection &tables,
    Operation *origin, const Field &field
) {
  Type elementType = arrayType.getElementType();
  if (isScalarType(elementType)) {
    builder.create<memref::StoreOp>(
        loc, elementValue.leaves.front(), arrayValue.leaves.front(), indices
    );
    return success();
  }

  for (auto [destLeaf, srcLeaf] : llvm::zip(arrayValue.leaves, elementValue.leaves)) {
    Value subview = createElementSubview(builder, loc, destLeaf, indices);
    builder.create<memref::CopyOp>(loc, srcLeaf, subview);
  }
  return success();
}

/// Flatten one lowered value into one target leaf type list.
static LogicalResult appendFlatLeavesToTypes(
    OpBuilder &builder, Location loc, const LoweredValue &value, ArrayRef<Type> targetLeafTypes,
    SmallVectorImpl<Value> &out, Operation *origin
) {
  if (targetLeafTypes.size() != value.leaves.size()) {
    origin->emitError("flattened leaf mismatch during call lowering");
    return failure();
  }
  for (auto [leafValue, leafType] : llvm::zip(value.leaves, targetLeafTypes)) {
    if (leafValue.getType() == leafType) {
      out.push_back(leafValue);
      continue;
    }
    if (isa<MemRefType>(leafValue.getType()) && isa<MemRefType>(leafType)) {
      out.push_back(builder.create<memref::CastOp>(loc, leafType, leafValue));
      continue;
    }
    origin->emitError("lowered leaf type mismatch during call lowering");
    return failure();
  }
  return success();
}

/// Lower one LLZK compute/free function body into `func.func`.
class BodyLowerer {
public:
  /// Create a lowerer that appends new `func.func` operations into the module.
  BodyLowerer(
      ModuleOp mod, SymbolTableCollection &symbolTables, const Field &moduleField,
      const WitgenOptions &options
  )
      : moduleOp(mod), tables(symbolTables), field(moduleField),
        uninitializedBehavior(options.uninitializedBehavior), rng(makeDefaultValueRng(options)) {}

  /// Lower one LLZK function into `func.func`.
  FailureOr<func::FuncOp> lowerFunction(function::FuncDefOp funcOp) {
    if (funcOp.isExternal()) {
      funcOp.emitError("execution-engine backend does not lower extern functions");
      return failure();
    }
    if (!funcOp.getBody().hasOneBlock()) {
      funcOp.emitError("execution-engine backend only supports single-block functions");
      return failure();
    }

    SmallVector<Type> loweredArgTypes;
    for (Type argType : funcOp.getArgumentTypes()) {
      if (failed(
              flattenABILeafTypes(argType, tables, funcOp.getOperation(), field, loweredArgTypes)
          )) {
        return failure();
      }
    }
    SmallVector<Type> loweredResultTypes;
    for (Type resultType : funcOp.getResultTypes()) {
      if (failed(flattenABILeafTypes(
              resultType, tables, funcOp.getOperation(), field, loweredResultTypes
          ))) {
        return failure();
      }
    }

    OpBuilder moduleBuilder(moduleOp.getContext());
    moduleBuilder.setInsertionPointToEnd(moduleOp.getBody());
    auto loweredFunc = moduleBuilder.create<func::FuncOp>(
        funcOp.getLoc(), mangleFunctionName(funcOp),
        moduleBuilder.getFunctionType(loweredArgTypes, loweredResultTypes)
    );
    Block *entry = loweredFunc.addEntryBlock();
    OpBuilder builder(entry, entry->begin());

    DenseMap<Value, LoweredValue> valueMap;
    unsigned cursor = 0;
    for (auto [arg, argType] :
         llvm::zip(funcOp.getBody().front().getArguments(), funcOp.getArgumentTypes())) {
      auto leafCount = getLeafCount(argType, tables, funcOp.getOperation(), field);
      if (failed(leafCount)) {
        loweredFunc.erase();
        return failure();
      }
      LoweredValue lowered {argType, {}};
      lowered.leaves.append(
          entry->getArguments().begin() + cursor,
          entry->getArguments().begin() + cursor + *leafCount
      );
      cursor += *leafCount;
      valueMap[arg] = std::move(lowered);
    }

    if (failed(lowerBlock(builder, funcOp.getBody().front(), valueMap))) {
      loweredFunc.erase();
      return failure();
    }
    return loweredFunc;
  }

private:
  ModuleOp moduleOp;
  SymbolTableCollection &tables;
  const Field &field;
  UninitializedBehavior uninitializedBehavior;
  std::mt19937_64 rng;

  /// Look up one already-lowered SSA value.
  FailureOr<LoweredValue>
  lookup(Value value, DenseMap<Value, LoweredValue> &valueMap, Operation *origin) {
    auto it = valueMap.find(value);
    if (it == valueMap.end()) {
      origin->emitError("failed to find lowered SSA value");
      return failure();
    }
    return it->second;
  }

  /// Require a scalar lowered value.
  FailureOr<Value>
  lookupScalar(Value value, DenseMap<Value, LoweredValue> &valueMap, Operation *origin) {
    auto lowered = lookup(value, valueMap, origin);
    if (failed(lowered) || lowered->leaves.size() != 1 ||
        isa<MemRefType>(lowered->leaves.front().getType())) {
      origin->emitError("expected scalar lowered value");
      return failure();
    }
    return lowered->leaves.front();
  }

  /// Lower every operation in a single block.
  LogicalResult
  lowerBlock(OpBuilder &builder, Block &block, DenseMap<Value, LoweredValue> &valueMap) {
    for (Operation &op : block) {
      if (failed(lowerOperation(builder, op, valueMap))) {
        return failure();
      }
    }
    return success();
  }

  /// Lower one field comparison predicate.
  FailureOr<Value>
  lowerFeltCmp(OpBuilder &builder, Location loc, boolean::CmpOp cmpOp, Value lhs, Value rhs) {
    arith::CmpIPredicate predicate;
    switch (cmpOp.getPredicate()) {
    case boolean::FeltCmpPredicate::EQ:
      predicate = arith::CmpIPredicate::eq;
      break;
    case boolean::FeltCmpPredicate::NE:
      predicate = arith::CmpIPredicate::ne;
      break;
    case boolean::FeltCmpPredicate::LT:
      predicate = arith::CmpIPredicate::ult;
      break;
    case boolean::FeltCmpPredicate::LE:
      predicate = arith::CmpIPredicate::ule;
      break;
    case boolean::FeltCmpPredicate::GT:
      predicate = arith::CmpIPredicate::ugt;
      break;
    case boolean::FeltCmpPredicate::GE:
      predicate = arith::CmpIPredicate::uge;
      break;
    }
    return builder.create<arith::CmpIOp>(loc, predicate, lhs, rhs).getResult();
  }

  /// Lower one LLZK operation into core MLIR dialects.
  LogicalResult
  lowerOperation(OpBuilder &builder, Operation &op, DenseMap<Value, LoweredValue> &valueMap) {
    Location loc = op.getLoc();

    auto bind = [&](Value result, LoweredValue lowered) {
      valueMap[result] = std::move(lowered);
      return success();
    };

    if (auto returnOp = dyn_cast<function::ReturnOp>(op)) {
      SmallVector<Value> results;
      for (Value operand : returnOp.getOperands()) {
        auto lowered = lookup(operand, valueMap, returnOp.getOperation());
        auto leafTypes = getABILeafTypes(operand.getType(), tables, returnOp.getOperation(), field);
        if (failed(lowered) || failed(leafTypes) ||
            failed(appendFlatLeavesToTypes(
                builder, loc, *lowered, *leafTypes, results, returnOp.getOperation()
            ))) {
          return failure();
        }
      }
      builder.create<func::ReturnOp>(loc, results);
      return success();
    }

    if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      SmallVector<Value> results;
      for (Value operand : yieldOp.getOperands()) {
        auto lowered = lookup(operand, valueMap, yieldOp.getOperation());
        auto leafTypes = getABILeafTypes(operand.getType(), tables, yieldOp.getOperation(), field);
        if (failed(lowered) || failed(leafTypes) ||
            failed(appendFlatLeavesToTypes(
                builder, loc, *lowered, *leafTypes, results, yieldOp.getOperation()
            ))) {
          return failure();
        }
      }
      builder.create<scf::YieldOp>(loc, results);
      return success();
    }

    if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
      Operation *clone = builder.clone(op);
      return bind(
          constantOp.getResult(), LoweredValue {constantOp.getType(), {clone->getResult(0)}}
      );
    }

    if (auto feltConst = dyn_cast<felt::FeltConstantOp>(op)) {
      auto intType = IntegerType::get(builder.getContext(), field.bitWidth());
      // Do a safe conversion to an APInt of the right bitwidth: mod prime before truncation
      auto constVal = toDynamicAPInt(feltConst.getValue().getValue());
      auto modVal = constVal % field.prime();
      auto intVal = toAPSInt(modVal);
      intVal.setIsUnsigned(true);
      intVal = intVal.trunc(field.bitWidth());
      Value lowered = builder.create<arith::ConstantOp>(loc, IntegerAttr::get(intType, intVal));
      return bind(feltConst.getResult(), LoweredValue {feltConst.getType(), {lowered}});
    }

    if (auto nondetOp = dyn_cast<llzk::NonDetOp>(op)) {
      auto lowered = createDefaultValue(
          builder, loc, nondetOp.getType(), tables, nondetOp.getOperation(), field,
          uninitializedBehavior, rng
      );
      if (failed(lowered)) {
        return failure();
      }
      return bind(nondetOp.getResult(), std::move(*lowered));
    }

    if (auto addOp = dyn_cast<felt::AddFeltOp>(op)) {
      auto lhs = lookupScalar(addOp.getLhs(), valueMap, addOp.getOperation());
      auto rhs = lookupScalar(addOp.getRhs(), valueMap, addOp.getOperation());
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      return bind(
          addOp.getResult(),
          LoweredValue {addOp.getType(), {lowerFeltAdd(builder, loc, *lhs, *rhs, field)}}
      );
    }
    if (auto subOp = dyn_cast<felt::SubFeltOp>(op)) {
      auto lhs = lookupScalar(subOp.getLhs(), valueMap, subOp.getOperation());
      auto rhs = lookupScalar(subOp.getRhs(), valueMap, subOp.getOperation());
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      return bind(
          subOp.getResult(),
          LoweredValue {subOp.getType(), {lowerFeltSub(builder, loc, *lhs, *rhs, field)}}
      );
    }
    if (auto mulOp = dyn_cast<felt::MulFeltOp>(op)) {
      auto lhs = lookupScalar(mulOp.getLhs(), valueMap, mulOp.getOperation());
      auto rhs = lookupScalar(mulOp.getRhs(), valueMap, mulOp.getOperation());
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      return bind(
          mulOp.getResult(),
          LoweredValue {mulOp.getType(), {lowerFeltMul(builder, loc, *lhs, *rhs, field)}}
      );
    }
    if (auto negOp = dyn_cast<felt::NegFeltOp>(op)) {
      auto operand = lookupScalar(negOp.getOperand(), valueMap, negOp.getOperation());
      if (failed(operand)) {
        return failure();
      }
      return bind(
          negOp.getResult(),
          LoweredValue {negOp.getType(), {lowerFeltNeg(builder, loc, *operand, field)}}
      );
    }
    if (auto invOp = dyn_cast<felt::InvFeltOp>(op)) {
      auto operand = lookupScalar(invOp.getOperand(), valueMap, invOp.getOperation());
      if (failed(operand)) {
        return failure();
      }
      return bind(
          invOp.getResult(),
          LoweredValue {invOp.getType(), {lowerFeltInv(builder, loc, *operand, field)}}
      );
    }
    if (auto divOp = dyn_cast<felt::DivFeltOp>(op)) {
      auto lhs = lookupScalar(divOp.getLhs(), valueMap, divOp.getOperation());
      auto rhs = lookupScalar(divOp.getRhs(), valueMap, divOp.getOperation());
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      return bind(
          divOp.getResult(),
          LoweredValue {divOp.getType(), {lowerFeltDiv(builder, loc, *lhs, *rhs, field)}}
      );
    }

    if (auto cmpOp = dyn_cast<boolean::CmpOp>(op)) {
      auto lhs = lookupScalar(cmpOp.getLhs(), valueMap, cmpOp.getOperation());
      auto rhs = lookupScalar(cmpOp.getRhs(), valueMap, cmpOp.getOperation());
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      auto lowered = lowerFeltCmp(builder, loc, cmpOp, *lhs, *rhs);
      if (failed(lowered)) {
        return failure();
      }
      return bind(cmpOp.getResult(), LoweredValue {cmpOp.getType(), {*lowered}});
    }
    if (auto assertOp = dyn_cast<boolean::AssertOp>(op)) {
      auto condition = lookupScalar(assertOp.getCondition(), valueMap, assertOp.getOperation());
      if (failed(condition)) {
        return failure();
      }
      builder.create<cf::AssertOp>(
          loc, *condition, assertOp.getMsg() ? assertOp.getMsg()->str() : "bool.assert failed"
      );
      return success();
    }
    if (auto andOp = dyn_cast<boolean::AndBoolOp>(op)) {
      auto lhs = lookupScalar(andOp.getLhs(), valueMap, andOp.getOperation());
      auto rhs = lookupScalar(andOp.getRhs(), valueMap, andOp.getOperation());
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      return bind(
          andOp.getResult(),
          LoweredValue {andOp.getType(), {builder.create<arith::AndIOp>(loc, *lhs, *rhs)}}
      );
    }
    if (auto orOp = dyn_cast<boolean::OrBoolOp>(op)) {
      auto lhs = lookupScalar(orOp.getLhs(), valueMap, orOp.getOperation());
      auto rhs = lookupScalar(orOp.getRhs(), valueMap, orOp.getOperation());
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      return bind(
          orOp.getResult(),
          LoweredValue {orOp.getType(), {builder.create<arith::OrIOp>(loc, *lhs, *rhs)}}
      );
    }
    if (auto xorOp = dyn_cast<boolean::XorBoolOp>(op)) {
      auto lhs = lookupScalar(xorOp.getLhs(), valueMap, xorOp.getOperation());
      auto rhs = lookupScalar(xorOp.getRhs(), valueMap, xorOp.getOperation());
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      return bind(
          xorOp.getResult(),
          LoweredValue {xorOp.getType(), {builder.create<arith::XOrIOp>(loc, *lhs, *rhs)}}
      );
    }
    if (auto notOp = dyn_cast<boolean::NotBoolOp>(op)) {
      auto operand = lookupScalar(notOp.getOperand(), valueMap, notOp.getOperation());
      if (failed(operand)) {
        return failure();
      }
      Value one = builder.create<arith::ConstantOp>(
          loc, IntegerAttr::get(IntegerType::get(builder.getContext(), 1), 1)
      );
      return bind(
          notOp.getResult(),
          LoweredValue {notOp.getType(), {builder.create<arith::XOrIOp>(loc, *operand, one)}}
      );
    }

    if (auto intToFelt = dyn_cast<cast::IntToFeltOp>(op)) {
      auto operand = lookupScalar(intToFelt.getValue(), valueMap, intToFelt.getOperation());
      if (failed(operand)) {
        return failure();
      }
      auto dstType = IntegerType::get(builder.getContext(), field.bitWidth());
      Value lowered;
      if (isa<IndexType>((*operand).getType())) {
        lowered = builder.create<arith::IndexCastUIOp>(loc, dstType, *operand);
      } else {
        auto intType = mlir::cast<IntegerType>((*operand).getType());
        if (intType.getWidth() < dstType.getWidth()) {
          lowered = builder.create<arith::ExtUIOp>(loc, dstType, *operand);
        } else if (intType.getWidth() > dstType.getWidth()) {
          lowered = normalizeWideValue(builder, loc, *operand, dstType.getWidth(), field);
        } else {
          lowered = *operand;
        }
      }
      return bind(intToFelt.getResult(), LoweredValue {intToFelt.getType(), {lowered}});
    }
    if (auto feltToIndex = dyn_cast<cast::FeltToIndexOp>(op)) {
      auto operand = lookupScalar(feltToIndex.getValue(), valueMap, feltToIndex.getOperation());
      if (failed(operand)) {
        return failure();
      }
      return bind(
          feltToIndex.getResult(),
          LoweredValue {
              feltToIndex.getType(),
              {builder.create<arith::IndexCastUIOp>(loc, builder.getIndexType(), *operand)}
          }
      );
    }

    if (auto structNewOp = dyn_cast<component::CreateStructOp>(op)) {
      auto lowered = createDefaultValue(
          builder, loc, structNewOp.getType(), tables, structNewOp.getOperation(), field,
          uninitializedBehavior, rng
      );
      if (failed(lowered)) {
        return failure();
      }
      return bind(structNewOp.getResult(), std::move(*lowered));
    }
    if (auto readMemberOp = dyn_cast<component::MemberReadOp>(op)) {
      auto componentValue =
          lookup(readMemberOp.getComponent(), valueMap, readMemberOp.getOperation());
      if (failed(componentValue)) {
        return failure();
      }
      auto lowered = readNamedAggregateValue(
          builder, loc, readMemberOp.getComponent().getType(), readMemberOp.getMemberName(),
          *componentValue, tables, readMemberOp.getOperation(), field
      );
      if (failed(lowered)) {
        return failure();
      }
      return bind(readMemberOp.getResult(), std::move(*lowered));
    }
    if (auto writeMemberOp = dyn_cast<component::MemberWriteOp>(op)) {
      auto componentValue =
          lookup(writeMemberOp.getComponent(), valueMap, writeMemberOp.getOperation());
      auto memberValue = lookup(writeMemberOp.getVal(), valueMap, writeMemberOp.getOperation());
      if (failed(componentValue) || failed(memberValue)) {
        return failure();
      }
      return writeNamedAggregateValue(
          builder, loc, writeMemberOp.getComponent().getType(), writeMemberOp.getMemberName(),
          valueMap[writeMemberOp.getComponent()], *memberValue, tables,
          writeMemberOp.getOperation(), field
      );
    }

    if (auto newPodOp = dyn_cast<pod::NewPodOp>(op)) {
      auto lowered = createDefaultValue(
          builder, loc, newPodOp.getType(), tables, newPodOp.getOperation(), field,
          uninitializedBehavior, rng
      );
      if (failed(lowered)) {
        return failure();
      }
      for (pod::RecordValue init : newPodOp.getInitializedRecordValues()) {
        auto value = lookup(init.value, valueMap, newPodOp.getOperation());
        if (failed(value) || failed(writeNamedAggregateValue(
                                 builder, loc, newPodOp.getType(), init.name, *lowered, *value,
                                 tables, newPodOp.getOperation(), field
                             ))) {
          return failure();
        }
      }
      return bind(newPodOp.getResult(), std::move(*lowered));
    }
    if (auto readPodOp = dyn_cast<pod::ReadPodOp>(op)) {
      auto podValue = lookup(readPodOp.getPodRef(), valueMap, readPodOp.getOperation());
      if (failed(podValue)) {
        return failure();
      }
      auto lowered = readNamedAggregateValue(
          builder, loc, readPodOp.getPodRef().getType(), readPodOp.getRecordName(), *podValue,
          tables, readPodOp.getOperation(), field
      );
      if (failed(lowered)) {
        return failure();
      }
      return bind(readPodOp.getResult(), std::move(*lowered));
    }
    if (auto writePodOp = dyn_cast<pod::WritePodOp>(op)) {
      auto recordValue = lookup(writePodOp.getValue(), valueMap, writePodOp.getOperation());
      if (failed(recordValue)) {
        return failure();
      }
      return writeNamedAggregateValue(
          builder, loc, writePodOp.getPodRef().getType(), writePodOp.getRecordName(),
          valueMap[writePodOp.getPodRef()], *recordValue, tables, writePodOp.getOperation(), field
      );
    }

    if (auto arrayNewOp = dyn_cast<array::CreateArrayOp>(op)) {
      auto lowered = createDefaultValue(
          builder, loc, arrayNewOp.getType(), tables, arrayNewOp.getOperation(), field,
          uninitializedBehavior, rng
      );
      if (failed(lowered)) {
        return failure();
      }
      if (!arrayNewOp.getElements().empty()) {
        size_t elementCount = llzk::checkedCast<size_t>(arrayNewOp.getType().getNumElements());
        if (arrayNewOp.getElements().size() != elementCount) {
          arrayNewOp.emitError("expected one explicit element per array slot in witgen lowering");
          return failure();
        }
        auto shape = arrayNewOp.getType().getShape();
        for (auto [flatIndex, operand] : llvm::enumerate(arrayNewOp.getElements())) {
          auto elementValue = lookup(operand, valueMap, arrayNewOp.getOperation());
          if (failed(elementValue)) {
            return failure();
          }
          SmallVector<Value> indices;
          auto strides = mlir::computeStrides(shape);
          for (int64_t index : mlir::delinearize(flatIndex, strides)) {
            indices.push_back(makeIndexConstant(builder, loc, index));
          }
          if (failed(writeArrayElement(
                  builder, loc, arrayNewOp.getType(), *lowered, indices, *elementValue, tables,
                  arrayNewOp.getOperation(), field
              ))) {
            return failure();
          }
        }
      }
      return bind(arrayNewOp.getResult(), std::move(*lowered));
    }
    if (auto readArrayOp = dyn_cast<array::ReadArrayOp>(op)) {
      SmallVector<Value> indices;
      for (Value indexValue : readArrayOp.getIndices()) {
        auto loweredIndex = lookupScalar(indexValue, valueMap, readArrayOp.getOperation());
        if (failed(loweredIndex)) {
          return failure();
        }
        indices.push_back(*loweredIndex);
      }
      auto arrayValue = lookup(readArrayOp.getArrRef(), valueMap, readArrayOp.getOperation());
      if (failed(arrayValue)) {
        return failure();
      }
      auto lowered = readArrayElement(
          builder, loc, mlir::cast<array::ArrayType>(readArrayOp.getArrRef().getType()),
          *arrayValue, indices, tables, readArrayOp.getOperation(), field
      );
      if (failed(lowered)) {
        return failure();
      }
      return bind(readArrayOp.getResult(), std::move(*lowered));
    }
    if (auto writeArrayOp = dyn_cast<array::WriteArrayOp>(op)) {
      SmallVector<Value> indices;
      for (Value indexValue : writeArrayOp.getIndices()) {
        auto loweredIndex = lookupScalar(indexValue, valueMap, writeArrayOp.getOperation());
        if (failed(loweredIndex)) {
          return failure();
        }
        indices.push_back(*loweredIndex);
      }
      auto elementValue = lookup(writeArrayOp.getRvalue(), valueMap, writeArrayOp.getOperation());
      if (failed(elementValue)) {
        return failure();
      }
      return writeArrayElement(
          builder, loc, mlir::cast<array::ArrayType>(writeArrayOp.getArrRef().getType()),
          valueMap[writeArrayOp.getArrRef()], indices, *elementValue, tables,
          writeArrayOp.getOperation(), field
      );
    }

    if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
      auto cond = lookupScalar(selectOp.getCondition(), valueMap, selectOp.getOperation());
      auto trueValue = lookupScalar(selectOp.getTrueValue(), valueMap, selectOp.getOperation());
      auto falseValue = lookupScalar(selectOp.getFalseValue(), valueMap, selectOp.getOperation());
      if (failed(cond) || failed(trueValue) || failed(falseValue)) {
        return failure();
      }
      return bind(
          selectOp.getResult(),
          LoweredValue {
              selectOp.getType(),
              {builder.create<arith::SelectOp>(loc, *cond, *trueValue, *falseValue)}
          }
      );
    }
    if (auto addiOp = dyn_cast<arith::AddIOp>(op)) {
      auto lhs = lookupScalar(addiOp.getLhs(), valueMap, addiOp.getOperation());
      auto rhs = lookupScalar(addiOp.getRhs(), valueMap, addiOp.getOperation());
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      return bind(
          addiOp.getResult(),
          LoweredValue {addiOp.getType(), {builder.create<arith::AddIOp>(loc, *lhs, *rhs)}}
      );
    }
    if (auto subiOp = dyn_cast<arith::SubIOp>(op)) {
      auto lhs = lookupScalar(subiOp.getLhs(), valueMap, subiOp.getOperation());
      auto rhs = lookupScalar(subiOp.getRhs(), valueMap, subiOp.getOperation());
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      return bind(
          subiOp.getResult(),
          LoweredValue {subiOp.getType(), {builder.create<arith::SubIOp>(loc, *lhs, *rhs)}}
      );
    }

    if (auto callOp = dyn_cast<function::CallOp>(op)) {
      if (callOp.getTemplateParams() || !callOp.getMapOperands().empty()) {
        callOp.emitError("execution-engine backend encountered an unflattened function.call");
        return failure();
      }
      auto *callable = callOp.resolveCallableInTable(&tables);
      auto callee = dyn_cast_or_null<function::FuncDefOp>(callable);
      if (!callee) {
        callOp.emitError("failed to resolve callee during execution-engine lowering");
        return failure();
      }
      SmallVector<Type> resultTypes;
      for (Type resultType : callOp.getResultTypes()) {
        if (failed(
                flattenABILeafTypes(resultType, tables, callOp.getOperation(), field, resultTypes)
            )) {
          return failure();
        }
      }
      SmallVector<Value> flatArgs;
      for (Value operand : callOp.getArgOperands()) {
        auto lowered = lookup(operand, valueMap, callOp.getOperation());
        auto leafTypes = getABILeafTypes(operand.getType(), tables, callOp.getOperation(), field);
        if (failed(lowered) || failed(leafTypes) ||
            failed(appendFlatLeavesToTypes(
                builder, loc, *lowered, *leafTypes, flatArgs, callOp.getOperation()
            ))) {
          return failure();
        }
      }
      auto loweredCall =
          builder.create<func::CallOp>(loc, mangleFunctionName(callee), resultTypes, flatArgs);
      unsigned cursor = 0;
      for (auto [oldResult, oldType] : llvm::zip(callOp.getResults(), callOp.getResultTypes())) {
        auto leafCount = getLeafCount(oldType, tables, callOp.getOperation(), field);
        if (failed(leafCount)) {
          return failure();
        }
        LoweredValue lowered {oldType, {}};
        lowered.leaves.append(
            loweredCall.getResults().begin() + cursor,
            loweredCall.getResults().begin() + cursor + *leafCount
        );
        cursor += *leafCount;
        valueMap[oldResult] = std::move(lowered);
      }
      return success();
    }

    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      auto lb = lookupScalar(forOp.getLowerBound(), valueMap, forOp.getOperation());
      auto ub = lookupScalar(forOp.getUpperBound(), valueMap, forOp.getOperation());
      auto step = lookupScalar(forOp.getStep(), valueMap, forOp.getOperation());
      if (failed(lb) || failed(ub) || failed(step)) {
        return failure();
      }

      SmallVector<Value> initArgs;
      SmallVector<unsigned> initLeafCounts;
      for (auto [init, resultType] : llvm::zip(forOp.getInitArgs(), forOp.getResultTypes())) {
        auto lowered = lookup(init, valueMap, forOp.getOperation());
        auto leafTypes = getABILeafTypes(resultType, tables, forOp.getOperation(), field);
        if (failed(lowered) || failed(leafTypes) ||
            failed(appendFlatLeavesToTypes(
                builder, loc, *lowered, *leafTypes, initArgs, forOp.getOperation()
            ))) {
          return failure();
        }
        auto count = getLeafCount(resultType, tables, forOp.getOperation(), field);
        if (failed(count)) {
          return failure();
        }
        initLeafCounts.push_back(*count);
      }

      auto newFor = builder.create<scf::ForOp>(loc, *lb, *ub, *step, initArgs);
      if (Attribute unsignedCmpAttr = forOp->getAttr("unsignedCmp")) {
        newFor->setAttr("unsignedCmp", unsignedCmpAttr);
      }
      DenseMap<Value, LoweredValue> bodyMap(valueMap.begin(), valueMap.end());
      bodyMap[forOp.getInductionVar()] =
          LoweredValue {forOp.getInductionVar().getType(), {newFor.getInductionVar()}};
      unsigned cursor = 0;
      for (auto [oldIterArg, oldType, leafCount] :
           llvm::zip(forOp.getRegionIterArgs(), forOp.getResultTypes(), initLeafCounts)) {
        LoweredValue lowered {oldType, {}};
        lowered.leaves.append(
            newFor.getRegionIterArgs().begin() + cursor,
            newFor.getRegionIterArgs().begin() + cursor + leafCount
        );
        bodyMap[oldIterArg] = std::move(lowered);
        cursor += leafCount;
      }
      newFor.getBody()->clear();
      OpBuilder bodyBuilder = OpBuilder::atBlockBegin(newFor.getBody());
      if (failed(lowerBlock(bodyBuilder, *forOp.getBody(), bodyMap))) {
        return failure();
      }

      cursor = 0;
      for (auto [oldResult, oldType, leafCount] :
           llvm::zip(forOp.getResults(), forOp.getResultTypes(), initLeafCounts)) {
        LoweredValue lowered {oldType, {}};
        lowered.leaves.append(
            newFor.getResults().begin() + cursor, newFor.getResults().begin() + cursor + leafCount
        );
        valueMap[oldResult] = std::move(lowered);
        cursor += leafCount;
      }
      return success();
    }

    op.emitError("unsupported operation in execution-engine lowering: ") << op.getName();
    return failure();
  }
};

/// Lower all free functions and compute functions into top-level `func.func`s.
class LowerComputeToCorePass : public PassWrapper<LowerComputeToCorePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerComputeToCorePass)

  explicit LowerComputeToCorePass(const WitgenOptions &opts) : options(opts) {}

  /// Run the pass over one module.
  StringRef getArgument() const final { return "llzk-lower-compute-to-core"; }

  /// Return the human-readable pass description.
  StringRef getDescription() const final {
    return "Lower LLZK compute IR to func/arith/cf/scf/memref";
  }

  /// Return the pass name for diagnostics.
  StringRef getName() const override { return "LowerComputeToCorePass"; }

  /// Run the pass over one module.
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    auto field = getModuleField(moduleOp);
    if (failed(field)) {
      signalPassFailure();
      return;
    }

    SymbolTableCollection tables;
    SmallVector<function::FuncDefOp> funcs;
    moduleOp.walk([&](function::FuncDefOp funcOp) {
      if (funcOp.nameIsConstrain()) {
        return;
      }
      funcs.push_back(funcOp);
    });

    BodyLowerer lowerer(moduleOp, tables, field->get(), options);
    for (function::FuncDefOp funcOp : funcs) {
      if (failed(lowerer.lowerFunction(funcOp))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  WitgenOptions options;
};

/// Create the stable JIT entry wrapper and erase the remaining LLZK declarations.
class CreateWitgenEntryPass : public PassWrapper<CreateWitgenEntryPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CreateWitgenEntryPass)

  /// Build the pass for either public-only or full-witness emission.
  explicit CreateWitgenEntryPass(bool fullWitness = false) : emitFullWitness(fullWitness) {}

  /// Run the pass over one module.
  StringRef getArgument() const final { return "llzk-create-witgen-entry"; }

  /// Return the human-readable pass description.
  StringRef getDescription() const final {
    return "Create the llzk-witgen execution-engine entry wrapper";
  }

  /// Return the pass name for diagnostics.
  StringRef getName() const override { return "CreateWitgenEntryPass"; }

  /// Run the pass over one module.
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    auto field = getModuleField(moduleOp);
    if (failed(field)) {
      signalPassFailure();
      return;
    }

    SymbolTableCollection tables;
    auto mainDef = getMainInstanceDef(tables, moduleOp.getOperation());
    if (failed(mainDef) || !mainDef.value()) {
      moduleOp.emitError("module is missing a concrete llzk.main struct");
      signalPassFailure();
      return;
    }
    function::FuncDefOp computeFunc = mainDef->get().getComputeFuncOp();
    if (!computeFunc) {
      moduleOp.emitError("main struct is missing @compute");
      signalPassFailure();
      return;
    }

    auto outputs = collectOutputBindings(
        mainDef->get(), tables, computeFunc.getOperation(),
        emitFullWitness ? OutputScope::FullWitness : OutputScope::Public
    );
    if (failed(outputs)) {
      signalPassFailure();
      return;
    }

    OpBuilder builder(moduleOp.getContext());
    builder.setInsertionPointToEnd(moduleOp.getBody());

    SmallVector<Type> wrapperArgs;
    for (Type argType : computeFunc.getArgumentTypes()) {
      SmallVector<Type> loweredLeafTypes;
      if (failed(flattenTypeLeaves(
              argType, tables, computeFunc.getOperation(), field->get(), loweredLeafTypes, {}, true
          ))) {
        signalPassFailure();
        return;
      }
      if (loweredLeafTypes.size() != 1 || !isa<MemRefType>(loweredLeafTypes.front())) {
        computeFunc.emitError(
            "execution-engine wrapper only supports felt and array<...xfelt> inputs"
        );
        signalPassFailure();
        return;
      }
      wrapperArgs.push_back(loweredLeafTypes.front());
    }
    for (const OutputBinding &output : *outputs) {
      SmallVector<Type> loweredLeafTypes;
      if (failed(flattenTypeLeaves(
              output.type, tables, computeFunc.getOperation(), field->get(), loweredLeafTypes, {},
              true
          ))) {
        signalPassFailure();
        return;
      }
      if (loweredLeafTypes.size() != 1 || !isa<MemRefType>(loweredLeafTypes.front())) {
        computeFunc.emitError(
            "execution-engine wrapper only supports felt and array<...xfelt> outputs"
        );
        signalPassFailure();
        return;
      }
      wrapperArgs.push_back(loweredLeafTypes.front());
    }

    auto wrapper = builder.create<func::FuncOp>(
        computeFunc.getLoc(), "__llzk_witgen_main",
        builder.getFunctionType(wrapperArgs, TypeRange {})
    );
    wrapper->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(), builder.getUnitAttr());
    Block *entry = wrapper.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    SmallVector<Type> loweredMainResultTypes;
    for (Type resultType : computeFunc.getResultTypes()) {
      if (failed(flattenABILeafTypes(
              resultType, tables, computeFunc.getOperation(), field->get(), loweredMainResultTypes
          ))) {
        signalPassFailure();
        return;
      }
    }

    SmallVector<Value> mainArgs;
    for (auto [argType, wrapperArg] : llvm::zip(
             computeFunc.getArgumentTypes(),
             entry->getArguments().take_front(computeFunc.getNumArguments())
         )) {
      if (isScalarType(argType)) {
        mainArgs.push_back(loadStorageScalar(builder, computeFunc.getLoc(), wrapperArg));
      } else {
        auto abiLeafTypes =
            getABILeafTypes(argType, tables, computeFunc.getOperation(), field->get());
        if (failed(abiLeafTypes) || abiLeafTypes->size() != 1 ||
            !isa<MemRefType>(abiLeafTypes->front())) {
          computeFunc.emitError("failed to derive execution-engine ABI type for main input");
          signalPassFailure();
          return;
        }
        if (wrapperArg.getType() == abiLeafTypes->front()) {
          mainArgs.push_back(wrapperArg);
        } else {
          mainArgs.push_back(builder.create<memref::CastOp>(
              computeFunc.getLoc(), abiLeafTypes->front(), wrapperArg
          ));
        }
      }
    }
    auto loweredMain = builder.create<func::CallOp>(
        computeFunc.getLoc(), mangleFunctionName(computeFunc), loweredMainResultTypes, mainArgs
    );

    LoweredValue mainResultValue {
        computeFunc.getResultTypes().front(),
        llvm::SmallVector<Value>(loweredMain.getResults().begin(), loweredMain.getResults().end())
    };

    auto extractOutputSlice = [&](ArrayRef<std::string> path, Type currentType,
                                  ArrayRef<Value> leaves,
                                  auto &self) -> FailureOr<SmallVector<Value>> {
      if (path.empty()) {
        return SmallVector<Value>(leaves.begin(), leaves.end());
      }
      if (auto structType = dyn_cast<component::StructType>(currentType)) {
        auto defLookup = structType.getDefinition(tables, computeFunc.getOperation());
        if (failed(defLookup)) {
          return failure();
        }
        unsigned localCursor = 0;
        for (component::MemberDefOp member : defLookup->get().getMemberDefs()) {
          auto leafCount =
              getLeafCount(member.getType(), tables, member.getOperation(), field->get());
          if (failed(leafCount)) {
            return failure();
          }
          ArrayRef<Value> slice = ArrayRef<Value>(leaves).slice(localCursor, *leafCount);
          localCursor += *leafCount;
          if (member.getSymName() == path.front()) {
            return self(path.drop_front(), member.getType(), slice, self);
          }
        }
        computeFunc.emitError("failed to find struct member while wiring witgen outputs");
        return failure();
      }
      if (auto podType = dyn_cast<pod::PodType>(currentType)) {
        unsigned localCursor = 0;
        for (pod::RecordAttr record : podType.getRecords()) {
          auto leafCount =
              getLeafCount(record.getType(), tables, computeFunc.getOperation(), field->get());
          if (failed(leafCount)) {
            return failure();
          }
          ArrayRef<Value> slice = ArrayRef<Value>(leaves).slice(localCursor, *leafCount);
          localCursor += *leafCount;
          if (record.getName().getValue() == path.front()) {
            return self(path.drop_front(), record.getType(), slice, self);
          }
        }
        computeFunc.emitError("failed to find POD record while wiring witgen outputs");
        return failure();
      }
      computeFunc.emitError("extra witness path components for non-aggregate output");
      return failure();
    };

    auto outputArgs = entry->getArguments().drop_front(computeFunc.getNumArguments());
    for (auto [output, outputMemRef] : llvm::zip(*outputs, outputArgs)) {
      auto slice = extractOutputSlice(
          output.path, mainResultValue.sourceType, mainResultValue.leaves, extractOutputSlice
      );
      if (failed(slice) || slice->empty()) {
        wrapper.emitError("missing selected witness output slice while building witgen entry");
        signalPassFailure();
        return;
      }
      if (isScalarType(output.type)) {
        storeStorageScalar(
            builder, computeFunc.getLoc(),
            loadStorageScalar(builder, computeFunc.getLoc(), slice->front()), outputMemRef
        );
      } else {
        builder.create<memref::CopyOp>(computeFunc.getLoc(), slice->front(), outputMemRef);
      }
    }
    builder.create<func::ReturnOp>(computeFunc.getLoc());

    SmallVector<Operation *> toErase;
    for (Operation &op : moduleOp.getBody()->getOperations()) {
      if (!isa<func::FuncOp>(op)) {
        toErase.push_back(&op);
      }
    }
    for (Operation *op : toErase) {
      op->erase();
    }
  }

private:
  bool emitFullWitness;
};

} // namespace

void addWitgenPreparePipeline(OpPassManager &pm, const WitgenOptions &) {
  llzk::polymorphic::FlatteningPassOptions flatteningOptions = {
      .cleanupMode = llzk::polymorphic::StructCleanupMode::ConcreteAsRoot
  };
  pm.addPass(llzk::polymorphic::createFlatteningPass(std::move(flatteningOptions)));
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(llzk::createInlineStructsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

std::unique_ptr<Pass> createLowerComputeToCorePass(const WitgenOptions &options) {
  return std::make_unique<LowerComputeToCorePass>(options);
}

std::unique_ptr<Pass> createCreateWitgenEntryPass(bool emitFullWitness) {
  return std::make_unique<CreateWitgenEntryPass>(emitFullWitness);
}

} // namespace llzk::witgen
