//===-- ArrayTypeHelper.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation for array type helpers.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/Util/ArrayTypeHelper.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/Matchers.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>

using namespace mlir;
using namespace llzk;
using namespace llzk::array;

ArrayIndexGen::ArrayIndexGen(ArrayType t)
    : shape(t.getShape()), linearSize(t.getNumElements()), strides(mlir::computeStrides(shape)) {}

ArrayIndexGen ArrayIndexGen::from(ArrayType t) {
  assert(t.hasStaticShape());
  return ArrayIndexGen(t);
}

namespace {

inline bool isInRange(int64_t idx, int64_t dimSize) { return 0 <= idx && idx < dimSize; }

// This can support Value, Attribute, and Operation* per matchPattern() implementations.
template <typename TypeOfIndex> inline std::optional<int64_t> toI64(TypeOfIndex index) {
  llvm::APInt idxAP;
  if (!mlir::matchPattern(index, mlir::m_ConstantInt(&idxAP))) {
    return std::nullopt;
  }
  return llzk::fromAPInt(idxAP);
}

template <typename OutType> struct CheckAndConvert {
  template <typename InType>
  static std::optional<OutType> from(InType /*index*/, int64_t /*dimSize*/) {
    static_assert(sizeof(OutType) == 0, "CheckAndConvert not implemented for requested type.");
    assert(false);
  }
};

// Specialization to produce `int64_t`
template <> struct CheckAndConvert<int64_t> {
  template <typename InType> static std::optional<int64_t> from(InType index, int64_t dimSize) {
    if (auto idxVal = toI64<InType>(index)) {
      if (isInRange(*idxVal, dimSize)) {
        return idxVal;
      }
    }
    return std::nullopt;
  }
};

// Specialization to produce `Attribute`
template <> struct CheckAndConvert<Attribute> {
  template <typename InType> static std::optional<Attribute> from(InType index, int64_t dimSize) {
    if (auto c = CheckAndConvert<int64_t>::from(index, dimSize)) {
      return IntegerAttr::get(IndexType::get(index.getContext()), *c);
    }
    return std::nullopt;
  }
};

template <typename OutType, typename InListType>
inline std::optional<SmallVector<OutType>>
checkAndConvertMulti(InListType multiDimIndex, ArrayRef<int64_t> shape, bool mustBeEqual) {
  if (mustBeEqual) {
    assert(
        llvm::all_equal({llvm::range_size(multiDimIndex), llvm::range_size(shape)}) &&
        "Iteratees do not have equal length"
    );
  }
  SmallVector<OutType> ret;
  for (auto [idx, dimSize] : llvm::zip_first(multiDimIndex, shape)) {
    std::optional<OutType> next = CheckAndConvert<OutType>::from(idx, dimSize);
    if (!next.has_value()) {
      return std::nullopt;
    }
    ret.push_back(next.value());
  }
  return ret;
}

inline std::optional<int64_t> linearizeImpl(
    ArrayRef<int64_t> multiDimIndex, const ArrayRef<int64_t> &shape,
    const SmallVector<int64_t> &strides
) {
  // Ensure the index for each dimension is in range. Then the linearized index will be as well.
  for (auto [idx, dimSize] : llvm::zip_equal(multiDimIndex, shape)) {
    if (!isInRange(idx, dimSize)) {
      return std::nullopt;
    }
  }
  return mlir::linearize(multiDimIndex, strides);
}

template <typename TypeOfIndex>
inline std::optional<int64_t> linearizeImpl(
    ArrayRef<TypeOfIndex> multiDimIndex, const ArrayRef<int64_t> &shape,
    const SmallVector<int64_t> &strides
) {
  std::optional<SmallVector<int64_t>> conv =
      checkAndConvertMulti<int64_t>(multiDimIndex, shape, true /*TODO: I think*/);
  if (!conv.has_value()) {
    return std::nullopt;
  }
  return mlir::linearize(conv.value(), strides);
}

template <typename ResultElemType>
inline std::optional<SmallVector<ResultElemType>> delinearizeImpl(
    int64_t linearIndex, int64_t linearSize, const SmallVector<int64_t> &strides, MLIRContext *ctx,
    llvm::function_ref<ResultElemType(IntegerAttr)> convert
) {
  if (!isInRange(linearIndex, linearSize)) {
    return std::nullopt;
  }
  SmallVector<ResultElemType> ret;
  for (int64_t idx : mlir::delinearize(linearIndex, strides)) {
    ret.push_back(convert(IntegerAttr::get(IndexType::get(ctx), idx)));
  }
  return ret;
}

} // namespace

std::optional<SmallVector<Value>>
ArrayIndexGen::delinearize(int64_t linearIndex, Location loc, OpBuilder &bldr) const {
  return delinearizeImpl<Value>(
      linearIndex, linearSize, strides, bldr.getContext(),
      [&](IntegerAttr a) { return bldr.create<arith::ConstantOp>(loc, a); }
  );
}

std::optional<SmallVector<Attribute>>
ArrayIndexGen::delinearize(int64_t linearIndex, MLIRContext *ctx) const {
  return delinearizeImpl<Attribute>(linearIndex, linearSize, strides, ctx, [](IntegerAttr a) {
    return a;
  });
}

template <typename InListType> std::optional<int64_t> ArrayIndexGen::linearize(InListType) const {
  static_assert(sizeof(InListType) == 0, "linearize() not implemented for requested type.");
  llvm_unreachable("must have concrete instantiation");
  return std::nullopt;
}

template <> std::optional<int64_t> ArrayIndexGen::linearize(ArrayRef<int64_t> multiDimIndex) const {
  return linearizeImpl(multiDimIndex, shape, strides);
}

template <>
std::optional<int64_t> ArrayIndexGen::linearize(ArrayRef<Attribute> multiDimIndex) const {
  return linearizeImpl(multiDimIndex, shape, strides);
}

template <>
std::optional<int64_t> ArrayIndexGen::linearize(ArrayRef<Operation *> multiDimIndex) const {
  return linearizeImpl(multiDimIndex, shape, strides);
}

template <> std::optional<int64_t> ArrayIndexGen::linearize(ArrayRef<Value> multiDimIndex) const {
  return linearizeImpl(multiDimIndex, shape, strides);
}

template <typename InListType>
std::optional<SmallVector<Attribute>> ArrayIndexGen::checkAndConvert(InListType) {
  static_assert(sizeof(InListType) == 0, "checkAndConvert() not implemented for requested type.");
  llvm_unreachable("must have concrete instantiation");
  return std::nullopt;
}

template <>
std::optional<SmallVector<Attribute>> ArrayIndexGen::checkAndConvert(OperandRange multiDimIndex) {
  return checkAndConvertMulti<Attribute>(multiDimIndex, shape, false);
}
