//===-- Support.cpp - C API general utilities ---------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/CAPI/Support.h"
#include "llzk/Util/SymbolLookup.h"

#include "llzk-c/Support.h"

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/CAPI/Wrap.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Iterators.h>

#include <mlir-c/BuiltinAttributes.h>

#include <cstdint>
#include <cstring>

using namespace llzk;
using namespace mlir;

/// Destroys the lookup result, releasing its resources.
void llzkSymbolLookupResultDestroy(LlzkSymbolLookupResult result) {
  delete reinterpret_cast<SymbolLookupResultUntyped *>(result.ptr);
}

/// Returns the looked up Operation.
///
/// The lifetime of the Operation is tied to the lifetime of the lookup result.
MlirOperation LlzkSymbolLookupResultGetOperation(LlzkSymbolLookupResult wrapped) {
  SymbolLookupResultUntyped *result = reinterpret_cast<SymbolLookupResultUntyped *>(wrapped.ptr);
  return wrap(result->get());
}

/// Note: Duplicated from upstream LLVM. Available in 21.1.8 and later.
void mlirOperationReplaceUsesOfWith(MlirOperation op, MlirValue oldValue, MlirValue newValue) {
  unwrap(op)->replaceUsesOfWith(unwrap(oldValue), unwrap(newValue));
}

/// Note: Duplicated from upstream LLVM.
static mlir::WalkResult unwrap(MlirWalkResult result) {
  switch (result) {
  case MlirWalkResultAdvance:
    return mlir::WalkResult::advance();

  case MlirWalkResultInterrupt:
    return mlir::WalkResult::interrupt();

  case MlirWalkResultSkip:
    return mlir::WalkResult::skip();
  }
  llvm_unreachable("unknown result in WalkResult::unwrap");
}

void mlirOperationWalkReverse(
    MlirOperation from, MlirOperationWalkCallback callback, void *userData, MlirWalkOrder walkOrder
) {
  switch (walkOrder) {
  case MlirWalkPreOrder:
    unwrap(from)->walk<WalkOrder::PreOrder, ReverseIterator>([callback, userData](Operation *op) {
      return unwrap(callback(wrap(op), userData));
    });
    break;
  case MlirWalkPostOrder:
    unwrap(from)->walk<WalkOrder::PostOrder, ReverseIterator>([callback, userData](Operation *op) {
      return unwrap(callback(wrap(op), userData));
    });
  }
}

//===----------------------------------------------------------------------===//
// LlzkAffineMapOperandsBuilder implementation.
//===----------------------------------------------------------------------===//

namespace {
template <typename T> void appendElems(T const *src, intptr_t srcSize, T *&dst, intptr_t &dstSize) {
  assert(srcSize >= 0 && "Negative source size");
  assert(dstSize >= 0 && "Negative destination size");
  dst = static_cast<T *>(std::realloc(dst, (srcSize + dstSize) * sizeof(T)));
  assert(dst && "Failed to increase the size of buffer");
  std::memcpy(dst + dstSize, src, srcSize * sizeof(T));
  dstSize += srcSize;
}

static void maybeDeallocArray(LlzkAffineMapOperandsBuilder *builder) {
  if (builder->nDimsPerMap >= 0 && builder->dimsPerMap.array) {
    std::free(builder->dimsPerMap.array);
    builder->dimsPerMap.array = nullptr;
  }
}

/// Asserts that the length of both arrays is the same.
///
/// If the builder is in Attr mode uses the length of the attribute's
/// internal buffer instead of the length property.
static void assertArraysAreInSync(LlzkAffineMapOperandsBuilder *builder) {
  intptr_t nDimsPerMap = builder->nDimsPerMap < 0
                             ? unwrap_cast<DenseI32ArrayAttr>(builder->dimsPerMap.attr).size()
                             : builder->nDimsPerMap;
  (void)nDimsPerMap; // To silence unused variable warning if the assert below is compiled out.
  assert(builder->nMapOperands == nDimsPerMap);
}

} // namespace

LlzkAffineMapOperandsBuilder llzkAffineMapOperandsBuilderCreate() {
  LlzkAffineMapOperandsBuilder builder;
  builder.nMapOperands = 0;
  builder.mapOperands = nullptr;
  builder.nDimsPerMap = 0;
  builder.dimsPerMap.array = nullptr;

  return builder;
}

void llzkAffineMapOperandsBuilderDestroy(LlzkAffineMapOperandsBuilder *builder) {
  if (!builder) {
    return;
  }
  if (builder->mapOperands) {
    std::free(builder->mapOperands);
  }
  maybeDeallocArray(builder);
  // Reset values to 0/NULL.
  *builder = llzkAffineMapOperandsBuilderCreate();
}

void llzkAffineMapOperandsBuilderAppendOperands(
    LlzkAffineMapOperandsBuilder *builder, intptr_t n, MlirValueRange const *mapOperands
) {
  appendElems(mapOperands, n, builder->mapOperands, builder->nMapOperands);
}

void llzkAffineMapOperandsBuilderAppendOperandsWithDimCount(
    LlzkAffineMapOperandsBuilder *builder, intptr_t n, MlirValueRange const *mapOperands,
    int32_t const *dimsPerMap
) {
  assertArraysAreInSync(builder);
  llzkAffineMapOperandsBuilderAppendOperands(builder, n, mapOperands);
  llzkAffineMapOperandsBuilderAppendDimCount(builder, n, dimsPerMap);
  assertArraysAreInSync(builder);
}

void llzkAffineMapOperandsBuilderAppendDimCount(
    LlzkAffineMapOperandsBuilder *builder, intptr_t n, int32_t const *dimsPerMap
) {
  llzkAffineMapOperandsBuilderConvertDimsPerMapToArray(builder);
  appendElems(dimsPerMap, n, builder->dimsPerMap.array, builder->nDimsPerMap);
}

void llzkAffineMapOperandsBuilderSetDimsPerMapFromAttr(
    LlzkAffineMapOperandsBuilder *builder, MlirAttribute attribute
) {
  assert(mlirAttributeIsADenseI32Array(attribute) && "Attribute is not a DenseI32Array");
  maybeDeallocArray(builder);
  builder->nDimsPerMap = -1;
  builder->dimsPerMap.attr = attribute;
}

void llzkAffineMapOperandsBuilderConvertDimsPerMapToArray(LlzkAffineMapOperandsBuilder *builder) {
  if (builder->nDimsPerMap >= 0) {
    return;
  }
  auto attrData = mlir::unwrap_cast<DenseI32ArrayAttr>(builder->dimsPerMap.attr).asArrayRef();
  size_t realSize = attrData.size() * sizeof(decltype(attrData)::value_type);
  builder->dimsPerMap.array = static_cast<int32_t *>(std::malloc(realSize));
  std::memcpy(builder->dimsPerMap.array, attrData.data(), realSize);
  builder->nDimsPerMap = static_cast<intptr_t>(attrData.size());
}

void llzkAffineMapOperandsBuilderConvertDimsPerMapToAttr(
    LlzkAffineMapOperandsBuilder *builder, MlirContext context
) {

  auto attr = llzkAffineMapOperandsBuilderGetDimsPerMapAttr(*builder, context);
  llzkAffineMapOperandsBuilderSetDimsPerMapFromAttr(builder, attr);
}

MlirAttribute llzkAffineMapOperandsBuilderGetDimsPerMapAttr(
    LlzkAffineMapOperandsBuilder builder, MlirContext context
) {
  if (builder.nDimsPerMap < 0) {
    return builder.dimsPerMap.attr;
  }
  return mlirDenseI32ArrayGet(context, builder.nDimsPerMap, builder.dimsPerMap.array);
}
