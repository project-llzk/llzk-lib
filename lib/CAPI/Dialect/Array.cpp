//===-- Array.cpp - Array dialect C API implementation ----------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/CAPI/Builder.h"
#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Array/Transforms/TransformationPasses.h"

#include "llzk-c/Dialect/Array.h"
#include "llzk-c/Support.h"

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Wrap.h>

#include <mlir-c/IR.h>
#include <mlir-c/Pass.h>

using namespace mlir;
using namespace llzk;
using namespace llzk::array;

static void registerLLZKArrayTransformationPasses() { registerTransformationPasses(); }

// Include impl for transformation passes
#include "llzk/Dialect/Array/Transforms/TransformationPasses.capi.cpp.inc"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Array, llzk__array, ArrayDialect)

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

MlirType llzkArrayTypeGet(MlirType elementType, intptr_t nDims, MlirAttribute const *dims) {
  SmallVector<Attribute> dimsSto;
  return wrap(ArrayType::get(unwrap(elementType), unwrapList(nDims, dims, dimsSto)));
}

MlirType
llzkArrayTypeGetWithNumericDims(MlirType elementType, intptr_t nDims, int64_t const *dims) {
  return wrap(ArrayType::get(unwrap(elementType), ArrayRef(dims, nDims)));
}

bool llzkTypeIsA_Array_ArrayType(MlirType type) { return llvm::isa<ArrayType>(unwrap(type)); }

MlirType llzkArrayTypeGetElementType(MlirType type) {
  return wrap(unwrap_cast<ArrayType>(type).getElementType());
}

intptr_t llzkArrayTypeGetNumDims(MlirType type) {
  return static_cast<intptr_t>(unwrap_cast<ArrayType>(type).getDimensionSizes().size());
}

MlirAttribute llzkArrayTypeGetDim(MlirType type, intptr_t idx) {
  return wrap(unwrap_cast<ArrayType>(type).getDimensionSizes()[idx]);
}

//===----------------------------------------------------------------------===//
// CreateArrayOp
//===----------------------------------------------------------------------===//

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Array, CreateArrayOp, WithValues, MlirType arrayType, intptr_t nValues, MlirValue const *values
) {
  SmallVector<Value> valueSto;
  return wrap(
      create<CreateArrayOp>(
          builder, location, unwrap_cast<ArrayType>(arrayType),
          ValueRange(unwrapList(nValues, values, valueSto))
      )
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Array, CreateArrayOp, WithMapOperands, MlirType arrayType,
    LlzkAffineMapOperandsBuilder mapOperands
) {
  MapOperandsHelper<> mapOps(mapOperands.nMapOperands, mapOperands.mapOperands);
  auto numDimsPerMap =
      llzkAffineMapOperandsBuilderGetDimsPerMapAttr(mapOperands, mlirLocationGetContext(location));
  return wrap(
      create<CreateArrayOp>(
          builder, location, unwrap_cast<ArrayType>(arrayType), *mapOps,
          unwrap_cast<DenseI32ArrayAttr>(numDimsPerMap)
      )
  );
}
