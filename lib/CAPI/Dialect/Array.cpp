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

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Wrap.h>

#include <mlir-c/Pass.h>

using namespace mlir;
using namespace llzk;
using namespace llzk::array;

static void registerLLZKArrayTransformationPasses() { registerTransformationPasses(); }

// Include the generated CAPI
#include "llzk/Dialect/Array/IR/Ops.capi.cpp.inc"
#include "llzk/Dialect/Array/IR/Types.capi.cpp.inc"
#include "llzk/Dialect/Array/Transforms/TransformationPasses.capi.cpp.inc"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Array, llzk__array, ArrayDialect)

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

MlirType
llzkArray_ArrayTypeGetWithDims(MlirType elementType, intptr_t nDims, MlirAttribute const *dims) {
  SmallVector<Attribute> dimsSto;
  return wrap(ArrayType::get(unwrap(elementType), unwrapList(nDims, dims, dimsSto)));
}

MlirType
llzkArray_ArrayTypeGetWithShape(MlirType elementType, intptr_t nDims, int64_t const *dims) {
  return wrap(ArrayType::get(unwrap(elementType), ArrayRef(dims, nDims)));
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
    Array, CreateArrayOp, WithMapOperands, MlirType arrayType, intptr_t nMapOperands,
    MlirValueRange const *mapOperands, MlirAttribute numDimsPerMap
) {
  MapOperandsHelper<> mapOps(nMapOperands, mapOperands);
  return wrap(
      create<CreateArrayOp>(
          builder, location, unwrap_cast<ArrayType>(arrayType), *mapOps,
          unwrap_cast<DenseI32ArrayAttr>(numDimsPerMap)
      )
  );
}

/// Creates a CreateArrayOp with its size information declared with AffineMaps and operands.
LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Array, CreateArrayOp, WithMapOperandsAndDims, MlirType arrayType, intptr_t nMapOperands,
    MlirValueRange const *mapOperands, intptr_t nNumsDimsPerMap, int32_t const *numDimsPerMap
) {
  MapOperandsHelper<> mapOps(nMapOperands, mapOperands);
  return wrap(
      create<CreateArrayOp>(
          builder, location, unwrap_cast<ArrayType>(arrayType), *mapOps,
          ArrayRef(numDimsPerMap, nNumsDimsPerMap)
      )
  );
}
