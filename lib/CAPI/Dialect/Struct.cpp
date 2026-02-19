//===-- Struct.cpp - Struct dialect C API implementation --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/CAPI/Builder.h"
#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "llzk/Util/TypeHelper.h"

#include "llzk-c/Dialect/Struct.h"

#include <mlir/CAPI/AffineMap.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Support.h>
#include <mlir/CAPI/Wrap.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <mlir-c/Support.h>

#include <llvm/ADT/STLExtras.h>

using namespace mlir;
using namespace llzk;
using namespace llzk::component;

// Include the generated CAPI
#include "llzk/Dialect/Struct/IR/Ops.capi.cpp.inc"
#include "llzk/Dialect/Struct/IR/Types.capi.cpp.inc"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Struct, llzk__component, StructDialect)

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

MlirType llzkStruct_StructTypeGet(MlirAttribute name) {
  return wrap(StructType::get(llvm::cast<SymbolRefAttr>(unwrap(name))));
}

MlirType llzkStruct_StructTypeGetWithArrayAttr(MlirAttribute name, MlirAttribute params) {
  return wrap(
      StructType::get(
          llvm::cast<SymbolRefAttr>(unwrap(name)), llvm::cast<ArrayAttr>(unwrap(params))
      )
  );
}

MlirType llzkStruct_StructTypeGetWithAttrs(
    MlirAttribute name, intptr_t numParams, MlirAttribute const *params
) {
  SmallVector<Attribute> paramsSto;
  return wrap(
      StructType::get(
          llvm::cast<SymbolRefAttr>(unwrap(name)), unwrapList(numParams, params, paramsSto)
      )
  );
}

//===----------------------------------------------------------------------===//
// StructDefOp
//===----------------------------------------------------------------------===//

MlirBlock llzkStruct_StructDefOpGetBody(MlirOperation op) {
  return wrap(llvm::cast<StructDefOp>(unwrap(op)).getBody());
}

MlirType llzkStruct_StructDefOpGetType(MlirOperation op) {
  return wrap(llvm::cast<StructDefOp>(unwrap(op)).getType());
}

MlirType llzkStruct_StructDefOpGetTypeWithParams(MlirOperation op, MlirAttribute attr) {
  return wrap(llvm::cast<StructDefOp>(unwrap(op)).getType(llvm::cast<ArrayAttr>(unwrap(attr))));
}

void llzkStruct_StructDefOpGetFieldDefs(MlirOperation op, MlirOperation *dst) {
  for (auto [offset, field] : llvm::enumerate(llvm::cast<StructDefOp>(unwrap(op)).getFieldDefs())) {
    dst[offset] = wrap(field);
  }
}

intptr_t llzkStruct_StructDefOpGetNumFieldDefs(MlirOperation op) {
  return static_cast<intptr_t>(llvm::cast<StructDefOp>(unwrap(op)).getFieldDefs().size());
}

const char *llzkStruct_StructDefOpGetHeaderString(
    MlirOperation op, intptr_t *strSize, char *(*alloc_string)(size_t)
) {
  auto header = llvm::cast<StructDefOp>(unwrap(op)).getHeaderString();
  *strSize = static_cast<intptr_t>(header.size()) + 1; // Plus one because it's a C string.
  char *dst = alloc_string(*strSize);
  dst[header.size()] = 0;
  memcpy(dst, header.data(), header.size());
  return dst;
}

bool llzkStruct_StructDefOpGetHasParamName(MlirOperation op, MlirStringRef name) {
  Builder builder(unwrap(op)->getContext());
  return llvm::cast<StructDefOp>(unwrap(op)).hasParamNamed(builder.getStringAttr(unwrap(name)));
}

//===----------------------------------------------------------------------===//
// FieldReadOp
//===----------------------------------------------------------------------===//

LLZK_DEFINE_OP_BUILD_METHOD(
    Struct, FieldReadOp, MlirType fieldType, MlirValue component, MlirStringRef name
) {
  return wrap(
      create<FieldReadOp>(
          builder, location, unwrap(fieldType), unwrap(component),
          unwrap(builder)->getStringAttr(unwrap(name))
      )
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Struct, FieldReadOp, WithAffineMapDistance, MlirType fieldType, MlirValue component,
    MlirStringRef name, MlirAffineMap map, MlirValueRange mapOperands, int32_t numDimsPerMap
) {
  SmallVector<Value> mapOperandsSto;
  auto nameAttr = unwrap(builder)->getStringAttr(unwrap(name));
  auto mapAttr = AffineMapAttr::get(unwrap(map));
  return wrap(
      create<FieldReadOp>(
          builder, location, unwrap(fieldType), unwrap(component), nameAttr, mapAttr,
          unwrapList(mapOperands.size, mapOperands.values, mapOperandsSto), numDimsPerMap
      )
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Struct, FieldReadOp, WithConstParamDistance, MlirType fieldType, MlirValue component,
    MlirStringRef name, MlirStringRef symbol
) {
  auto nameAttr = unwrap(builder)->getStringAttr(unwrap(name));
  return wrap(
      create<FieldReadOp>(
          builder, location, unwrap(fieldType), unwrap(component), nameAttr,
          FlatSymbolRefAttr::get(unwrap(builder)->getStringAttr(unwrap(symbol)))
      )
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Struct, FieldReadOp, WithLiteralDistance, MlirType fieldType, MlirValue component,
    MlirStringRef name, int64_t distance
) {
  auto nameAttr = unwrap(builder)->getStringAttr(unwrap(name));
  return wrap(
      create<FieldReadOp>(
          builder, location, unwrap(fieldType), unwrap(component), nameAttr,
          unwrap(builder)->getIndexAttr(distance)
      )
  );
}
