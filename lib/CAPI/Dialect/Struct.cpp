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
#include "llzk/Util/SymbolLookup.h"
#include "llzk/Util/TypeHelper.h"

#include "llzk-c/Dialect/Struct.h"

#include <mlir/CAPI/AffineMap.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Support.h>
#include <mlir/CAPI/Wrap.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/SymbolTable.h>

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
MlirLogicalResult llzkStructStructTypeGetDefinition(
    MlirType type, MlirOperation root, LlzkSymbolLookupResult *result
) {
  auto structType = mlir::unwrap_cast<StructType>(type);
  auto *rootOp = unwrap(root);
  SymbolTableCollection stc;
  mlir::FailureOr<llzk::SymbolLookupResult<StructDefOp>> lookup =
      structType.getDefinition(stc, rootOp);

  if (succeeded(lookup)) {
    // Allocate the result in the heap and store the pointer in the out var.
    result->ptr = new llzk::SymbolLookupResultUntyped(std::move(*lookup));
  }
  return wrap(lookup);
}

MlirLogicalResult llzkStructStructTypeGetDefinitionFromModule(
    MlirType type, MlirModule root, LlzkSymbolLookupResult *result
) {
  return llzkStructStructTypeGetDefinition(type, mlirModuleGetOperation(root), result);
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

void llzkStruct_StructDefOpGetMemberDefs(MlirOperation op, MlirOperation *dst) {
  for (auto [offset, member] :
       llvm::enumerate(llvm::cast<StructDefOp>(unwrap(op)).getMemberDefs())) {
    dst[offset] = wrap(member);
  }
}

intptr_t llzkStruct_StructDefOpGetNumMemberDefs(MlirOperation op) {
  return static_cast<intptr_t>(llvm::cast<StructDefOp>(unwrap(op)).getMemberDefs().size());
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

bool llzkStruct_StructDefOpHasParamName(MlirOperation op, MlirStringRef name) {
  return llvm::cast<StructDefOp>(unwrap(op)).hasParamNamed(unwrap(name));
}

//===----------------------------------------------------------------------===//
// MemberReadOp
//===----------------------------------------------------------------------===//

LLZK_DEFINE_OP_BUILD_METHOD(
    Struct, MemberReadOp, MlirType memberType, MlirValue component, MlirStringRef name
) {
  return wrap(
      create<MemberReadOp>(
          builder, location, unwrap(memberType), unwrap(component),
          unwrap(builder)->getStringAttr(unwrap(name))
      )
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Struct, MemberReadOp, WithAffineMapDistance, MlirType memberType, MlirValue component,
    MlirStringRef name, MlirAffineMap map, MlirValueRange mapOperands
) {
  SmallVector<Value> mapOperandsSto;
  auto nameAttr = unwrap(builder)->getStringAttr(unwrap(name));
  auto mapAttr = AffineMapAttr::get(unwrap(map));
  return wrap(
      create<MemberReadOp>(
          builder, location, unwrap(memberType), unwrap(component), nameAttr, mapAttr,
          unwrapList(mapOperands.size, mapOperands.values, mapOperandsSto),
          mapAttr.getAffineMap().getNumDims()
      )
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Struct, MemberReadOp, WithConstParamDistance, MlirType memberType, MlirValue component,
    MlirStringRef name, MlirStringRef symbol
) {
  auto nameAttr = unwrap(builder)->getStringAttr(unwrap(name));
  return wrap(
      create<MemberReadOp>(
          builder, location, unwrap(memberType), unwrap(component), nameAttr,
          FlatSymbolRefAttr::get(unwrap(builder)->getStringAttr(unwrap(symbol)))
      )
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Struct, MemberReadOp, WithLiteralDistance, MlirType memberType, MlirValue component,
    MlirStringRef name, int64_t distance
) {
  auto nameAttr = unwrap(builder)->getStringAttr(unwrap(name));
  return wrap(
      create<MemberReadOp>(
          builder, location, unwrap(memberType), unwrap(component), nameAttr,
          unwrap(builder)->getIndexAttr(distance)
      )
  );
}
