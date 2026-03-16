//===-- Poly.cpp - Polymorphic dialect C API impl ---------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/CAPI/Builder.h"
#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/Polymorphic/IR/Dialect.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Types.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Util/Compare.h"

#include "llzk-c/Dialect/Poly.h"

#include <mlir/CAPI/AffineExpr.h>
#include <mlir/CAPI/AffineMap.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Wrap.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LLVM.h>

#include <mlir-c/Pass.h>

using namespace mlir;
using namespace llzk;
using namespace llzk::polymorphic;

static void registerLLZKPolymorphicTransformationPasses() { registerTransformationPasses(); }

// Include the generated CAPI
#include "llzk/Dialect/Polymorphic/IR/Ops.capi.cpp.inc"
#include "llzk/Dialect/Polymorphic/IR/Types.capi.cpp.inc"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.capi.cpp.inc"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Polymorphic, llzk__polymorphic, PolymorphicDialect)

//===----------------------------------------------------------------------===//
// TypeVarType
//===----------------------------------------------------------------------===//

MlirType llzkPoly_TypeVarTypeGetFromStringRef(MlirContext ctx, MlirStringRef name) {
  return wrap(TypeVarType::get(FlatSymbolRefAttr::get(StringAttr::get(unwrap(ctx), unwrap(name)))));
}

MlirType llzkPoly_TypeVarTypeGetFromAttr(MlirAttribute attrWrapper) {
  auto attr = unwrap(attrWrapper);
  if (auto sym = llvm::dyn_cast<FlatSymbolRefAttr>(attr)) {
    return wrap(TypeVarType::get(sym));
  }
  return wrap(TypeVarType::get(FlatSymbolRefAttr::get(llvm::cast<StringAttr>(attr))));
}

//===----------------------------------------------------------------------===//
// TemplateOp
//===----------------------------------------------------------------------===//

static inline TemplateOp asTemplateOp(MlirOperation op) { return unwrap_cast<TemplateOp>(op); }

static inline void copyAttrs(SmallVector<Attribute> attrs, MlirAttribute *dst) {
  for (auto [n, attr] : llvm::enumerate(attrs)) {
    dst[n] = wrap(attr);
  }
}

bool llzkPoly_TemplateOpHasConstParamOps(MlirOperation op) {
  return asTemplateOp(op).hasConstOps<TemplateParamOp>();
}

intptr_t llzkPoly_TemplateOpNumConstParamOps(MlirOperation op) {
  return llzk::checkedCast<intptr_t>(asTemplateOp(op).numConstOps<TemplateParamOp>());
}

void llzkPoly_TemplateOpGetConstParamNames(MlirOperation op, MlirAttribute *dst) {
  copyAttrs(asTemplateOp(op).getConstNames<TemplateParamOp>(), dst);
}

bool llzkPoly_TemplateOpHasConstParamNamed(MlirOperation op, MlirStringRef find) {
  return asTemplateOp(op).hasConstNamed<TemplateParamOp>(unwrap(find));
}

bool llzkPoly_TemplateOpHasConstExprOps(MlirOperation op) {
  return asTemplateOp(op).hasConstOps<TemplateExprOp>();
}

intptr_t llzkPoly_TemplateOpNumConstExprOps(MlirOperation op) {
  return llzk::checkedCast<intptr_t>(asTemplateOp(op).numConstOps<TemplateExprOp>());
}

void llzkPoly_TemplateOpGetConstExprNames(MlirOperation op, MlirAttribute *dst) {
  copyAttrs(asTemplateOp(op).getConstNames<TemplateExprOp>(), dst);
}

bool llzkPoly_TemplateOpHasConstExprNamed(MlirOperation op, MlirStringRef find) {
  return asTemplateOp(op).hasConstNamed<TemplateExprOp>(unwrap(find));
}

//===----------------------------------------------------------------------===//
// ApplyMapOp
//===----------------------------------------------------------------------===//

LLZK_DEFINE_OP_BUILD_METHOD(Poly, ApplyMapOp, MlirAttribute map, MlirValueRange mapOperands) {
  SmallVector<Value> mapOperandsSto;
  return wrap(
      create<ApplyMapOp>(
          builder, location, llvm::cast<AffineMapAttr>(unwrap(map)),
          ValueRange(unwrapList(mapOperands.size, mapOperands.values, mapOperandsSto))
      )
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Poly, ApplyMapOp, WithAffineMap, MlirAffineMap map, MlirValueRange mapOperands
) {
  SmallVector<Value> mapOperandsSto;
  return wrap(
      create<ApplyMapOp>(
          builder, location, unwrap(map),
          ValueRange(unwrapList(mapOperands.size, mapOperands.values, mapOperandsSto))
      )
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Poly, ApplyMapOp, WithAffineExpr, MlirAffineExpr expr, MlirValueRange mapOperands
) {
  SmallVector<Value> mapOperandsSto;
  return wrap(
      create<ApplyMapOp>(
          builder, location, unwrap(expr),
          ValueRange(unwrapList(mapOperands.size, mapOperands.values, mapOperandsSto))
      )
  );
}

static inline ValueRange dimOperands(MlirOperation op) {
  return unwrap_cast<ApplyMapOp>(op).getDimOperands();
}

static inline ValueRange symbolOperands(MlirOperation op) {
  return unwrap_cast<ApplyMapOp>(op).getSymbolOperands();
}

static inline void copyValues(ValueRange in, MlirValue *out) {
  for (auto [n, value] : llvm::enumerate(in)) {
    out[n] = wrap(value);
  }
}

/// Returns the number of operands that correspond to dimensions in the affine map.
intptr_t llzkPoly_ApplyMapOpGetNumDimOperands(MlirOperation op) {
  return llzk::checkedCast<intptr_t>(dimOperands(op).size());
}

/// Writes into the destination buffer the operands that correspond to dimensions in the affine map.
/// The buffer needs to be preallocated first with the necessary amount and the caller is
/// responsible of its lifetime. See `llzkPoly_ApplyMapOpGetNumDimOperands`.
void llzkPoly_ApplyMapOpGetDimOperands(MlirOperation op, MlirValue *dst) {
  copyValues(dimOperands(op), dst);
}

/// Returns the number of operands that correspond to symbols in the affine map.
intptr_t llzkPoly_ApplyMapOpGetNumSymbolOperands(MlirOperation op) {
  return llzk::checkedCast<intptr_t>(symbolOperands(op).size());
}

/// Writes into the destination buffer the operands that correspond to symbols in the affine map.
/// The buffer needs to be preallocated first with the necessary amount and the caller is
/// responsible of its lifetime. See `llzkPoly_ApplyMapOpGetNumSymbolOperands`.
void llzkPoly_ApplyMapOpGetSymbolOperands(MlirOperation op, MlirValue *dst) {
  copyValues(symbolOperands(op), dst);
}
