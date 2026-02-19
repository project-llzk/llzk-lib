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

// Include impl for transformation passes
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.capi.cpp.inc"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Polymorphic, llzk__polymorphic, PolymorphicDialect)

//===----------------------------------------------------------------------===//
// TypeVarType
//===----------------------------------------------------------------------===//

MlirType llzkTypeVarTypeGet(MlirContext ctx, MlirStringRef name) {
  return wrap(TypeVarType::get(FlatSymbolRefAttr::get(StringAttr::get(unwrap(ctx), unwrap(name)))));
}

bool llzkTypeIsATypeVarType(MlirType type) { return llvm::isa<TypeVarType>(unwrap(type)); }

MlirType llzkTypeVarTypeGetFromAttr(MlirContext /*ctx*/, MlirAttribute attrWrapper) {
  auto attr = unwrap(attrWrapper);
  if (auto sym = llvm::dyn_cast<FlatSymbolRefAttr>(attr)) {
    return wrap(TypeVarType::get(sym));
  }
  return wrap(TypeVarType::get(FlatSymbolRefAttr::get(llvm::cast<StringAttr>(attr))));
}

MlirStringRef llzkTypeVarTypeGetNameRef(MlirType type) {
  return wrap(llvm::cast<TypeVarType>(unwrap(type)).getRefName());
}

MlirAttribute llzkTypeVarTypeGetName(MlirType type) {
  return wrap(llvm::cast<TypeVarType>(unwrap(type)).getNameRef());
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

bool llzkOperationIsAApplyMapOp(MlirOperation op) { return llvm::isa<ApplyMapOp>(unwrap(op)); }

/// Returns the affine map associated with the op.
MlirAffineMap llzkApplyMapOpGetAffineMap(MlirOperation op) {
  return wrap(unwrap_cast<ApplyMapOp>(op).getAffineMap());
}

static ValueRange dimOperands(MlirOperation op) {
  return unwrap_cast<ApplyMapOp>(op).getDimOperands();
}

static ValueRange symbolOperands(MlirOperation op) {
  return unwrap_cast<ApplyMapOp>(op).getSymbolOperands();
}

static void copyValues(ValueRange in, MlirValue *out) {
  for (auto [n, value] : llvm::enumerate(in)) {
    out[n] = wrap(value);
  }
}

/// Returns the number of operands that correspond to dimensions in the affine map.
intptr_t llzkApplyMapOpGetNumDimOperands(MlirOperation op) {
  return static_cast<intptr_t>(dimOperands(op).size());
}

/// Writes into the destination buffer the operands that correspond to dimensions in the affine map.
/// The buffer needs to be preallocated first with the necessary amount and the caller is
/// responsible of its lifetime. See `llzkApplyMapOpGetNumDimOperands`.
void llzkApplyMapOpGetDimOperands(MlirOperation op, MlirValue *dst) {
  copyValues(dimOperands(op), dst);
}

/// Returns the number of operands that correspond to symbols in the affine map.
intptr_t llzkApplyMapOpGetNumSymbolOperands(MlirOperation op) {
  return static_cast<intptr_t>(symbolOperands(op).size());
}

/// Writes into the destination buffer the operands that correspond to symbols in the affine map.
/// The buffer needs to be preallocated first with the necessary amount and the caller is
/// responsible of its lifetime. See `llzkApplyMapOpGetNumSymbolOperands`.
void llzkApplyMapOpGetSymbolOperands(MlirOperation op, MlirValue *dst) {
  copyValues(symbolOperands(op), dst);
}
