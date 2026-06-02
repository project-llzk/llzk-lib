//===-- Verif.cpp - Verif dialect C API implementation --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Verif.h"

#include "llzk/CAPI/Builder.h"
#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/Verif/IR/Dialect.h"
#include "llzk/Dialect/Verif/IR/Ops.h"

#include <mlir-c/BuiltinAttributes.h>

#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Wrap.h>
#include <mlir/IR/BuiltinAttributes.h>

using namespace mlir;
using namespace llzk::verif;

// Include the generated CAPI
#include "llzk/Dialect/Verif/IR/Ops.capi.cpp.inc"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Verif, llzk__verif, VerifDialect)

//===----------------------------------------------------------------------===//
// ContractOp
//===----------------------------------------------------------------------===//

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Verif, ContractOp, FromTargetIdentifier, MlirIdentifier sym_name, MlirIdentifier target
) {
  return mlirOpBuilderInsert(
      builder, wrap(llzk::create<ContractOp>(builder, location, unwrap(sym_name), unwrap(target)))
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Verif, ContractOp, FromTargetAttr, MlirIdentifier sym_name, MlirAttribute target
) {
  return mlirOpBuilderInsert(
      builder, wrap(
                   llzk::create<ContractOp>(
                       builder, location, unwrap(sym_name), cast<SymbolRefAttr>(unwrap(target))
                   )
               )
  );
}

//===----------------------------------------------------------------------===//
// IncludeOp
//===----------------------------------------------------------------------===//

LLZK_DEFINE_OP_BUILD_METHOD(
    Verif, IncludeOp, MlirAttribute callee, MlirValueRange argOperands, MlirAttribute templateParams
) {
  SmallVector<Value> argOperandsSto;
  ArrayAttr paramsAttr;
  if (!mlirAttributeIsNull(templateParams)) {
    paramsAttr = cast<ArrayAttr>(unwrap(templateParams));
  }

  return mlirOpBuilderInsert(
      builder, wrap(
                   llzk::create<IncludeOp>(
                       builder, location, cast<SymbolRefAttr>(unwrap(callee)),
                       unwrapList(argOperands.size, argOperands.values, argOperandsSto),
                       paramsAttr ? paramsAttr.getValue() : ArrayRef<Attribute> {}
                   )
               )
  );
}
