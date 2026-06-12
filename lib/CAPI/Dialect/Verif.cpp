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

void llzkVerif_attachInterfaces(MlirContext context) {
  ::llzk::verif::attachInterfaces(*unwrap(context));
}

//===----------------------------------------------------------------------===//
// ContractOp
//===----------------------------------------------------------------------===//

LLZK_DEFINE_OP_BUILD_METHOD(
    Verif, ContractOp, MlirIdentifier sym_name, MlirAttribute target, MlirAttribute function_type,
    MlirAttribute arg_attrs
) {
  ArrayAttr argAttrs;
  if (!mlirAttributeIsNull(arg_attrs)) {
    argAttrs = cast<ArrayAttr>(unwrap(arg_attrs));
  }

  return mlirOpBuilderInsert(
      builder,
      wrap(
          llzk::create<ContractOp>(
              builder, location, unwrap(sym_name), cast<SymbolRefAttr>(unwrap(target)),
              cast<FunctionType>(cast<TypeAttr>(unwrap(function_type)).getValue()), argAttrs
          )
      )
  );
}

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

//===----------------------------------------------------------------------===//
// InvariantOp
//===----------------------------------------------------------------------===//

LLZK_DEFINE_OP_BUILD_METHOD(
    Verif, InvariantOp, MlirStringRef loopName, intptr_t numArgs, MlirType const *types, MlirLocation const *locs
) {
  SmallVector<Type> typesSto;
  SmallVector<Location> locsSto;
  auto typesRef = unwrapList(numArgs, types, typesSto);
  auto locsRef = unwrapList(numArgs, locs, locsSto);

  return mlirOpBuilderInsert(builder, wrap(llzk::create<InvariantOp>(builder, location, unwrap(loopName), typesRef, locsRef)));
}


MlirBlock llzkVerif_InvariantOpGetBody(MlirOperation op) {
  return wrap(unwrap_cast<InvariantOp>(op).getBody());
}
