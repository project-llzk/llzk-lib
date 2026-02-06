//===-- Types.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_FELT_TYPES
#define LLZK_FELT_TYPES

include "llzk/Dialect/Felt/IR/Dialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"

def LLZK_FeltType : TypeDef<FeltDialect, "Felt"> {
  let mnemonic = "type";
  let summary = "finite field element";
  let description = [{
    An element of a finite field. The field element is either of the specified
    field or of an unspecified field to be filled in by any backend translations.

    Field names are either built-in or specified using `felt.field` specification
    attributes on the root module.
  }];

  let parameters = (ins OptionalParameter<"::mlir::StringAttr">:$fieldName);

  let assemblyFormat = "(`<` $fieldName^ `>`)?";

  // Default when omitted: felt.type with no fieldName
  let builderCall = "::llzk::felt::FeltType::get($_builder.getContext(), "
                    "::mlir::StringAttr())";

  let extraClassDeclaration = [{
    static ::llzk::felt::FeltType get(::mlir::MLIRContext *context) {
      return ::llzk::felt::FeltType::get(context, ::mlir::StringAttr());
    }

    const ::llzk::Field &getField() const;
  }];

  let genVerifyDecl = 1;
}

#endif // LLZK_FELT_TYPES
