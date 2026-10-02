//===-- Attrs.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_FELT_ATTRS
#define LLZK_FELT_ATTRS

include "llzk/Dialect/Felt/IR/Dialect.td"
include "llzk/Dialect/LLZK/IR/AttributeHelper.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def LLZK_FeltConstAttr
    : AttrDef<FeltDialect, "FeltConst", [TypedAttrInterface]> {
  let mnemonic = "const";
  let summary = "finite field element";
  let description = [{
    A felt attribute stores an exact signed integer literal and its field type.
    Its value is independent of storage width. Consumers reduce the literal
    modulo the field prime before applying field operations; an unspecified
    field does not cause the literal to be reduced during construction.
  }];

  let parameters = (ins DynamicAPIntParameter<"The felt constant value">:$value,
      DefaultValuedParameter<"FeltType", "FeltType::get($_ctxt)">:$type);

  let hasCustomAssemblyFormat = 1;

  let builders =
      [AttrBuilder<
           (ins "const ::llvm::DynamicAPInt &":$value, "FeltType":$ty),
           [{ return $_get(context, ::llzk::DynamicAPIntValue(value), ty); }]>,
       AttrBuilder<(ins "const ::llvm::DynamicAPInt &":$value),
                   [{ return get(context, value, FeltType::get(context)); }]>,
       AttrBuilder<
           (ins "const ::llvm::DynamicAPInt &":$value,
               "::llvm::StringRef":$fieldName),
           [{ return get(context, value, FeltType::get(context, fieldName)); }]>];

  let extraClassDeclaration = [{
    /// Returns the field name from the stored FeltType, or a null StringAttr
    /// if no type is stored.
    ::mlir::StringAttr getFieldName() const;
  }];
}

def LLZK_FieldSpecAttr : AttrDef<FeltDialect, "FieldSpec"> {
  let mnemonic = "field";
  let summary = "prime field specification";
  let description = [{
    A specification of a prime field for use by felt types.

    These specifications are provided in the `llzk.fields` attribute on the root
    module as either a single element or a flat array, for example:

        module attributes {llzk.lang, llzk.fields = field<foo, 7> { ... }
        module attributes {llzk.lang, llzk.fields = [field<>]} { ... }

    Specifications should not be provided for built-in fields, which include:
    - babybear
    - bn128/bn254
    - goldilocks
    - grumpkin
    - koalabear
    - mersenne31
  }];

  let parameters = (ins "::mlir::StringAttr":$fieldName,
      DynamicAPIntParameter<"The prime modulus">:$prime);

  // Format is [{ `<` $fieldName `,` $prime `>` }], but with custom parsing
  // to enable caching of the Field object definition.
  let hasCustomAssemblyFormat = 1;

  let builders = [AttrBuilder<
      (ins "::mlir::StringAttr":$fieldName,
          "const ::llvm::DynamicAPInt &":$prime),
      [{ return $_get(context, fieldName, ::llzk::DynamicAPIntValue(prime)); }]>];
  let genVerifyDecl = 1;
}

#endif // LLZK_FELT_ATTRS
