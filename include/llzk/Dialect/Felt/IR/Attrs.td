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
    A felt attribute represents a finite field element.
  }];

  let parameters = (ins APIntParameter<"The felt constant value">:$value,
      DefaultValuedParameter<"FeltType", "FeltType::get($_ctxt)">:$type);

  let hasCustomAssemblyFormat = 1;

  let builders = [
      // Preserve the APInt-facing builder while using a canonical numeric
      // storage key.
      AttrBuilder<
          (ins "::llvm::APInt":$value, "FeltType":$ty),
          [{ return $_get(context, ::llzk::APIntValue(std::move(value)), ty); }]>,
      // Like the default builder but with convenience overloads for
      // constructing the APInt.
      AttrBuilder<
          (ins "unsigned":$numBits, "::llvm::StringRef":$str, "FeltType":$ty),
          [{ return $_get(context, ::llvm::APInt(numBits, str, 10), ty); }]>,
      AttrBuilder<
          (ins "unsigned":$numBits, "::llvm::ArrayRef<uint64_t>":$parts,
              "FeltType":$ty),
          [{ return $_get(context, ::llvm::APInt(numBits, parts), ty); }]>,
      // All APInt combinations but using felt type with unspecified field.
      AttrBuilder<(ins "::llvm::APInt":$value),
                  [{ return get(context, value, FeltType::get(context)); }]>,
      AttrBuilder<
          (ins "unsigned":$numBits, "::llvm::StringRef":$str),
          [{ return get(context, numBits, str, FeltType::get(context)); }]>,
      AttrBuilder<
          (ins "unsigned":$numBits, "::llvm::ArrayRef<uint64_t>":$parts),
          [{ return get(context, numBits, parts, FeltType::get(context)); }]>,
      // All APInt combinations but using the field name to construct felt type.
      AttrBuilder<
          (ins "::llvm::APInt":$value, "::llvm::StringRef":$fieldName),
          [{ return get(context, value, FeltType::get(context, fieldName)); }]>,
      AttrBuilder<
          (ins "unsigned":$numBits, "::llvm::StringRef":$str,
              "::llvm::StringRef":$fieldName),
          [{ return get(context, numBits, str, FeltType::get(context, fieldName)); }]>,
      AttrBuilder<
          (ins "unsigned":$numBits, "::llvm::ArrayRef<uint64_t>":$parts,
              "::llvm::StringRef":$fieldName),
          [{ return get(context, numBits, parts, FeltType::get(context, fieldName)); }]>];

  let extraClassDeclaration = [{
    operator ::llvm::APInt() const { return getValue(); }

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
      APIntParameter<"The prime modulus">:$prime);

  // Format is [{ `<` $fieldName `,` $prime `>` }], but with custom parsing
  // to enable caching of the Field object definition.
  let hasCustomAssemblyFormat = 1;

  let builders =
      [AttrBuilder<
           (ins "::mlir::StringAttr":$fieldName, "::llvm::APInt":$prime),
           [{ return $_get(context, fieldName, ::llzk::APIntValue(std::move(prime))); }]>,
       AttrBuilder<(ins "::llvm::StringRef":$fieldName, "unsigned":$numBits,
                       "::llvm::StringRef":$primeStr),
                   [{
      return $_get(context, ::mlir::StringAttr::get(context, fieldName), ::llvm::APInt(numBits, primeStr, 10));
    }]>,
       AttrBuilder<(ins "::llvm::StringRef":$fieldName, "unsigned":$numBits,
                       "::llvm::ArrayRef<uint64_t>":$parts),
                   [{
      return $_get(context, ::mlir::StringAttr::get(context, fieldName), ::llvm::APInt(numBits, parts));
    }]>];
}

#endif // LLZK_FELT_ATTRS
