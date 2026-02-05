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

  let parameters = (ins APIntParameter<"The felt constant value">:$value);

  let returnType = "::llvm::APInt";
  let convertFromStorage = "$_self.getValue()";

  let assemblyFormat = [{ $value }];

  let builders =
      [AttrBuilder<(ins "unsigned":$numBits, "::llvm::StringRef":$str), [{
      return $_get(context, ::llvm::APInt(numBits, str, 10));
    }]>,
       AttrBuilder<
           (ins "unsigned":$numBits, "::llvm::ArrayRef<uint64_t>":$parts), [{
      return $_get(context, ::llvm::APInt(numBits, parts));
    }]>];

  let extraClassDeclaration = [{
    ::mlir::Type getType() const;
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
    - koalabear
    - mersenne31
  }];

  let parameters = (ins "::mlir::StringAttr":$fieldName, APIntParameter<"The prime modulus">:$prime);

  // Format is [{ `<` $fieldName `,` $prime `>` }], but with custom parsing
  // to enable caching of the Field object definition.
  let hasCustomAssemblyFormat = 1;

  let builders =
      [AttrBuilder<(ins "::llvm::StringRef":$fieldName, "unsigned":$numBits, "::llvm::StringRef":$primeStr), [{
      return $_get(context, ::mlir::StringAttr::get(context, fieldName), ::llvm::APInt(numBits, primeStr, 10));
    }]>,
       AttrBuilder<
           (ins "::llvm::StringRef":$fieldName, "unsigned":$numBits, "::llvm::ArrayRef<uint64_t>":$parts), [{
      return $_get(context, ::mlir::StringAttr::get(context, fieldName), ::llvm::APInt(numBits, parts));
    }]>];
}

#endif // LLZK_FELT_ATTRS
