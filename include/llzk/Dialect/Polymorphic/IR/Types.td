//===-- Types.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_POLYMORPHIC_TYPES
#define LLZK_POLYMORPHIC_TYPES

include "llzk/Dialect/Shared/Types.td"
include "llzk/Dialect/Polymorphic/IR/Dialect.td"

class PolymorphicDialectType<string name, string typeMnemonic,
                             list<Trait> traits = []>
    : TypeDef<PolymorphicDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TypeVarSummary { string r = "type variable"; }

def TypeVarType : PolymorphicDialectType<"TypeVar", "tvar"> {
  let summary = TypeVarSummary.r;
  let description = [{
    This type serves as a placeholder for a type that is instantiated
    via a parameter of the struct.

    For example, we can define a struct that holds a size-2 array where the
    type of the elements in the array is specified by a parameter of the
    struct and instantiated with a specific type at the uses of the struct.


    ```llzk
    struct.def @A<[@Ty]> {
      member @x : !array.type<2 x !poly.tvar<@Ty>>
      ...
    }
    ```
  }];

  let parameters =
      (ins TypeParameter<"::mlir::FlatSymbolRefAttr",
                         "reference to the struct parameter">:$nameRef);

  let assemblyFormat = [{ `<` $nameRef `>` }];

  let builders = [TypeBuilderWithInferredContext<
      (ins "::mlir::FlatSymbolRefAttr":$nameRef),
      [{ return $_get(nameRef.getContext(), nameRef); }]>];

  let extraClassDeclaration = [{
    inline ::mlir::StringRef getRefName() const { return getNameRef().getValue(); }
  }];
}

// Type constraint matching only the given Type or TypeVarType.
class VarTypeOr<Type allowedConcreteType, string text = TypeVarSummary.r>
    : Type<CPred<"::llvm::isa<llzk::polymorphic::"
                 "TypeVarType,"#allowedConcreteType.cppType#">($_self)">,
           allowedConcreteType.summary#" or "#text, "::mlir::Type">;

#endif // LLZK_POLYMORPHIC_TYPES
