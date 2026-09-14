//===-- Types.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_STRUCT_TYPES
#define LLZK_STRUCT_TYPES

include "llzk/Dialect/Shared/Types.td"
include "llzk/Dialect/Struct/IR/Dialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class StructDialectType<string name, string typeMnemonic,
                        list<Trait> traits = []>
    : TypeDef<StructDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def LLZK_StructType : StructDialectType<"Struct", "type"> {
  let summary = "circuit component";
  let description = [{
    Type of a `struct.def` instance. An instantiation argument list supplies one attribute per
    `poly.param` in the definition's enclosing `poly.template`, in declaration order. For a
    definition with no parameters, the list may be omitted or explicitly empty. Every
    argument must satisfy the corresponding parameter's type restriction. A flat symbol reference
    may name a `poly.param` or `poly.expr` binding; a constant global reference must be
    fully-qualified. Unresolved symbols, mutable globals, and symbols that resolve to another
    operation kind are rejected. Each attribute must be one of the following:
      - IntegerAttr (with IndexType), specifying a fixed parameter value; the wildcard (`?`) is
        not allowed
      - FeltConstAttr, specifying a fixed finite-field parameter value
      - SymbolRefAttr, specifying a parameter value by a flat reference to a `poly.param` or
        `poly.expr` binding or a fully-qualified reference to a constant global
      - AffineMapAttr, specifying a deferred single-result affine expression for an integer-like
        argument; one use is varying a struct argument across array elements
      - TypeAttr, specifying a type parameter

    A `TypeAttr` argument may contain any valid LLZK type. The verifier recursively resolves
    references in `struct.type` and `array.type` arguments. A `poly.tvar` must name a parameter of
    the use site's enclosing `poly.template`. Record member types within a `pod.type` are not
    traversed.

    The examples below are schematic: `@A` names a non-parameterized struct; `@T` declares index,
    felt, and type parameters in that order, names its index parameter `@N`, and contains `@B`.
    ```llzk
    // Type for struct `A` with no parameters.
    !struct.type<@A>

    // Supply concrete arguments to all three parameters of @T::@B.
    !struct.type<@T::@B<[5, #felt<const 35>, !felt.type]>>

    // An argument can use the flat name of a template binding with a compatible restriction.
    !struct.type<@T::@B<[@N, #felt<const 35>, !felt.type]>>
    ```
  }];

  let parameters =
      (ins TypeParameter<
           "::mlir::SymbolRefAttr",
           "Fully-qualified name of the struct definition.">:$nameRef,
          OptionalParameter<"::mlir::ArrayAttr", "Struct parameters">:$params);

  let assemblyFormat =
      [{ `<` $nameRef ( `<` custom<TemplateParams>($params)^ `>` )? `>` }];

  let genVerifyDecl = 1;

  let skipDefaultBuilders = 1;
  let builders = [TypeBuilderWithInferredContext<
                      (ins "::mlir::SymbolRefAttr":$structName), [{
                    return $_get(structName.getContext(), structName, ::mlir::ArrayAttr());
                  }]>,
                  TypeBuilderWithInferredContext<
                      (ins "::mlir::SymbolRefAttr":$structName,
                          "::mlir::ArrayAttr":$params),
                      [{
                    ::mlir::MLIRContext *ctx = structName.getContext();
                    if (params) {
                      OwningEmitErrorFn emitErrorFn = ::llzk::wrapNullableInFlightDiagnostic(emitError, ctx);
                      auto paramsRes = forceIntAttrTypes(params.getValue(), emitErrorFn);
                      if(::mlir::failed(paramsRes)) { return StructType(); }
                      params = ::mlir::ArrayAttr::get(ctx, *paramsRes);
                    }
                    return $_get(ctx, structName, params);
                  }]>,
                  TypeBuilderWithInferredContext<
                      (ins "::mlir::SymbolRefAttr":$structName,
                          "::llvm::ArrayRef<::mlir::Attribute>":$paramsRef),
                      [{
                    ::mlir::MLIRContext *ctx = structName.getContext();
                    OwningEmitErrorFn emitErrorFn = ::llzk::wrapNullableInFlightDiagnostic(emitError, ctx);
                    auto paramsRes = forceIntAttrTypes(paramsRef, emitErrorFn);
                    if(::mlir::failed(paramsRes)) { return StructType(); }
                    return $_get(ctx, structName, ::mlir::ArrayAttr::get(ctx, *paramsRes));
                  }]>];

  let extraClassDeclaration = [{
    /// Gets the `struct` op that defines this struct. Provided `op` is
    /// used as a starting point for the lookup. Should not be assumed to
    /// be non-`null` as we don't verify all types during verification.
    ::mlir::FailureOr<SymbolLookupResult<StructDefOp>> getDefinition
        (::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op, bool reportMissing = true) const;

    // Verifies that this type references a valid struct, relative to the given `op`.
    ::mlir::LogicalResult verifySymbolRef(::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op);

    /// Returns wether the struct this type refers to has members marked as columns.
    /// A lookup is necessary first and will forward the failure state if it fails.
    mlir::LogicalResult hasColumns(mlir::SymbolTableCollection &symbolTable, mlir::Operation *op) const;
  }];

  let extraClassDefinition = [{
    namespace {
      /// This definition of `emitError` is used by the `get()` functions generated by the
      /// custom builders for this type. The `getChecked()` functions generated by those same
      /// builders have a parameter with this same name that shadows this definition so the
      /// `getChecked()` versions will use the function supplied via the parameter. Regardless,
      /// `wrapNullableInFlightDiagnostic()` checks for `nullptr` and generates a default if
      /// necessary. This approach, although a bit hacky, allows a legitimate error function to
      /// be used whenever available, only reverting to a default in the `get()` function.
      const ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError = nullptr;
    }
  }];
}

#endif // LLZK_STRUCT_TYPES
