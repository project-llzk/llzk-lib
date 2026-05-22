//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_CAST_OPS
#define LLZK_CAST_OPS

include "llzk/Dialect/Shared/Types.td"
include "llzk/Dialect/Cast/IR/Dialect.td"
include "llzk/Dialect/Cast/IR/Attrs.td"
include "llzk/Dialect/Felt/IR/Types.td"
include "llzk/Dialect/Shared/OpsBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "llzk/Dialect/Function/IR/OpTraits.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

class CastOp<string mnemonic, list<Trait> traits = []>
    : NaryOpBase<CastDialect, mnemonic, LLZK_FeltType.builderCall, traits>;

def LLZK_IntToFeltOp
    : CastOp<
          "tofelt", [Pure,
                     DeclareOpInterfaceMethods<
                         InferTypeOpInterface, ["isCompatibleReturnTypes"]>]> {
  let summary = "convert an integer into a field element";
  let description = [{
    This operation converts a supported integer type value into a field element value.

    Example:

    ```llzk
    %0 = bool.cmp lt(%a, %b)
    %1 = cast.tofelt %0 : i1
    ```
  }];

  let arguments = (ins DefaultValuedAttr<LLZK_OverflowSemanticsAttr,
                                         "OverflowSemantics::ASSERT">:$overflow,
      AnyLLZKIntType:$value);
  let results = (outs LLZK_FeltType:$result);
  let assemblyFormat =
      [{`` custom<OptionalOverflowSemantics>($overflow) $value `:` type($value) `` custom<InferredOrParsedType>(type($result), "false") attr-dict }];

  // Children of NaryOpBase must add their extra C++ code in the extra-extra
  // version.
  let extraExtraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    static ::mlir::ParseResult parseOptionalOverflowSemantics(
        ::mlir::OpAsmParser &parser, ::llzk::cast::OverflowSemanticsAttr &overflow
    );

    static void printOptionalOverflowSemantics(
        ::mlir::OpAsmPrinter &printer, IntToFeltOp op,
        ::llzk::cast::OverflowSemanticsAttr overflow
    );
  }];

  let hasCanonicalizeMethod = 1;
}

def LLZK_FeltToIndexOp : CastOp<"toindex", [Pure, NotFieldNative]> {
  let summary = "convert a field element into an index";
  let description = [{
    This operation converts a field element value into an index value to allow use
    as an array index or loop bound.

    Example:
    ```llzk
    %0 = cast.toindex %a
    %1 = array.read %b[%0]
    ```
  }];

  let arguments = (ins DefaultValuedAttr<LLZK_OverflowSemanticsAttr,
                                         "OverflowSemantics::ASSERT">:$overflow,
      LLZK_FeltType:$value);
  let results = (outs Index:$result);
  let assemblyFormat =
      [{`` custom<OptionalOverflowSemantics>($overflow) $value `` custom<InferredOrParsedType>(type($value), "true") attr-dict }];

  let extraExtraClassDeclaration = [{
    static ::mlir::ParseResult parseOptionalOverflowSemantics(
        ::mlir::OpAsmParser &parser, ::llzk::cast::OverflowSemanticsAttr &overflow
    );
  
    static void printOptionalOverflowSemantics(
        ::mlir::OpAsmPrinter &printer, FeltToIndexOp op,
        ::llzk::cast::OverflowSemanticsAttr overflow
    );
  }];

  let hasCanonicalizeMethod = 1;
}

#endif // LLZK_CAST_OPS
