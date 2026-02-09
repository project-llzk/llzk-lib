//===-- Attrs.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_ATTRS
#define LLZK_ATTRS

include "llzk/Dialect/LLZK/IR/Dialect.td"
include "llzk/Dialect/LLZK/IR/AttributeHelper.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"

class LLZKDialectAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<LLZKDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def LLZK_PublicAttr : LLZKDialectAttr<"Public", "pub"> {
  let summary = "A unit attribute to mark a type as public";
  let description = [{
    Examples:

    ```llzk
    struct.member @member_name : !felt.type {llzk.pub}

    function.def @func_name(%0: !felt.type {llzk.pub})
    ```
  }];
}

def LLZK_LoopBoundsAttr : LLZKDialectAttr<"LoopBounds", "loopbounds"> {
  let summary = "Annotation with the bounds of a loop";
  let description = [{
    This attribute holds information useful for the analysis of loops.
    Holds the bounds of the loop and the step size.

    Example:

    ```llzk
    scf.while ... {
      ...
    } do {
      ...
    } attributes { llzk.loopbounds = #llzk.loopbounds<0 to 10 step 1> }
    ```
  }];

  let parameters =
      (ins APIntParameter<"Loop variable lower bound (inclusive)">:$lower,
          APIntParameter<"Loop variable upper bound (exclusive)">:$upper,
          APIntParameter<"Loop variable step/increment">:$step);

  let assemblyFormat = "`<` $lower `to` $upper `step` $step `>`";
}

#endif // LLZK_ATTRS
