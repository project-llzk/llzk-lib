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
  let summary = "A unit attribute to mark an input/output as public";
  let description = [{
    This attribute is used on parameters of `compute()` and `constrain()` functions
    within a `struct.def` to denote the public inputs of the circuit and on
    `struct.member` ops to denote public output of the circuit.

    Examples:

    ```llzk
    struct.member @member_name : !felt.type {llzk.pub}

    function.def @compute(%0: !felt.type {llzk.pub})
    ```
  }];
}

def LLZK_LoopBoundsAttr : LLZKDialectAttr<"LoopBounds", "loopbounds"> {
  let summary = "Trusted annotation with the bounds of a loop";
  let description = [{
    This attribute indicates the loop variable's lower bound (inclusive), upper
    bound (exclusive), and nonzero step size. Positive and negative steps are
    allowed; a zero step is a verification error.

    All three values are signed mathematical integers of arbitrary precision,
    not field elements. They are interpreted without reduction modulo any field,
    even when the loop variable has a felt type. For example, a step equal to
    the field modulus denotes that integer increment, not zero.

    These attributes are trusted unconditionally, so the producer must ensure
    these values agree with the loop's behavior. Passes are not required to
    produce an error diagnostic if the values specified by the attribute
    contradict the IR itself; incorrect annotations can change program behavior.

    Example:

    ```llzk
    scf.while ... {
      ...
    } do {
      ...
    } attributes { llzk.loopbounds = #llzk.loopbounds<0 to 10 step 1> }
    ```
  }];

  let parameters = (ins DynamicAPIntParameter<
                        "Loop variable lower bound (inclusive)">:$lower,
      DynamicAPIntParameter<"Loop variable upper bound (exclusive)">:$upper,
      DynamicAPIntParameter<"Nonzero loop variable step/increment">:$step);

  let builders = [AttrBuilder<(ins "const ::llvm::DynamicAPInt &":$lower,
                                  "const ::llvm::DynamicAPInt &":$upper,
                                  "const ::llvm::DynamicAPInt &":$step),
                              [{ return $_get(context,
          ::llzk::DynamicAPIntValue(lower),
          ::llzk::DynamicAPIntValue(upper),
          ::llzk::DynamicAPIntValue(step)); }]>];

  let assemblyFormat = "`<` $lower `to` $upper `step` $step `>`";
  let genVerifyDecl = 1;
}

#endif // LLZK_ATTRS
