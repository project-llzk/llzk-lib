//===-- Attrs.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_FUNCTION_ATTRS
#define LLZK_FUNCTION_ATTRS

include "llzk/Dialect/Function/IR/Dialect.td"
include "llzk/Dialect/Function/IR/Attrs.td" // must be included to generate docs

include "mlir/IR/AttrTypeBase.td"

class FunctionDialectAttr<string name, string attrMnemonic,
                          list<Trait> traits = []>
    : AttrDef<FunctionDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def LLZK_DeterministicAttr
    : FunctionDialectAttr<"Deterministic", "deterministic"> {
  let summary = "A unit attribute marking repeatable function results";
  let description = [{
    The unit attribute `function.deterministic` may be attached to any `function.def`,
    including struct `compute` and `constrain` functions, free functions, and
    external function declarations. It promises that repeated calls with the
    same arguments return the same results.

    It does not imply that the function is free of side effects.

    Example:

    ```llzk
    function.def private @external_fn(!felt.type) -> !felt.type
        attributes {function.deterministic}
    ```
  }];
}

def LLZK_AllowConstraintAttr
    : FunctionDialectAttr<"AllowConstraint", "allow_constraint"> {
  let summary =
      "Marks functions to allow `ConstraintGen` ops within their body";
  let description = [{
    A unit attribute that can be attached to a `FuncDefOp` to indicate that ops
    marked with `ConstraintGen` (e.g., emit ops) are allowed within its body.
  }];
}

def LLZK_AllowWitnessAttr
    : FunctionDialectAttr<"AllowWitness", "allow_witness"> {
  let summary = "Marks functions to allow `WitnessGen` ops within their body";
  let description = [{
    A unit attribute that can be attached to a `FuncDefOp` to indicate that ops marked
    with `WitnessGen` (e.g., construct and write ops) are allowed within its body.
  }];
}

def LLZK_AllowNonNativeFieldOpsAttr
    : FunctionDialectAttr<"AllowNonNativeFieldOps",
                          "allow_non_native_field_ops"> {
  let summary =
      "Marks functions to allow `NotFieldNative` ops within their body";
  let description = [{
    A unit attribute that can be attached to a `FuncDefOp` to indicate that ops marked
    with `NotFieldNative` (e.g., boolean and bitwise ops) are allowed within its body.
  }];
}

def LLZK_AllowVerifOpsAttr
    : FunctionDialectAttr<"AllowVerifOps", "allow_verif_ops"> {
  let summary = "Marks functions to allow certain `verif` dialect operations "
                "within their body";
  let description = [{
    A unit attribute that can be attached to a `FuncDefOp` to indicate that ops
    marked with `Verification` are allowed within its body.
  }];
}

#endif // LLZK_FUNCTION_ATTRS
