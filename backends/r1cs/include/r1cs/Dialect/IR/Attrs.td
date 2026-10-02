//===-- Attrs.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef R1CS_ATTRS
#define R1CS_ATTRS

include "r1cs/Dialect/IR/Dialect.td"
include "mlir/IR/AttrTypeBase.td"
include "llzk/Dialect/LLZK/IR/AttributeHelper.td"

def FeltAttr : AttrDef<R1CSDialect, "Felt"> {
  let summary = "Field element represented as an arbitrary precision integer";
  let cppClassName = "FeltAttr";
  let mnemonic = "felt";
  let description = [{
    An exact signed mathematical coefficient, independent of storage width.
    Binary export reduces coefficients modulo the selected field prime.
  }];
  let parameters = (ins DynamicAPIntParameter<"signed coefficient">:$value);
  let assemblyFormat = "$value";
}

def PublicAttr : AttrDef<R1CSDialect, "Public"> {
  let summary = "An attribute to mark a type as public";
  let mnemonic = "pub";
  let description = [{
    Examples:

   ```
   %0 = r1cs.def 0 : !r1cs.signal {pub = #r1cs.pub}
    ```
  }];
}

#endif // R1CS_ATTRS
