//===-- Attrs.td -----------------------------------------*--- tablegen -*-===//
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

def FeltAttr : AttrDef<R1CSDialect, "Felt"> {
  let summary = "Field element represented as an arbitrary precision integer";
  let cppClassName = "FeltAttr";
  let mnemonic = "felt";
  let parameters = (ins "::mlir::IntegerAttr":$value);

  let assemblyFormat = "$value";

  let builders = [AttrBuilder<(ins "const ::llvm::APInt &":$v), [{
      return $_get(context, ::mlir::IntegerAttr::get(::mlir::IntegerType::get(context, v.getBitWidth()), v));
    }]>,
                  AttrBuilder<(ins "const ::llvm::APSInt &":$v), [{
      return $_get(context, ::mlir::IntegerAttr::get(::mlir::IntegerType::get(context, v.getBitWidth()), v));
    }]>];
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
