//===-- Attrs.td -----------------------------------------*--- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef PCL_ATTRS
#define PCL_ATTRS

include "Dialect.td"
include "mlir/IR/AttrTypeBase.td"

def FeltAttr : AttrDef<PCLDialect, "Felt"> {
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

def PrimeAttr : AttrDef<PCLDialect, "Prime"> {
  let mnemonic = "prime"; // prints as #pcl.prime<...>
  let summary = "Prime field modulus for this module";
  let description = [{
    Attach this as a named attribute (e.g. pcl.prime) on builtin.module.
    Use an APInt literal so very large primes are supported.
  }];
  let parameters = (ins "::mlir::IntegerAttr":$value);
  let assemblyFormat = "`<`$value`>`";
}

#endif // PCL_ATTRS
