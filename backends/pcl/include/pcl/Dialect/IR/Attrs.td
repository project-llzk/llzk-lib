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
include "llzk/Dialect/LLZK/IR/AttributeHelper.td"

def FeltAttr : AttrDef<PCLDialect, "Felt"> {
  let summary = "Field element represented as an arbitrary precision integer";
  let cppClassName = "FeltAttr";
  let mnemonic = "felt";
  let parameters = (ins APIntParameter<"felt constant value">:$value);

  let assemblyFormat = "$value";

  // let builders = [AttrBuilder<(ins "const ::llvm::APInt &":$v), [{
  //     return $_get(context,
  //     ::mlir::IntegerAttr::get(::mlir::IntegerType::get(context,
  //     v.getBitWidth()), v));
  //   }]>,
  //                 AttrBuilder<(ins "const ::llvm::APSInt &":$v), [{
  //     return $_get(context,
  //     ::mlir::IntegerAttr::get(::mlir::IntegerType::get(context,
  //     v.getBitWidth()), v));
  //   }]>];
}

def PrimeAttr : AttrDef<PCLDialect, "Prime"> {
  let mnemonic = "prime"; // prints as #pcl.prime<...>
  let summary = "Prime field modulus for this module";
  let description = [{
    Attach this as a named attribute (e.g. pcl.prime) on builtin.module.
    Use an APInt literal so very large primes are supported.
  }];
  let parameters = (ins APIntParameter<"field prime modulus">:$value);
  let assemblyFormat = "`<`$value`>`";

  let extraClassDeclaration = [{
  /// Returns the bit width of the prime.
  unsigned getBitWidth() { return getValue().getBitWidth(); }

  /// Reduces the given felt such that is within the bounds of the prime.
  ::pcl::FeltAttr reduce(::pcl::FeltAttr attr);
  ::pcl::FeltAttr reduce(::llvm::APInt value) {
    return reduce(::pcl::FeltAttr::get(getContext(), value));
  }
  }];
}

def PCLBoolAttr : AttrDef<PCLDialect, "Bool"> {
  let mnemonic = "bool";
  let cppClassName = "BoolAttr";
  let summary = "boolean value";
  let parameters = (ins "bool":$value);
  let assemblyFormat = "`<` $value `>`";
}

#endif // PCL_ATTRS
