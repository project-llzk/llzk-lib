//===-- ZKLeanLeanTypes.td ---------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DIALECT_ZKLEANLEAN_ZKLEANLEANTYPES_TD_
#define LIB_DIALECT_ZKLEANLEAN_ZKLEANLEANTYPES_TD_

include "ZKLeanLeanDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"

// ZKLean struct instance type.
def ZKLeanLean_StructType : TypeDef<ZKLeanLean_Dialect, "Struct"> {
  let summary = "ZKLean struct value type";
  let mnemonic = "type";
  let description = [{
    Type of a ZKLean struct instance. The referenced struct definition is
    identified by a symbol reference.
  }];

  let parameters =
      (ins TypeParameter<
           "::mlir::SymbolRefAttr",
           "Fully-qualified name of the struct definition.">:$nameRef);
  let assemblyFormat = "`<` $nameRef `>`";
}

#endif  // LIB_DIALECT_ZKLEANLEAN_ZKLEANLEANTYPES_TD_
