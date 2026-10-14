//===-- AttributeHelper.td ---------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_ATTRIBUTE_HELPER
#define LLZK_ATTRIBUTE_HELPER

include "mlir/IR/AttrTypeBase.td"

// Signed mathematical integer with representation-independent equality and
// hashing.
class DynamicAPIntParameter<string desc = "">
    : AttrOrTypeParameter<"::llzk::DynamicAPIntValue", desc,
                          "const ::llvm::DynamicAPInt &"> {
  let parser = "::llzk::parseDynamicAPIntValue($_parser)";
  let printer = "$_printer << $_self";
  let comparator = "$_lhs == $_rhs";
}

#endif // LLZK_ATTRIBUTE_HELPER
