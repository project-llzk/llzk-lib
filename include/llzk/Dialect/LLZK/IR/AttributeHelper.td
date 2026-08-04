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

// APInts whose numeric value is independent of bitwidth.
class APIntParameter<string desc = "">
    : AttrOrTypeParameter<"::llzk::APIntValue", desc, "const ::llvm::APInt &"> {
  let parser = "::llzk::parseAPIntValue($_parser)";
  let printer = "$_printer << $_self";
}

// APInts whose bitwidth is part of their identity. Check it before operator==,
// which requires equal widths.
class WidthSensitiveAPIntParameter<string desc = "">
    : AttrOrTypeParameter<"::llvm::APInt", desc> {
  let comparator =
      "$_lhs.getBitWidth() == $_rhs.getBitWidth() && $_lhs == $_rhs";
}

#endif // LLZK_ATTRIBUTE_HELPER
