//===-- Enums.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_CAST_ENUMS
#define LLZK_CAST_ENUMS

include "mlir/IR/EnumAttr.td"

def LLZK_OverflowSemantics
    : I32EnumAttr<"OverflowSemantics", "Cast overflow semantics",
                  [I32EnumAttrCase<"ASSERT", 0, "assert">,
                   I32EnumAttrCase<"SATURATE", 1, "sat">,
                   I32EnumAttrCase<"WRAP", 2, "wrap">,
                   I32EnumAttrCase<"TRUNCATE", 3, "trunc">]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::llzk::cast";
}

#endif // LLZK_CAST_ENUMS
