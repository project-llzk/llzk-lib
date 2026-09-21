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

def OverflowSemanticsDescription {
  string r =
      "Specifies how a cast handles a source value outside the target's safe "
      "range. For a felt target with prime `p`, the safe range is `[0, p-1]`. "
      "For an index target with bitwidth `N`, it is `[0, 2^(N-1)-1]`.";
}

def LLZK_OverflowSemantics
    : I32EnumAttr<"OverflowSemantics", OverflowSemanticsDescription.r,
                  [
                      // Assert (default): Fail when the source value is outside
                      // the safe range.
                      I32EnumAttrCase<"ASSERT", 0, "assert">,
                      // Saturate: Clamp to the safe range.
                      I32EnumAttrCase<"SATURATE", 1, "sat">,
                      // Wrap: Reduce modulo `p` for `tofelt` or modulo
                      // `2^(N-1)` for `toindex`, keeping the result in the
                      // index safe range.
                      I32EnumAttrCase<"WRAP", 2, "wrap">,
                      // Truncate: Keep the low bits. For `tofelt`, the bitwidth
                      // of `p` determines the number of bits. For `toindex`,
                      // `N-1` bits are preserved to keep the result in the
                      // safe range (i.e. requiring the sign bit to be 0). A
                      // truncated felt may be greater than or equal to `p`.
                      I32EnumAttrCase<"TRUNCATE", 3, "trunc">,
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::llzk::cast";
}

#endif // LLZK_CAST_ENUMS
