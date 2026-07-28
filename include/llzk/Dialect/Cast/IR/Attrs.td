//===-- Attrs.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_CAST_ATTRS
#define LLZK_CAST_ATTRS

include "llzk/Dialect/Cast/IR/Dialect.td"
include "llzk/Dialect/Cast/IR/Enums.td"

class CastDialectEnumAttr<EnumAttrInfo enum, string name, string enumMnemonic,
                          list<Trait> traits = []>
    : EnumAttr<CastDialect, enum, name, traits> {
  let mnemonic = enumMnemonic;
}

def LLZK_OverflowSemanticsAttr
    : CastDialectEnumAttr<LLZK_OverflowSemantics, "overflow_semantics",
                          "overflow"> {
  let assemblyFormat = [{ $value }];
}

#endif // LLZK_CAST_ATTRS
