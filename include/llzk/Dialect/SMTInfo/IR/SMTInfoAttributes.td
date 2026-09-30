//===-- SMTInfoAttributes.td - Attributes for SMT metadata -*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_DIALECT_SMTINFO_IR_SMTINFOATTRIBUTES_TD
#define LLZK_DIALECT_SMTINFO_IR_SMTINFOATTRIBUTES_TD

include "SMTInfoDialect.td"
include "mlir/IR/AttrTypeBase.td"

def SMTInfoKeywordAttr : AttrDef<SMTInfoDialect, "Keyword"> {
  let mnemonic = "keyword";
  let summary = "an SMT-LIB keyword";
  let parameters = (ins StringRefParameter<"The keyword text">:$value);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def SMTInfoSymbolAttr : AttrDef<SMTInfoDialect, "Symbol"> {
  let mnemonic = "symbol";
  let summary = "an SMT-LIB bare symbol";
  let parameters = (ins StringRefParameter<"The symbol text">:$value);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

#endif // LLZK_DIALECT_SMTINFO_IR_SMTINFOATTRIBUTES_TD
