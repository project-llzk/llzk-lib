//===-- SMTInfoOps.td - Operations for SMT metadata --------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_DIALECT_SMTINFO_IR_SMTINFOOPS_TD
#define LLZK_DIALECT_SMTINFO_IR_SMTINFOOPS_TD

include "SMTInfoAttributes.td"
include "mlir/IR/OpBase.td"

class SMTInfoOp<string mnemonic, list<Trait> traits = []>
    : Op<SMTInfoDialect, mnemonic, traits>;

def SMTInfoSetOp : SMTInfoOp<"set"> {
  let summary = "attach SMT-LIB set-info metadata to a script";
  let arguments = (ins SMTInfoKeywordAttr:$key, AnyAttr:$value);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // LLZK_DIALECT_SMTINFO_IR_SMTINFOOPS_TD
