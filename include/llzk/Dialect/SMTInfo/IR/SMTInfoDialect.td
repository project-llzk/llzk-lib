//===-- SMTInfoDialect.td - SMT metadata dialect -----------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_DIALECT_SMTINFO_IR_SMTINFODIALECT_TD
#define LLZK_DIALECT_SMTINFO_IR_SMTINFODIALECT_TD

include "mlir/IR/DialectBase.td"

def SMTInfoDialect : Dialect {
  let name = "smt_info";
  let summary = "SMT-LIB script metadata retained through emission";
  let cppNamespace = "::llzk::smt_info";
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerAttributes();
  }];
}

#endif // LLZK_DIALECT_SMTINFO_IR_SMTINFODIALECT_TD
