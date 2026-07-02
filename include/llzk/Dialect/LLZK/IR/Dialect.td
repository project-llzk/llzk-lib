//===-- Dialect.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_DIALECT
#define LLZK_DIALECT

include "mlir/IR/DialectBase.td"

def LLZKDialect : Dialect {
  let name = "llzk";
  let summary = "Common LLZK attributes.";

  let cppNamespace = "::llzk";

  let useDefaultAttributePrinterParser = true;
  let hasOperationAttrVerify = 1;
  let dependentDialects = ["::llzk::felt::FeltDialect"];
}

#endif // LLZK_DIALECT
