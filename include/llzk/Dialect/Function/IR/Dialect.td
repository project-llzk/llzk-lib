//===-- Dialect.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_FUNCTION_DIALECT
#define LLZK_FUNCTION_DIALECT

include "mlir/IR/DialectBase.td"

def FunctionDialect : Dialect {
  let name = "function";
  let summary = "LLZK function operations.";

  let cppNamespace = "::llzk::function";

  let useDefaultAttributePrinterParser = true;
  let hasOperationAttrVerify = 1;
}

#endif // LLZK_FUNCTION_DIALECT
