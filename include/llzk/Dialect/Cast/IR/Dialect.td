//===-- Dialect.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_CAST_DIALECT
#define LLZK_CAST_DIALECT

include "mlir/IR/DialectBase.td"

def CastDialect : Dialect {
  let name = "cast";
  let summary = "LLZK type conversion operations.";

  let cppNamespace = "::llzk::cast";
  let useDefaultAttributePrinterParser = true;

  let dependentDialects = ["::llzk::felt::FeltDialect",
                           "::mlir::arith::ArithDialect"];
}

#endif // LLZK_CAST_DIALECT
