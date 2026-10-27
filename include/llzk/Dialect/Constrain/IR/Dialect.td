//===-- Dialect.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_CONSTRAIN_DIALECT
#define LLZK_CONSTRAIN_DIALECT

include "mlir/IR/DialectBase.td"

def ConstrainDialect : Dialect {
  let name = "constrain";
  let summary = "LLZK constraint emission operations.";

  let cppNamespace = "::llzk::constrain";
}

#endif // LLZK_CONSTRAIN_DIALECT
