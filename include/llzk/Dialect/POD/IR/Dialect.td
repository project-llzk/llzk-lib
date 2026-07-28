//===-- Dialect.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_DIALECT
#define LLZK_DIALECT

include "mlir/IR/DialectBase.td"

def PODDialect : Dialect {
  let name = "pod";
  let summary = "Dialect for working with plain-old-data structs.";

  let cppNamespace = "::llzk::pod";

  let useDefaultAttributePrinterParser = true;
  let useDefaultTypePrinterParser = true;
}

#endif // LLZK_DIALECT
