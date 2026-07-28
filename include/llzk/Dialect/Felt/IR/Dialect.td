//===-- Dialect.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_FELT_DIALECT
#define LLZK_FELT_DIALECT

include "mlir/IR/DialectBase.td"

def FeltDialect : Dialect {
  let name = "felt";
  let summary = "LLZK felt (Field ELemenT) type and operations.";

  let cppNamespace = "::llzk::felt";

  let useDefaultTypePrinterParser = true;
  let useDefaultAttributePrinterParser = true;
  let hasConstantMaterializer = 1;
}

#endif // LLZK_FELT_DIALECT
