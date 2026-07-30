//===-- Dialect.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_POLYMORPHIC_DIALECT
#define LLZK_POLYMORPHIC_DIALECT

include "mlir/IR/DialectBase.td"

def PolymorphicDialect : Dialect {
  let name = "poly";
  let summary =
      "LLZK types and operations to support templated/parameterized structs.";

  let cppNamespace = "::llzk::polymorphic";

  let useDefaultTypePrinterParser = true;
  let hasConstantMaterializer = 1;
}

#endif // LLZK_POLYMORPHIC_DIALECT
