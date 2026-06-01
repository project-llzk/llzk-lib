//===-- Dialect.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_VERIF_DIALECT
#define LLZK_VERIF_DIALECT

include "mlir/IR/DialectBase.td"

def VerifDialect : Dialect {
  let name = "verif";
  let summary = "LLZK verification dialect for encoding specifications for "
                "functions and structs.";

  let cppNamespace = "::llzk::verif";
}

#endif // LLZK_VERIF_DIALECT
