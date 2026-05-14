//===-- MLIRAsmParserCompat.h - MLIR parser compatibility ------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Compatibility specializations for MLIR ODS field parsing of builtin
/// attributes that LLZK textual syntax prints in stripped form.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectImplementation.h>

namespace mlir {

/// Parse a stripped string attribute such as `"babybear"`.
template <>
struct FieldParser<StringAttr> {
  static FailureOr<StringAttr> parse(AsmParser &parser) {
    StringAttr value;
    OptionalParseResult result = parser.parseOptionalAttribute(value);
    if (!result.has_value() || failed(*result)) {
      return failure();
    }
    return value;
  }
};

/// Parse a stripped symbol reference attribute such as `@Main::@Child`.
template <>
struct FieldParser<SymbolRefAttr> {
  static FailureOr<SymbolRefAttr> parse(AsmParser &parser) {
    SymbolRefAttr value;
    OptionalParseResult result = parser.parseOptionalAttribute(value);
    if (!result.has_value() || failed(*result)) {
      return failure();
    }
    return value;
  }
};

/// Parse a stripped flat symbol reference attribute such as `@X`.
template <>
struct FieldParser<FlatSymbolRefAttr> {
  static FailureOr<FlatSymbolRefAttr> parse(AsmParser &parser) {
    FlatSymbolRefAttr value;
    OptionalParseResult result = parser.parseOptionalAttribute(value);
    if (!result.has_value() || failed(*result)) {
      return failure();
    }
    return value;
  }
};

} // namespace mlir
