//===-- SMTInfoAttributes.cpp - SMT script metadata attributes --*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/SMTInfo/IR/SMTInfoAttributes.h"

#include "SMTInfoDetail.h"

#include "llzk/Dialect/SMTInfo/IR/SMTInfoDialect.h"

#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

using namespace mlir;
using namespace llzk::smt_info;

#define GET_ATTRDEF_CLASSES
#include "llzk/Dialect/SMTInfo/IR/SMTInfoAttributes.cpp.inc"

namespace {

bool isValidAtomChar(char ch) {
  return llvm::isAlnum(ch) || ch == '_' || ch == '.' || ch == '$' || ch == '-' || ch == '!';
}

LogicalResult
verifySymbol(function_ref<InFlightDiagnostic()> emitError, StringRef text, bool keyword) {
  if (text.empty()) {
    return emitError() << "symbol text must not be empty";
  }
  if (keyword) {
    if (!text.starts_with(':')) {
      return emitError() << "keyword must start with ':'";
    }
    text = text.drop_front();
  } else if (text.starts_with(':')) {
    return emitError() << "symbol must not start with ':'";
  }
  if (text.empty()) {
    return emitError() << "keyword must contain characters after ':'";
  }
  for (char ch : text) {
    if (!isValidAtomChar(ch)) {
      return emitError() << "invalid SMT-LIB symbol character '" << ch << '\'';
    }
  }
  return success();
}

} // namespace

LogicalResult KeywordAttr::verify(function_ref<InFlightDiagnostic()> emitError, StringRef value) {
  return verifySymbol(emitError, value, true);
}

Attribute KeywordAttr::parse(AsmParser &parser, Type) {
  SMLoc loc = parser.getCurrentLocation();
  StringRef value;
  if (parser.parseLess() || parser.parseColon() || parser.parseKeyword(&value) ||
      parser.parseGreater()) {
    return {};
  }
  return parser.getChecked<KeywordAttr>(loc, parser.getContext(), (":" + value).str());
}

void KeywordAttr::print(AsmPrinter &printer) const { printer << '<' << getValue() << '>'; }

LogicalResult SymbolAttr::verify(function_ref<InFlightDiagnostic()> emitError, StringRef value) {
  return verifySymbol(emitError, value, false);
}

Attribute SymbolAttr::parse(AsmParser &parser, Type) {
  SMLoc loc = parser.getCurrentLocation();
  StringRef value;
  if (parser.parseLess() || parser.parseKeyword(&value) || parser.parseGreater()) {
    return {};
  }
  return parser.getChecked<SymbolAttr>(loc, parser.getContext(), value.str());
}

void SymbolAttr::print(AsmPrinter &printer) const { printer << '<' << getValue() << '>'; }

Attribute llzk::smt_info::detail::getKeywordAttr(MLIRContext *context, StringRef value) {
  return KeywordAttr::get(context, value);
}

Attribute llzk::smt_info::detail::getSymbolAttr(MLIRContext *context, StringRef value) {
  return SymbolAttr::get(context, value);
}

LogicalResult llzk::smt_info::detail::verifyKeyword(
    function_ref<InFlightDiagnostic()> emitError, StringRef value
) {
  return verifySymbol(emitError, value, true);
}

void SMTInfoDialect::registerAttributes() {
  // clang-format off
  // Suppress false positive from `clang-tidy`.
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "llzk/Dialect/SMTInfo/IR/SMTInfoAttributes.cpp.inc"
  >();
  // clang-format on
}
