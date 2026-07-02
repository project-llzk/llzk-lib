//===- SMTAttributes.cpp - Implement SMT attributes -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/SMT/IR/SMTAttributes.h"

#include "llzk/Dialect/SMT/IR/SMTDialect.h"
#include "llzk/Dialect/SMT/IR/SMTTypes.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

using namespace mlir;
using namespace llzk::smt;

static bool isValidSMTLibAtomChar(char ch) {
  return llvm::isAlnum(ch) || ch == '_' || ch == '.' || ch == '$' || ch == '-' || ch == '!';
}

static LogicalResult verifySMTLibSymbolText(
    function_ref<InFlightDiagnostic()> emitError, StringRef text, bool requireLeadingColon
) {
  if (text.empty()) {
    return emitError() << "symbol text must not be empty";
  }
  if (requireLeadingColon) {
    if (!text.starts_with(':')) {
      return emitError() << "keyword must start with ':'";
    }
    text = text.drop_front();
    if (text.empty()) {
      return emitError() << "keyword must contain at least one character after ':'";
    }
  } else if (text.starts_with(':')) {
    return emitError() << "symbol must not start with ':'";
  }

  for (char ch : text) {
    if (!isValidSMTLibAtomChar(ch)) {
      return emitError() << "invalid SMT-LIB symbol character '" << ch << "'";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// BitVectorAttr
//===----------------------------------------------------------------------===//

LogicalResult BitVectorAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    APInt value // NOLINT(performance-unnecessary-value-param)
) {
  if (value.getBitWidth() < 1) {
    return emitError() << "bit-width must be at least 1, but got " << value.getBitWidth();
  }
  return success();
}

std::string BitVectorAttr::getValueAsString(bool prefix) const {
  unsigned width = getValue().getBitWidth();
  SmallVector<char> toPrint;
  StringRef pref = prefix ? "#" : "";
  if (width % 4 == 0) {
    getValue().toString(toPrint, 16, false, false, false);
    // APInt's 'toString' omits leading zeros. However, those are critical here
    // because they determine the bit-width of the bit-vector.
    SmallVector<char> leadingZeros(width / 4 - toPrint.size(), '0');
    return (pref + "x" + Twine(leadingZeros) + toPrint).str();
  }

  getValue().toString(toPrint, 2, false, false, false);
  // APInt's 'toString' omits leading zeros
  SmallVector<char> leadingZeros(width - toPrint.size(), '0');
  return (pref + "b" + Twine(leadingZeros) + toPrint).str();
}

/// Parse an SMT-LIB formatted bit-vector string.
static FailureOr<APInt>
parseBitVectorString(function_ref<InFlightDiagnostic()> emitError, StringRef value) {
  auto reportError = [emitError](StringRef msg) -> FailureOr<APInt> {
    if (emitError) {
      return emitError() << msg;
    }
    return failure();
  };

  if (value[0] != '#') {
    return reportError("expected '#'");
  }

  if (value.size() < 3) {
    return reportError("expected at least one digit");
  }

  if (value[1] == 'b') {
    return APInt(value.size() - 2, std::string(value.begin() + 2, value.end()), 2);
  }

  if (value[1] == 'x') {
    return APInt((value.size() - 2) * 4, std::string(value.begin() + 2, value.end()), 16);
  }

  return reportError("expected either 'b' or 'x'");
}

BitVectorAttr BitVectorAttr::get(MLIRContext *context, StringRef value) {
  auto maybeValue = parseBitVectorString(nullptr, value);

  assert(succeeded(maybeValue) && "string must have SMT-LIB format");
  return Base::get(context, *maybeValue);
}

BitVectorAttr BitVectorAttr::getChecked(
    function_ref<InFlightDiagnostic()> emitError, MLIRContext *context, StringRef value
) {
  auto maybeValue = parseBitVectorString(emitError, value);
  if (failed(maybeValue)) {
    return {};
  }

  return Base::getChecked(emitError, context, *maybeValue);
}

BitVectorAttr BitVectorAttr::get(MLIRContext *context, uint64_t value, unsigned width) {
  return Base::get(context, APInt(width, value));
}

BitVectorAttr BitVectorAttr::getChecked(
    function_ref<InFlightDiagnostic()> emitError, MLIRContext *context, uint64_t value,
    unsigned width
) {
  if (width < 64 && value >= (UINT64_C(1) << width)) {
    emitError() << "value does not fit in a bit-vector of desired width";
    return {};
  }
  return Base::getChecked(emitError, context, APInt(width, value));
}

Attribute BitVectorAttr::parse(AsmParser &odsParser, Type odsType) {
  llvm::SMLoc loc = odsParser.getCurrentLocation();

  APInt val;
  if (odsParser.parseLess() || odsParser.parseInteger(val) || odsParser.parseGreater()) {
    return {};
  }

  // Requires the use of `quantified(<attr>)` in operation assembly formats.
  if (!odsType || !llvm::isa<BitVectorType>(odsType)) {
    odsParser.emitError(loc) << "explicit bit-vector type required";
    return {};
  }

  unsigned width = llvm::cast<BitVectorType>(odsType).getWidth();

  if (width > val.getBitWidth()) {
    // sext is always safe here, even for unsigned values, because the
    // parseOptionalInteger method will return something with a zero in the
    // top bits if it is a positive number.
    val = val.sext(width);
  } else if (width < val.getBitWidth()) {
    // The parser can return an unnecessarily wide result.
    // This isn't a problem, but truncating off bits is bad.
    unsigned neededBits = val.isNegative() ? val.getSignificantBits() : val.getActiveBits();
    if (width < neededBits) {
      odsParser.emitError(loc) << "integer value out of range for given bit-vector type "
                               << odsType;
      return {};
    }
    val = val.trunc(width);
  }

  return BitVectorAttr::get(odsParser.getContext(), val);
}

void BitVectorAttr::print(AsmPrinter &odsPrinter) const {
  // This printer only works for the extended format where the MLIR
  // infrastructure prints the type for us. This means, the attribute should
  // never be used without `quantified` in an assembly format.
  odsPrinter << "<" << getValue() << ">";
}

Type BitVectorAttr::getType() const {
  return BitVectorType::get(getContext(), getValue().getBitWidth());
}

//===----------------------------------------------------------------------===//
// KeywordAttr / SymbolAttr
//===----------------------------------------------------------------------===//

LogicalResult KeywordAttr::verify(function_ref<InFlightDiagnostic()> emitError, StringRef value) {
  return verifySMTLibSymbolText(emitError, value, /*requireLeadingColon=*/true);
}

Attribute KeywordAttr::parse(AsmParser &parser, Type) {
  SMLoc loc = parser.getCurrentLocation();
  StringRef keyword;
  if (parser.parseLess() || parser.parseColon() || parser.parseKeyword(&keyword) ||
      parser.parseGreater()) {
    return {};
  }
  return parser.getChecked<KeywordAttr>(loc, parser.getContext(), (":" + keyword).str());
}

void KeywordAttr::print(AsmPrinter &printer) const { printer << '<' << getValue() << '>'; }

LogicalResult SymbolAttr::verify(function_ref<InFlightDiagnostic()> emitError, StringRef value) {
  return verifySMTLibSymbolText(emitError, value, /*requireLeadingColon=*/false);
}

Attribute SymbolAttr::parse(AsmParser &parser, Type) {
  SMLoc loc = parser.getCurrentLocation();
  StringRef symbol;
  if (parser.parseLess() || parser.parseKeyword(&symbol) || parser.parseGreater()) {
    return {};
  }
  return parser.getChecked<SymbolAttr>(loc, parser.getContext(), symbol.str());
}

void SymbolAttr::print(AsmPrinter &printer) const { printer << '<' << getValue() << '>'; }

//===----------------------------------------------------------------------===//
// ODS Boilerplate
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "llzk/Dialect/SMT/IR/SMTAttributes.cpp.inc"

void SMTDialect::registerAttributes() {
  // clang-format off
  // Suppress false positive from `clang-tidy`
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "llzk/Dialect/SMT/IR/SMTAttributes.cpp.inc"
  >();
  // clang-format on
}
