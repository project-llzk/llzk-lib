//===-- SMTInfoOps.cpp - SMT script metadata operations ---------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/SMTInfo/IR/SMTInfoOps.h"

#include "SMTInfoDetail.h"

#include <mlir/IR/Builders.h>

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

#define GET_OP_CLASSES
#include "llzk/Dialect/SMTInfo/IR/SMTInfoOps.cpp.inc"

using namespace mlir;
using namespace llzk::smt_info;

namespace {

static bool isSetInfoValue(Attribute attr) {
  return TypeSwitch<Attribute, bool>(attr)
      .Case<BoolAttr, IntegerAttr, StringAttr, KeywordAttr, SymbolAttr>([](auto) { return true; })
      .Case<ArrayAttr>([](ArrayAttr values) {
    return llvm::all_of(values, isSetInfoValue);
  }).Default([](Attribute) { return false; });
}

static void printValue(AsmPrinter &printer, Attribute value) {
  TypeSwitch<Attribute>(value)
      .Case<KeywordAttr, SymbolAttr>([&](auto attr) { printer << attr.getValue(); })
      .Case<StringAttr, BoolAttr>([&](auto attr) { printer.printAttribute(attr); })
      .Case<IntegerAttr>([&](auto attr) {
    SmallString<32> text;
    attr.getValue().toStringSigned(text);
    printer << text;
  }).Case<ArrayAttr>([&](ArrayAttr values) {
    printer << '(';
    llvm::interleave(values, [&](Attribute attr) { printValue(printer, attr); }, [&] {
      printer << ' ';
    });
    printer << ')';
  });
}

static ParseResult parseValue(OpAsmParser &parser, Attribute &value) {
  Builder builder(parser.getContext());
  if (succeeded(parser.parseOptionalLParen())) {
    SmallVector<Attribute> values;
    while (failed(parser.parseOptionalRParen())) {
      Attribute element;
      if (parseValue(parser, element)) {
        return failure();
      }
      values.push_back(element);
    }
    value = builder.getArrayAttr(values);
    return success();
  }
  if (succeeded(parser.parseOptionalColon())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword)) {
      return failure();
    }
    value = llzk::smt_info::detail::getKeywordAttr(parser.getContext(), (":" + keyword).str());
    return success();
  }
  APInt number;
  if (auto parsed = parser.parseOptionalInteger(number); parsed.has_value()) {
    if (failed(*parsed)) {
      return failure();
    }
    value = IntegerAttr::get(IntegerType::get(parser.getContext(), number.getBitWidth()), number);
    return success();
  }
  StringAttr string;
  if (auto parsed = parser.parseOptionalAttribute(string, Type()); parsed.has_value()) {
    if (failed(*parsed)) {
      return failure();
    }
    value = string;
    return success();
  }
  StringRef symbol;
  if (succeeded(parser.parseOptionalKeyword(&symbol))) {
    value = symbol == "true" || symbol == "false"
                ? Attribute(builder.getBoolAttr(symbol == "true"))
                : llzk::smt_info::detail::getSymbolAttr(parser.getContext(), symbol);
    return success();
  }
  return parser.emitError(parser.getCurrentLocation()) << "expected SMT-LIB set-info value";
}

} // namespace

ParseResult SMTInfoSetOp::parse(OpAsmParser &parser, OperationState &result) {
  SMLoc loc = parser.getCurrentLocation();
  StringAttr key;
  Attribute value;
  if (parser.parseAttribute(key) || parseValue(parser, value) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  if (failed(llzk::smt_info::detail::verifyKeyword([&] {
    return parser.emitError(loc);
  }, key.getValue()))) {
    return failure();
  }
  result.addAttribute(
      "key", llzk::smt_info::detail::getKeywordAttr(parser.getContext(), key.getValue())
  );
  result.addAttribute("value", value);
  result.location = parser.getEncodedSourceLoc(loc);
  return success();
}

void SMTInfoSetOp::print(OpAsmPrinter &printer) {
  printer << ' ';
  printer.printAttribute(StringAttr::get(getContext(), getKey().getValue()));
  printer << ' ';
  printValue(printer, getValueAttr());
  printer.printOptionalAttrDict((*this)->getAttrs(), {"key", "value"});
}

LogicalResult SMTInfoSetOp::verify() {
  if (!isSetInfoValue(getValueAttr())) {
    return emitOpError(
        "requires an SMT-LIB set-info value built from strings, booleans, integers, SMT keywords, "
        "SMT symbols, or nested lists"
    );
  }
  return success();
}
