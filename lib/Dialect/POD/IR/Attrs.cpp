//===-- Attrs.cpp - POD attributes implementation ---------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/POD/IR/Attrs.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/Support/Debug.h>

using namespace mlir;

namespace llzk::pod {

//===----------------------------------------------------------------------===//
// RecordAttr
//===----------------------------------------------------------------------===//

FlatSymbolRefAttr RecordAttr::getNameSym() const { return FlatSymbolRefAttr::get(getName()); }

ParseResult parseRecord(AsmParser &parser, StringAttr &name, Type &type) {
  auto result = parser.parseSymbolName(name);
  if (mlir::failed(result)) {
    return result;
  }
  return parser.parseColonType(type);
}

void printRecord(AsmPrinter &printer, StringAttr name, Type type) {
  printer.printSymbolName(name.getValue());
  printer << ": ";
  printer.printType(type);
}

} // namespace llzk::pod
