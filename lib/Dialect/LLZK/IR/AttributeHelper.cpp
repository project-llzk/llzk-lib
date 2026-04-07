//===-- AttributeHelper.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"

#include "llzk/Util/TypeHelper.h"

using namespace mlir;

namespace llzk {

// Adapted from AsmPrinter::printStrippedAttrOrType(), but without printing type.
void printAttrs(AsmPrinter &printer, ArrayRef<Attribute> attrs, const StringRef &separator) {
  llvm::interleave(attrs, printer.getStream(), [&printer](Attribute a) {
    if (auto intAttr = mlir::dyn_cast_if_present<IntegerAttr>(a)) {
      if (isDynamic(intAttr)) {
        printer.getStream() << "?";
        return;
      }
    }
    if (succeeded(printer.printAlias(a))) {
      return;
    }
    raw_ostream &os = printer.getStream();
    uint64_t posPrior = os.tell();
    printer.printAttributeWithoutType(a);
    // Fallback to printing with prefix if the above failed to write anything to the output stream.
    if (posPrior == os.tell()) {
      printer << a;
    }
  }, separator);
}

} // namespace llzk
