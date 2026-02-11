//===-- Types.cpp - Struct type implementations -----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Types.h"

using namespace mlir;

namespace llzk::component {

ParseResult parseStructParams(AsmParser &parser, ArrayAttr &value) {
  auto parseResult = FieldParser<ArrayAttr>::parse(parser);
  if (failed(parseResult)) {
    return parser.emitError(parser.getCurrentLocation(), "failed to parse struct parameters");
  }
  auto emitError = [&parser] {
    return InFlightDiagnosticWrapper(parser.emitError(parser.getCurrentLocation()));
  };
  FailureOr<SmallVector<Attribute>> res = forceIntAttrTypes(parseResult->getValue(), emitError);
  if (failed(res)) {
    return failure();
  }
  value = parser.getBuilder().getArrayAttr(*res);
  return success();
}

void printStructParams(AsmPrinter &printer, ArrayAttr value) {
  printer << '[';
  printAttrs(printer, value.getValue(), ", ");
  printer << ']';
}

LogicalResult StructType::verify(
    function_ref<InFlightDiagnostic()> emitError, SymbolRefAttr nameRef, ArrayAttr params
) {
  return verifyStructTypeParams(wrapNonNullableInFlightDiagnostic(emitError), params);
}

FailureOr<SymbolLookupResult<StructDefOp>> StructType::getDefinition(
    SymbolTableCollection &symbolTable, Operation *op, bool reportMissing
) const {
  // First ensure this StructType passes verification
  ArrayAttr typeParams = this->getParams();
  if (failed(StructType::verify([op] { return op->emitError(); }, getNameRef(), typeParams))) {
    return failure();
  }
  // Perform lookup and ensure the symbol references a StructDefOp
  auto res = lookupTopLevelSymbol<StructDefOp>(symbolTable, getNameRef(), op, reportMissing);
  if (failed(res) || !res.value()) {
    if (reportMissing) {
      return op->emitError() << "could not find '" << StructDefOp::getOperationName()
                             << "' named \"" << getNameRef() << '"';
    } else {
      return failure();
    }
  }
  // If this StructType contains parameters, make sure they match the number from the StructDefOp.
  if (typeParams) {
    auto defParams = res.value().get().getConstParams();
    size_t numExpected = defParams ? defParams->size() : 0;
    if (typeParams.size() != numExpected) {
      return op->emitError() << '\'' << StructType::name << "' type has " << typeParams.size()
                             << " parameters but \"" << res.value().get().getSymName()
                             << "\" expects " << numExpected;
    }
  }
  return res;
}

LogicalResult StructType::verifySymbolRef(SymbolTableCollection &symbolTable, Operation *op) {
  return getDefinition(symbolTable, op);
}

LogicalResult StructType::hasColumns(SymbolTableCollection &symbolTable, Operation *op) const {
  auto lookup = getDefinition(symbolTable, op);
  if (failed(lookup)) {
    return lookup;
  }
  return lookup->get().hasColumns();
}

} // namespace llzk::component
