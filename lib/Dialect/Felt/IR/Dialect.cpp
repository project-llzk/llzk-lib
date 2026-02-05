//===-- Dialect.cpp - Felt dialect implementation ---------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Util/Field.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/LLZK/IR/Versioning.h"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

// TableGen'd implementation files
#include "llzk/Dialect/Felt/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/Felt/IR/Types.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "llzk/Dialect/Felt/IR/Attrs.cpp.inc"

namespace llzk::felt {

//===------------------------------------------------------------------===//
// FieldSpecAttr
//
// Custom parse/print needs to be here where Attrs.cpp.inc is included, and
// it doesn't work to put it in Attrs.cpp.
//===------------------------------------------------------------------===//

mlir::Attribute FieldSpecAttr::parse(mlir::AsmParser &odsParser, mlir::Type _) {
  mlir::Builder odsBuilder(odsParser.getContext());
  llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  mlir::FailureOr<::mlir::StringAttr> fieldNameAttrRes;
  mlir::FailureOr<::llvm::APInt> primeRes;

  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'fieldName'
  fieldNameAttrRes = mlir::FieldParser<::mlir::StringAttr>::parse(odsParser);
  if (::mlir::failed(fieldNameAttrRes)) {
    odsParser.emitError(
        odsParser.getCurrentLocation(),
        "failed to parse LLZK_FieldSpecAttr parameter 'fieldName' which is to be a `::mlir::StringAttr`"
    );
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) {
    return {};
  }

  // Parse variable 'prime'
  primeRes = mlir::FieldParser<::llvm::APInt>::parse(odsParser);
  if (::mlir::failed(primeRes)) {
    odsParser.emitError(
        odsParser.getCurrentLocation(),
        "failed to parse LLZK_FieldSpecAttr parameter 'prime' which is to be a `::llvm::APInt`"
    );
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) {
    return {};
  }
  assert(mlir::succeeded(fieldNameAttrRes));
  assert(mlir::succeeded(primeRes));

  // Custom logic: cache the field, reporting an error if there's a conflict
  auto errFn = [&odsParser]() {
    return InFlightDiagnosticWrapper(odsParser.emitError(odsParser.getCurrentLocation()));
  };
  Field::addField(fieldNameAttrRes.value(), primeRes.value(), errFn);

  return odsParser.getChecked<FieldSpecAttr>(
      odsLoc, odsParser.getContext(), mlir::StringAttr((*fieldNameAttrRes)),
      llvm::APInt((*primeRes))
  );
}

void FieldSpecAttr::print(mlir::AsmPrinter &odsPrinter) const {
  mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getFieldName());
  odsPrinter << ", ";
  odsPrinter.printStrippedAttrOrType(getPrime());
  odsPrinter << '>';
}

//===------------------------------------------------------------------===//
// FeltDialect
//===------------------------------------------------------------------===//

auto FeltDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/Felt/IR/Ops.cpp.inc"
  >();

  addTypes<
    #define GET_TYPEDEF_LIST
    #include "llzk/Dialect/Felt/IR/Types.cpp.inc"
  >();

  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "llzk/Dialect/Felt/IR/Attrs.cpp.inc"
  >();
  // clang-format on
  addInterfaces<LLZKDialectBytecodeInterface<FeltDialect>>();
}

} // namespace llzk::felt
