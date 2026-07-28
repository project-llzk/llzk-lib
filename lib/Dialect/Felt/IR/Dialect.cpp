//===-- Dialect.cpp - Felt dialect implementation ---------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Felt/IR/Dialect.h"

#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/LLZK/IR/Versioning.h"
#include "llzk/Util/ErrorHelper.h"
#include "llzk/Util/Field.h"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

#include <limits>

// TableGen'd implementation files
#include "llzk/Dialect/Felt/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/Felt/IR/Types.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "llzk/Dialect/Felt/IR/Attrs.cpp.inc"

using namespace mlir;

namespace {

/// Denotes which dialect attribute is serialized.
enum class FeltAttrEncoding : uint8_t {
  FeltConst = 0,
  FieldSpec = 1,
};

struct FeltDialectBytecodeInterface
    : public llzk::LLZKDialectBytecodeInterface<llzk::felt::FeltDialect> {
  using llzk::LLZKDialectBytecodeInterface<llzk::felt::FeltDialect>::LLZKDialectBytecodeInterface;

  Attribute readAttribute(DialectBytecodeReader &reader) const final {
    uint64_t encoding;
    if (failed(reader.readVarInt(encoding))) {
      return {};
    }
    if (encoding > std::numeric_limits<uint8_t>::max()) {
      reader.emitError() << "unknown felt attribute encoding: " << encoding;
      return {};
    }

    switch (static_cast<FeltAttrEncoding>(encoding)) {
    case FeltAttrEncoding::FeltConst: {
      FailureOr<APInt> value = llzk::readAPInt(reader);
      llzk::felt::FeltType type;
      if (failed(value) || failed(reader.readType(type))) {
        return {};
      }
      return llzk::felt::FeltConstAttr::get(getContext(), *value, type);
    }
    case FeltAttrEncoding::FieldSpec: {
      StringAttr fieldName;
      if (failed(reader.readAttribute(fieldName))) {
        return {};
      }
      FailureOr<APInt> prime = llzk::readAPInt(reader);
      if (failed(prime)) {
        return {};
      }

      llzk::Field::addField(fieldName.getValue(), *prime, [&reader]() {
        return llzk::InFlightDiagnosticWrapper(reader.emitError());
      });
      return llzk::felt::FieldSpecAttr::get(getContext(), fieldName, *prime);
    }
    }

    reader.emitError() << "unknown felt attribute encoding: " << encoding;
    return {};
  }

  LogicalResult writeAttribute(Attribute attr, DialectBytecodeWriter &writer) const final {
    if (auto feltConst = dyn_cast<llzk::felt::FeltConstAttr>(attr)) {
      writer.writeVarInt(static_cast<uint64_t>(FeltAttrEncoding::FeltConst));
      llzk::writeAPInt(writer, feltConst.getValue());
      writer.writeType(feltConst.getType());
      return success();
    }
    if (auto fieldSpec = dyn_cast<llzk::felt::FieldSpecAttr>(attr)) {
      writer.writeVarInt(static_cast<uint64_t>(FeltAttrEncoding::FieldSpec));
      writer.writeAttribute(fieldSpec.getFieldName());
      llzk::writeAPInt(writer, fieldSpec.getPrime());
      return success();
    }
    return failure();
  }
};

} // namespace

namespace llzk::felt {

//===------------------------------------------------------------------===//
// FieldSpecAttr
//
// Custom parse/print needs to be here where Attrs.cpp.inc is included, and
// it doesn't work to put it in Attrs.cpp.
//===------------------------------------------------------------------===//

Attribute FieldSpecAttr::parse(AsmParser &odsParser, Type) {
  Builder odsBuilder(odsParser.getContext());
  llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  FailureOr<StringAttr> fieldNameAttrRes;
  FailureOr<llvm::APInt> primeRes;

  // Parse literal '<'
  if (odsParser.parseLess()) {
    return {};
  }

  // Parse variable 'fieldName'
  fieldNameAttrRes = FieldParser<StringAttr>::parse(odsParser);
  if (failed(fieldNameAttrRes)) {
    odsParser.emitError(
        odsParser.getCurrentLocation(), "failed to parse LLZK_FieldSpecAttr parameter 'fieldName' "
                                        "which is to be a `StringAttr`"
    );
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) {
    return {};
  }

  // Parse variable 'prime'
  primeRes = FieldParser<llvm::APInt>::parse(odsParser);
  if (failed(primeRes)) {
    odsParser.emitError(
        odsParser.getCurrentLocation(),
        "failed to parse LLZK_FieldSpecAttr parameter 'prime' which is to be a `llvm::APInt`"
    );
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) {
    return {};
  }
  assert(succeeded(fieldNameAttrRes));
  assert(succeeded(primeRes));

  // Custom logic: cache the field, reporting an error if there's a conflict
  auto errFn = [&odsParser]() {
    return InFlightDiagnosticWrapper(odsParser.emitError(odsParser.getCurrentLocation()));
  };
  Field::addField(fieldNameAttrRes.value(), primeRes.value(), errFn);

  return odsParser.getChecked<FieldSpecAttr>(
      odsLoc, odsParser.getContext(), StringAttr(*fieldNameAttrRes), llvm::APInt(*primeRes)
  );
}

void FieldSpecAttr::print(AsmPrinter &odsPrinter) const {
  Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getFieldName());
  odsPrinter << ", ";
  odsPrinter.printStrippedAttrOrType(getPrime());
  odsPrinter << '>';
}

//===------------------------------------------------------------------===//
// FeltConstAttr
//===------------------------------------------------------------------===//

Attribute FeltConstAttr::parse(AsmParser &odsParser, Type) {
  SMLoc odsLoc = odsParser.getCurrentLocation();

  // Parse the APInt value.
  FailureOr<APInt> valueRes = FieldParser<APInt>::parse(odsParser);
  if (failed(valueRes)) {
    odsParser.emitError(
        odsParser.getCurrentLocation(),
        "failed to parse LLZK_FeltConstAttr parameter 'value' which is to be a `::llvm::APInt`"
    );
    return {};
  }

  FeltType type = FeltType::get(odsParser.getContext());

  // v2 syntax: VALUE : !felt.type<"fieldName">
  if (odsParser.parseOptionalColon().succeeded()) {
    FailureOr<FeltType> typeRes = FieldParser<FeltType>::parse(odsParser);
    if (failed(typeRes)) {
      odsParser.emitError(
          odsParser.getCurrentLocation(),
          "failed to parse LLZK_FeltConstAttr parameter 'type' which is to be a `FeltType`"
      );
      return {};
    }
    type = *typeRes;
  }
  // v1 compat syntax: VALUE <"fieldName">
  else if (odsParser.parseOptionalLess().succeeded()) {
    FailureOr<StringAttr> fieldNameRes = FieldParser<StringAttr>::parse(odsParser);
    if (failed(fieldNameRes)) {
      odsParser.emitError(
          odsParser.getCurrentLocation(), "failed to parse LLZK_FeltConstAttr(version 1) field "
                                          "name parameter which is to be a `StringAttr`"
      );
      return {};
    }
    if (odsParser.parseGreater()) {
      return {};
    }
    type = FeltType::get(odsParser.getContext(), (*fieldNameRes).getValue());
  }

  return odsParser.getChecked<FeltConstAttr>(odsLoc, odsParser.getContext(), *valueRes, type);
}

// Same as tablegen would generate to serialize version 2 IR.
void FeltConstAttr::print(AsmPrinter &odsPrinter) const {
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getValue());
  if (getType() != FeltType::get(getContext())) {
    odsPrinter << " : ";
    odsPrinter.printStrippedAttrOrType(getType());
  }
}

//===------------------------------------------------------------------===//
// FeltType
//===------------------------------------------------------------------===//

const Field &FeltType::getField() const { return Field::getField(getFieldName().getValue()); }

llvm::LogicalResult
FeltType::verify(llvm::function_ref<InFlightDiagnostic()> errFn, StringAttr fieldName) {
  return fieldName ? Field::verifyFieldDefined(
                         fieldName.getValue(), wrapNonNullableInFlightDiagnostic(errFn)
                     )
                   : success();
}

//===------------------------------------------------------------------===//
// FeltDialect
//===------------------------------------------------------------------===//

Operation *
FeltDialect::materializeConstant(OpBuilder &builder, Attribute value, Type, Location loc) {
  if (auto attr = llvm::dyn_cast<FeltConstAttr>(value)) {
    return builder.create<FeltConstantOp>(loc, attr);
  }
  return nullptr;
}

auto FeltDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/Felt/IR/Ops.cpp.inc"
  >();

  // Suppress false positive from `clang-tidy`
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addTypes<
    #define GET_TYPEDEF_LIST
    #include "llzk/Dialect/Felt/IR/Types.cpp.inc"
  >();

  // Suppress false positive from `clang-tidy`
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "llzk/Dialect/Felt/IR/Attrs.cpp.inc"
  >();
  // clang-format on
  addInterfaces<FeltDialectBytecodeInterface>();
}

} // namespace llzk::felt
