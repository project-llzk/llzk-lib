//===-- Ops.cpp - POD operation implementations -----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>

#include <cstdint>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/POD/IR/Ops.cpp.inc"

using namespace mlir;

namespace llzk::pod {

//===----------------------------------------------------------------------===//
// NewPodOp
//===----------------------------------------------------------------------===//

namespace {
static void buildCommon(
    OpBuilder &builder, OperationState &state, PodType result, InitializedRecords initialValues
) {
  SmallVector<Value, 4> values;
  SmallVector<StringRef, 4> names;

  for (const auto &record : initialValues) {
    names.push_back(record.name);
    values.push_back(record.value);
  }

  auto &props = state.getOrAddProperties<NewPodOp::Properties>();
  state.addTypes(result);
  state.addOperands(values);
  props.setInitializedRecords(builder.getStrArrayAttr(names));
}
} // namespace

void NewPodOp::build(
    OpBuilder &builder, OperationState &state, PodType result, ArrayRef<ValueRange> mapOperands,
    DenseI32ArrayAttr numDimsPerMap, InitializedRecords initialValues
) {
  buildCommon(builder, state, result, initialValues);
  affineMapHelpers::buildInstantiationAttrs<NewPodOp>(builder, state, mapOperands, numDimsPerMap);
}

void NewPodOp::build(
    OpBuilder &builder, OperationState &state, PodType result, InitializedRecords initialValues
) {
  buildCommon(builder, state, result, initialValues);
  assert(std::cmp_less_equal(initialValues.size(), std::numeric_limits<int32_t>::max()));
  affineMapHelpers::buildInstantiationAttrsEmpty<NewPodOp>(
      builder, state, static_cast<int32_t>(initialValues.size())
  );
}

void NewPodOp::getAsmResultNames(llvm::function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "pod");
}

namespace {

static void collectMapAttrs(Type type, SmallVector<AffineMapAttr> &mapAttrs) {
  llvm::TypeSwitch<Type, void>(type)
      .Case([&mapAttrs](PodType t) {
    for (auto record : t.getRecords()) {
      collectMapAttrs(record.getType(), mapAttrs);
    }
  })
      .Case([&mapAttrs](array::ArrayType t) {
    for (auto a : t.getDimensionSizes()) {
      if (auto m = llvm::dyn_cast<AffineMapAttr>(a)) {
        mapAttrs.push_back(m);
      }
    }
  })
      .Case([&mapAttrs](component::StructType t) {
    for (auto param : t.getParams()) {
      if (auto m = llvm::dyn_cast<AffineMapAttr>(param)) {
        mapAttrs.push_back(m);
      }
    }
  }).Default([](Type) {});
}

/// Verifies the initialization values.
///
/// Checks the following conditions:
///   - The number of names and the number of values is the same.
///   - Names are not repeated.
///   - Name set is a subset of the type's name set.
///   - Type of initial value matches type of the corresponding record.
static LogicalResult verifyInitialValues(
    ValueRange values, ArrayRef<Attribute> names, PodType retTy,
    llvm::function_ref<InFlightDiagnostic()> emitError
) {
  bool failed = false;
  if (names.size() != values.size()) {
    emitError() << "number of initialized records and initial values does not match ("
                << names.size() << " != " << values.size() << ")";
    failed = true;
  }

  llvm::StringMap<Type> records = retTy.getRecordMap();
  llvm::StringSet<> seenNames;
  for (auto [nameAttr, value] : llvm::zip_equal(names, values)) {
    auto name = llvm::cast<StringAttr>(nameAttr).getValue(); // Per the ODS spec.
    if (seenNames.contains(name)) {
      emitError() << "found duplicated record name '" << name << '\'';
      failed = true;
    }
    seenNames.insert(name);

    if (!records.contains(name)) {
      emitError() << "record '" << name << "' is not part of the struct";
      failed = true;
      continue;
    }

    auto valueTy = value.getType();
    auto recordTy = records.at(name);
    if (valueTy != recordTy) {
      auto err = emitError();
      err << "record '" << name << "' expected type " << recordTy << " but got " << valueTy;
      if (typesUnify(valueTy, recordTy)) {
        err.attachNote()
            << "types " << valueTy << " and " << recordTy
            << " can be unified. Perhaps you can add a 'poly.unifiable_cast' operation?";
      }
      failed = true;
    }
  }

  return failure(failed);
}

static LogicalResult verifyAffineMapOperands(NewPodOp *op, Type retTy) {
  SmallVector<AffineMapAttr> mapAttrs;
  collectMapAttrs(retTy, mapAttrs);
  return affineMapHelpers::verifyAffineMapInstantiations(
      op->getMapOperands(), op->getNumDimsPerMap(), mapAttrs, *op
  );
}

} // namespace

#define check(x)                                                                                   \
  {                                                                                                \
    failed = failed || mlir::failed(x);                                                            \
  }

LogicalResult NewPodOp::verify() {
  auto retTy = llvm::dyn_cast<PodType>(getResult().getType());
  assert(retTy); // per ODS spec of NewPodOp

  bool failed = false;
  check(
      verifyInitialValues(getInitialValues(), getInitializedRecords().getValue(), retTy, [this]() {
    return this->emitError();
  })
  );
  check(verifyAffineMapOperands(this, retTy));

  return failure(failed);
}

#undef check

using UnresolvedOp = OpAsmParser::UnresolvedOperand;

ParseResult
parseRecordInitialization(OpAsmParser &parser, StringAttr &name, UnresolvedOp &operand) {
  if (failed(parser.parseSymbolName(name))) {
    return failure();
  }

  if (parser.parseEqual()) {
    return failure();
  }
  return parser.parseOperand(operand);
}

ParseResult NewPodOp::parse(OpAsmParser &parser, OperationState &result) {
  /* Grammar
   * op : record_init map_operands `:` type($result) attr-dict
   * record_init : `{` record_inits `}`| `{` `}` | $
   * map_operands : custom<MapOperands> | $
   * record_inits : symbol `=` operand `,` record_inits | symbol `=` operand
   */

  auto &props = result.getOrAddProperties<NewPodOp::Properties>();

  SmallVector<Attribute> initializedRecords;
  // The map may not preserve the order of the operands so it needs to be iterated using
  // `initializedRecords` that preserves the original order.
  llvm::StringMap<UnresolvedOp> initialValuesOperands;
  auto parseElementFn = [&parser, &initializedRecords, &initialValuesOperands] {
    StringAttr name;
    UnresolvedOp operand;
    if (failed(parseRecordInitialization(parser, name, operand))) {
      return failure();
    }
    initializedRecords.push_back(name);
    initialValuesOperands.insert({name.getValue(), operand});
    return success();
  };
  auto initialValuesLoc = parser.getCurrentLocation();
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::OptionalBraces, parseElementFn)) {
    return failure();
  }
  SmallVector<int32_t> mapOperandsGroupSizes;
  SmallVector<UnresolvedOp> allMapOperands;
  Type indexTy = parser.getBuilder().getIndexType();
  bool colonAlreadyParsed = true;
  auto mapOperandsLoc = parser.getCurrentLocation();
  // Peek to see if we have affine map operands.
  // If we don't then the next token must be `:`
  if (failed(parser.parseOptionalColon())) {
    colonAlreadyParsed = false;
    SmallVector<SmallVector<UnresolvedOp>> mapOperands {};
    if (parseMultiDimAndSymbolList(parser, mapOperands, props.numDimsPerMap)) {
      return failure();
    }

    mapOperandsGroupSizes.reserve(mapOperands.size());
    for (const auto &subRange : mapOperands) {
      allMapOperands.append(subRange.begin(), subRange.end());
      assert(std::cmp_less_equal(subRange.size(), std::numeric_limits<int32_t>::max()));
      mapOperandsGroupSizes.push_back(static_cast<int32_t>(subRange.size()));
    }
  }

  if (!colonAlreadyParsed && parser.parseColon()) {
    return failure();
  }

  PodType resultType;
  if (parser.parseCustomTypeWithFallback(resultType)) {
    return failure();
  }
  // Now that we have the struct type we can resolve the operands
  // using the types of the struct.
  for (auto attr : initializedRecords) {
    auto name = llvm::cast<StringAttr>(attr); // Per ODS spec of RecordAttr
    auto lookup = resultType.getRecord(name.getValue(), [&parser, initialValuesLoc] {
      return parser.emitError(initialValuesLoc);
    });
    if (failed(lookup)) {
      return failure();
    }
    const auto &operand = initialValuesOperands.at(name.getValue());
    if (failed(parser.resolveOperands({operand}, *lookup, initialValuesLoc, result.operands))) {
      return failure();
    }
  }
  assert(std::cmp_less_equal(initializedRecords.size(), std::numeric_limits<int32_t>::max()));
  assert(std::cmp_less_equal(allMapOperands.size(), std::numeric_limits<int32_t>::max()));
  props.operandSegmentSizes = {
      static_cast<int32_t>(initializedRecords.size()), static_cast<int32_t>(allMapOperands.size())
  };
  props.mapOpGroupSizes = parser.getBuilder().getDenseI32ArrayAttr(mapOperandsGroupSizes);
  props.initializedRecords = parser.getBuilder().getArrayAttr(initializedRecords);
  result.addTypes({resultType});

  if (failed(parser.resolveOperands(allMapOperands, indexTy, mapOperandsLoc, result.operands))) {
    return failure();
  }
  {
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalAttrDict(result.attributes)) {
      return failure();
    }
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
      return parser.emitError(loc) << '\'' << result.name.getStringRef() << "' op ";
    }))) {
      return failure();
    }
  }

  return success();
}

void NewPodOp::print(OpAsmPrinter &printer) {
  auto &os = printer.getStream();
  auto initializedRecords = getInitializedRecordValues();
  if (!initializedRecords.empty()) {
    os << " { ";
    llvm::interleaveComma(initializedRecords, os, [&os, &printer](auto record) {
      printer.printSymbolName(record.name);
      os << " = ";
      printer.printOperand(record.value);
    });
    os << " } ";
  }
  printMultiDimAndSymbolList(printer, getOperation(), getMapOperands(), getNumDimsPerMapAttr());

  os << " : ";

  auto type = getResult().getType();
  if (auto validType = llvm::dyn_cast<PodType>(type)) {
    printer.printStrippedAttrOrType(validType);
  } else {
    printer.printType(type);
  }

  printer.printOptionalAttrDict(
      (*this)->getAttrs(),
      {"initializedRecords", "mapOpGroupSizes", "numDimsPerMap", "operandSegmentSizes"}
  );
}

SmallVector<RecordValue> NewPodOp::getInitializedRecordValues() {
  return llvm::map_to_vector(
      llvm::zip_equal(getInitialValues(), getInitializedRecords()), [](auto pair) {
    auto [value, name] = pair;
    return RecordValue {.name = llvm::cast<StringAttr>(name).getValue(), .value = value};
  }
  );
}

//===----------------------------------------------------------------------===//
// ReadPodOp
//===----------------------------------------------------------------------===//

LogicalResult ReadPodOp::verify() {
  auto podTy = llvm::dyn_cast<PodType>(getPodRef().getType());
  if (!podTy) {
    return emitError() << "reference operand expected a plain-old-data struct but got "
                       << getPodRef().getType();
  }

  auto lookup = podTy.getRecord(getRecordName(), [this]() { return this->emitError(); });
  if (failed(lookup)) {
    return lookup;
  }

  if (getResult().getType() != *lookup) {
    return emitError() << "operation result type and type of record do not match ("
                       << getResult().getType() << " != " << *lookup << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// WritePodOp
//===----------------------------------------------------------------------===//

LogicalResult WritePodOp::verify() {
  auto podTy = llvm::dyn_cast<PodType>(getPodRef().getType());
  if (!podTy) {
    return emitError() << "reference operand expected a plain-old-data struct but got "
                       << getPodRef().getType();
  }

  auto lookup = podTy.getRecord(getRecordName(), [this]() { return this->emitError(); });
  if (failed(lookup)) {
    return lookup;
  }

  if (getValue().getType() != *lookup) {
    return emitError() << "type of source value and type of record do not match ("
                       << getValue().getType() << " != " << *lookup << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Parsing/Printing helpers
//===----------------------------------------------------------------------===//

ParseResult parseRecordName(AsmParser &parser, FlatSymbolRefAttr &name) {
  return parser.parseCustomAttributeWithFallback(name);
}

void printRecordName(AsmPrinter &printer, Operation *, FlatSymbolRefAttr name) {
  printer.printSymbolName(name.getValue());
}

} // namespace llzk::pod
