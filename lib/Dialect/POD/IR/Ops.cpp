//===-- Ops.cpp - POD operation implementations -----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/POD/IR/Ops.h"

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
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
#include <optional>

// TableGen'd implementation files
#include "llzk/Dialect/POD/IR/OpInterfaces.cpp.inc"

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
  affineMapHelpers::buildInstantiationAttrs<NewPodOp>(
      builder, state, mapOperands, numDimsPerMap, llzk::checkedCast<int32_t>(initialValues.size())
  );
}

void NewPodOp::build(
    OpBuilder &builder, OperationState &state, PodType result, InitializedRecords initialValues
) {
  buildCommon(builder, state, result, initialValues);
  affineMapHelpers::buildInstantiationAttrsEmpty<NewPodOp>(
      builder, state, llzk::checkedCast<int32_t>(initialValues.size())
  );
}

void NewPodOp::getAsmResultNames(llvm::function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "pod");
}

/// Required by DestructurableAllocationOpInterface / SROA pass
SmallVector<DestructurableMemorySlot> NewPodOp::getDestructurableSlots() {
  PodType podType = getType();
  if (podType.getRecords().size() <= 1 || !getMapOperands().empty()) {
    return {};
  }
  if (auto destructured = podType.getSubelementIndexMap()) {
    return {DestructurableMemorySlot {{getResult(), podType}, std::move(*destructured)}};
  }
  return {};
}

/// Required by DestructurableAllocationOpInterface / SROA pass
DenseMap<Attribute, MemorySlot> NewPodOp::destructure(
    const DestructurableMemorySlot &slot, const SmallPtrSetImpl<Attribute> &usedIndices,
    OpBuilder &builder, SmallVectorImpl<DestructurableAllocationOpInterface> &newAllocators
) {
  assert(slot.ptr == getResult());
  assert(slot.elemType == getType());

  builder.setInsertionPointAfter(*this);

  SmallVector<RecordValue> initializedRecords = getInitializedRecordValues();
  DenseMap<Attribute, MemorySlot> slotMap;
  for (Attribute index : usedIndices) {
    auto recordName = llvm::dyn_cast<StringAttr>(index);
    assert(recordName && "expected StringAttr");

    Type destructAs = getType().getTypeAtIndex(recordName);
    assert(destructAs == slot.subelementTypes.lookup(recordName));

    auto destructAsPodTy = llvm::dyn_cast<PodType>(destructAs);
    assert(destructAsPodTy && "expected PodType");

    SmallVector<RecordValue, 1> initialValue;
    for (RecordValue record : initializedRecords) {
      if (record.name == recordName.getValue()) {
        initialValue.push_back(record);
        break;
      }
    }

    auto subNew = builder.create<NewPodOp>(getLoc(), destructAsPodTy, initialValue);
    newAllocators.push_back(subNew);
    slotMap.try_emplace<MemorySlot>(index, {subNew.getResult(), destructAs});
  }

  return slotMap;
}

/// Required by DestructurableAllocationOpInterface / SROA pass
std::optional<DestructurableAllocationOpInterface> NewPodOp::handleDestructuringComplete(
    const DestructurableMemorySlot &slot, OpBuilder & /*builder*/
) {
  assert(slot.ptr == getResult());
  this->erase();
  return std::nullopt;
}

/// Required by PromotableAllocationOpInterface / mem2reg pass
SmallVector<MemorySlot> NewPodOp::getPromotableSlots() {
  ArrayRef<RecordAttr> records = getType().getRecords();
  if (records.size() != 1) {
    return {};
  }
  return {MemorySlot {getResult(), records.front().getType()}};
}

/// Required by PromotableAllocationOpInterface / mem2reg pass
Value NewPodOp::getDefaultValue(const MemorySlot &slot, OpBuilder &builder) {
  assert(slot.ptr == getResult());
  ArrayRef<RecordAttr> records = getType().getRecords();
  assert(records.size() == 1 && "only single-record pods are promotable");
  assert(records.front().getType() == slot.elemType);

  StringRef recordName = records.front().getName().getValue();
  for (RecordValue record : getInitializedRecordValues()) {
    if (record.name == recordName) {
      return record.value;
    }
  }
  return builder.create<llzk::NonDetOp>(getLoc(), slot.elemType);
}

/// Required by PromotableAllocationOpInterface / mem2reg pass
void NewPodOp::handleBlockArgument(const MemorySlot &, BlockArgument, OpBuilder &) {}

/// Required by PromotableAllocationOpInterface / mem2reg pass
std::optional<PromotableAllocationOpInterface> NewPodOp::handlePromotionComplete(
    const MemorySlot &slot, Value defaultValue, OpBuilder & /*builder*/
) {
  assert(slot.ptr == getResult());
  if (defaultValue && defaultValue.use_empty()) {
    if (Operation *defOp = defaultValue.getDefiningOp()) {
      if (llvm::isa<llzk::NonDetOp>(defOp)) {
        defOp->erase();
      }
    }
  }
  this->erase();
  return std::nullopt;
}

namespace {

static void collectMapAttrs(Type type, SmallVector<AffineMapAttr> &mapAttrs) {
  // clang-format off
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
      if (ArrayAttr params = t.getParams()) {
        for (auto param : params) {
          if (auto m = llvm::dyn_cast<AffineMapAttr>(param)) {
            mapAttrs.push_back(m);
          }
        }
      }
    }).Default([](Type) {});
  // clang-format on
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
      mapOperandsGroupSizes.push_back(llzk::checkedCast<int32_t>(subRange.size()));
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
  props.operandSegmentSizes = {
      llzk::checkedCast<int32_t>(initializedRecords.size()),
      llzk::checkedCast<int32_t>(allMapOperands.size())
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

SmallVector<RecordValue>
getInitializedRecordValues(ValueRange initialValues, ArrayAttr initializedRecords) {
  return llvm::map_to_vector(llvm::zip_equal(initialValues, initializedRecords), [](auto pair) {
    auto [value, name] = pair;
    return RecordValue {.name = llvm::cast<StringAttr>(name).getValue(), .value = value};
  });
}

SmallVector<RecordValue> NewPodOp::getInitializedRecordValues() {
  return llzk::pod::getInitializedRecordValues(getInitialValues(), getInitializedRecords());
}

//===----------------------------------------------------------------------===//
// PodAccessOpInterface
//===----------------------------------------------------------------------===//

/// Required by DestructurableAllocationOpInterface / SROA pass
bool PodAccessOpInterface::canRewire(
    const DestructurableMemorySlot &slot, SmallPtrSetImpl<Attribute> &usedIndices,
    SmallVectorImpl<MemorySlot> & /*mustBeSafelyUsed*/, const DataLayout & /*dataLayout*/
) {
  if (slot.ptr != getPodRef()) {
    return false;
  }

  StringAttr recordName = getRecordNameAttr();
  if (!slot.subelementTypes.contains(recordName)) {
    return false;
  }

  usedIndices.insert(recordName);
  return true;
}

/// Required by DestructurableAllocationOpInterface / SROA pass
DeletionKind PodAccessOpInterface::rewire(
    const DestructurableMemorySlot &slot, DenseMap<Attribute, MemorySlot> &subslots,
    OpBuilder & /*builder*/, const DataLayout & /*dataLayout*/
) {
  assert(slot.ptr == getPodRef());
  assert(slot.elemType == getPodRefType());

  StringAttr recordName = getRecordNameAttr();
  const MemorySlot &memorySlot = subslots.at(recordName);
  getPodRefMutable().set(memorySlot.ptr);

  return DeletionKind::Keep;
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

ParseResult parseRecordName(AsmParser &parser, StringAttr &name) {
  FlatSymbolRefAttr symRef;
  auto result = parser.parseCustomAttributeWithFallback(symRef);
  if (succeeded(result)) {
    name = symRef.getAttr();
  }
  return result;
}

void printRecordName(AsmPrinter &printer, Operation *, StringAttr name) {
  printer.printSymbolName(name.getValue());
}

} // namespace llzk::pod
