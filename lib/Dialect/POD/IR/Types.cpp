//===-- Types.cpp - POD types implementations -------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/POD/IR/Types.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringSet.h>

using namespace mlir;

namespace llzk::pod {

//===----------------------------------------------------------------------===//
// PodType
//===----------------------------------------------------------------------===//

LogicalResult
PodType::verify(llvm::function_ref<InFlightDiagnostic()> emitError, ArrayRef<RecordAttr> records) {
  llvm::StringSet<> seenNames;
  bool failed = false;
  for (auto record : records) {
    auto recordName = record.getName();
    if (seenNames.contains(recordName)) {
      emitError() << "found duplicated record name '" << recordName.getValue() << '\'';
      failed = true;
    }
    seenNames.insert(recordName);
  }
  return mlir::failure(failed);
}

PodType PodType::fromInitialValues(MLIRContext *ctx, InitializedRecords init) {
  auto records = llvm::map_to_vector(init, [ctx](auto record) {
    return RecordAttr::get(ctx, StringAttr::get(ctx, record.name), record.value.getType());
  });
  return get(ctx, records);
}

FailureOr<Type>
PodType::getRecord(StringRef recordName, function_ref<InFlightDiagnostic()> emitError) const {
  for (RecordAttr record : getRecords()) {
    if (record.getName() == recordName) {
      return record.getType();
    }
  }
  return emitError() << "record '" << recordName << "' was not found in plain-old-data type";
}

llvm::StringMap<Type> PodType::getRecordMap() const {
  llvm::StringMap<Type> map;
  for (RecordAttr record : getRecords()) {
    map.insert({record.getName(), record.getType()});
  }
  return map;
}

ParseResult parsePodType(AsmParser &parser, SmallVector<RecordAttr> &records) {
  return parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&records, &parser]() {
    StringAttr name;
    Type type;
    auto result = parseRecord(parser, name, type);
    if (mlir::succeeded(result)) {
      records.push_back(RecordAttr::get(parser.getContext(), name, type));
    }
    return result;
  });
}

void printPodType(AsmPrinter &printer, ArrayRef<RecordAttr> records) {
  auto &os = printer.getStream();
  os << '[';
  printer.printStrippedAttrOrType(records);
  os << ']';
}

} // namespace llzk::pod
