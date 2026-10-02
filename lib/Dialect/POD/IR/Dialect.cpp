//===-- Dialect.cpp - POD dialect implementation ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/POD/IR/Dialect.h"

#include "llzk/Dialect/LLZK/IR/Versioning.h"
#include "llzk/Dialect/POD/IR/Attrs.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Shared/ValueCopy.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

// TableGen'd implementation files
#include "llzk/Dialect/POD/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/POD/IR/Types.cpp.inc"

// Need a complete declaration of storage classes for below
#define GET_ATTRDEF_CLASSES
#include "llzk/Dialect/POD/IR/Attrs.cpp.inc"

using namespace mlir;
using namespace llzk;
using namespace llzk::pod;

namespace {

class PODValueCopyDialectInterface final : public ValueCopyDialectInterface {
public:
  explicit PODValueCopyDialectInterface(Dialect *owner) : ValueCopyDialectInterface(owner) {}

  bool canMaterializeValueCopy(Type type) const final {
    auto podType = llvm::dyn_cast<PodType>(type);
    if (!podType) {
      return false;
    }
    // Copy eligibility is type-based and must also hold for block arguments and read results.
    // Until copies can recover their instantiation groups, do not synthesize invalid pod.new ops.
    SmallVector<AffineMapAttr> maps;
    collectPodMapAttrs(podType, maps);
    if (!maps.empty()) {
      return false;
    }
    return llvm::all_of(podType.getRecords(), [](RecordAttr record) {
      return llzk::canMaterializeValueCopy(record.getType());
    });
  }

  std::string getValueCopyFailureReason(Type type) const final {
    auto podType = llvm::dyn_cast<PodType>(type);
    if (!podType) {
      return ValueCopyDialectInterface::getValueCopyFailureReason(type);
    }
    for (RecordAttr record : podType.getRecords()) {
      if (!llzk::canMaterializeValueCopy(record.getType())) {
        return "POD record '" + record.getName().getValue().str() +
               "': " + llzk::getValueCopyFailureReason(record.getType());
      }
    }
    return "POD copy requires affine-map instantiation operands that cannot be recovered";
  }

  FailureOr<Value>
  materializeValueCopy(OpBuilder &builder, Location loc, Value source) const final {
    auto podType = llvm::dyn_cast<PodType>(source.getType());
    if (!podType || !canMaterializeValueCopy(podType)) {
      return failure();
    }

    auto destination = NewPodOp::create(builder, loc, podType);
    for (RecordAttr record : podType.getRecords()) {
      auto read = ReadPodOp::create(builder, loc, record.getType(), source, record.getName());
      FailureOr<Value> copied = llzk::materializeValueCopy(builder, loc, read.getResult());
      if (failed(copied)) {
        return failure();
      }
      WritePodOp::create(builder, loc, destination.getResult(), record.getName(), *copied);
    }
    return destination.getResult();
  }
};

} // namespace

//===------------------------------------------------------------------===//
// PODDialect
//===------------------------------------------------------------------===//

auto PODDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/POD/IR/Ops.cpp.inc"
  >();

  addTypes<
    #define GET_TYPEDEF_LIST
    #include "llzk/Dialect/POD/IR/Types.cpp.inc"
  >();

  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "llzk/Dialect/POD/IR/Attrs.cpp.inc"
  >();

  // clang-format on
  addInterfaces<LLZKDialectBytecodeInterface<PODDialect>, PODValueCopyDialectInterface>();
}
