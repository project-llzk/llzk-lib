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

namespace {

class PODValueCopyDialectInterface final : public llzk::ValueCopyDialectInterface {
public:
  explicit PODValueCopyDialectInterface(mlir::Dialect *owner) : ValueCopyDialectInterface(owner) {}

  bool canMaterializeValueCopy(mlir::Type type) const final {
    auto podType = llvm::dyn_cast<llzk::pod::PodType>(type);
    if (!podType) {
      return false;
    }
    // Copy eligibility is type-based and must also hold for block arguments and read results.
    // Until copies can recover their instantiation groups, do not synthesize invalid pod.new ops.
    mlir::SmallVector<mlir::AffineMapAttr> maps;
    llzk::pod::collectPodMapAttrs(podType, maps);
    if (!maps.empty()) {
      return false;
    }
    return llvm::all_of(podType.getRecords(), [](llzk::pod::RecordAttr record) {
      return llzk::canMaterializeValueCopy(record.getType());
    });
  }

  std::string getValueCopyFailureReason(mlir::Type type) const final {
    auto podType = llvm::dyn_cast<llzk::pod::PodType>(type);
    if (!podType) {
      return ValueCopyDialectInterface::getValueCopyFailureReason(type);
    }
    for (llzk::pod::RecordAttr record : podType.getRecords()) {
      if (!llzk::canMaterializeValueCopy(record.getType())) {
        return "POD record '" + record.getName().getValue().str() +
               "': " + llzk::getValueCopyFailureReason(record.getType());
      }
    }
    return "POD copy requires affine-map instantiation operands that cannot be recovered";
  }

  mlir::FailureOr<mlir::Value> materializeValueCopy(
      mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source
  ) const final {
    auto podType = llvm::dyn_cast<llzk::pod::PodType>(source.getType());
    if (!podType || !canMaterializeValueCopy(podType)) {
      return mlir::failure();
    }

    auto destination = builder.create<llzk::pod::NewPodOp>(loc, podType);
    for (llzk::pod::RecordAttr record : podType.getRecords()) {
      auto read =
          builder.create<llzk::pod::ReadPodOp>(loc, record.getType(), source, record.getName());
      mlir::FailureOr<mlir::Value> copied =
          llzk::materializeValueCopy(builder, loc, read.getResult());
      if (mlir::failed(copied)) {
        return mlir::failure();
      }
      builder.create<llzk::pod::WritePodOp>(
          loc, destination.getResult(), record.getName(), *copied
      );
    }
    return destination.getResult();
  }
};

} // namespace

//===------------------------------------------------------------------===//
// PODDialect
//===------------------------------------------------------------------===//

auto llzk::pod::PODDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/POD/IR/Ops.cpp.inc"
  >();

  // Suppress false positive from `clang-tidy`
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addTypes<
    #define GET_TYPEDEF_LIST
    #include "llzk/Dialect/POD/IR/Types.cpp.inc"
  >();

  // Suppress false positive from `clang-tidy`
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "llzk/Dialect/POD/IR/Attrs.cpp.inc"
  >();

  // clang-format on
  addInterfaces<LLZKDialectBytecodeInterface<PODDialect>, PODValueCopyDialectInterface>();
}
