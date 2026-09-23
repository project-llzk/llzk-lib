//===-- Dialect.cpp - Array dialect implementation --------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Dialect.h"

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/LLZK/IR/Versioning.h"
#include "llzk/Dialect/Shared/ValueCopy.h"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

// TableGen'd implementation files
#include "llzk/Dialect/Array/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/Array/IR/Types.cpp.inc"

namespace {

class ArrayValueCopyDialectInterface final : public llzk::ValueCopyDialectInterface {
public:
  explicit ArrayValueCopyDialectInterface(mlir::Dialect *owner)
      : ValueCopyDialectInterface(owner) {}

  bool canMaterializeValueCopy(mlir::Type type) const final {
    auto arrayType = llvm::dyn_cast<llzk::array::ArrayType>(type);
    return arrayType && arrayType.hasStaticShape() &&
           llzk::canMaterializeValueCopy(arrayType.getElementType());
  }

  mlir::FailureOr<mlir::Value> materializeValueCopy(
      mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source
  ) const final {
    auto arrayType = llvm::dyn_cast<llzk::array::ArrayType>(source.getType());
    if (!arrayType || !canMaterializeValueCopy(arrayType)) {
      return mlir::failure();
    }

    auto destination = builder.create<llzk::array::CreateArrayOp>(loc, arrayType);
    std::optional<mlir::SmallVector<mlir::ArrayAttr>> indices = arrayType.getSubelementIndices();
    assert(indices.has_value() && "static arrays must provide concrete element indices");
    for (mlir::ArrayAttr index : *indices) {
      mlir::Value read = llzk::array::ArrayAccessOpInterface::genRead(builder, loc, source, index);
      mlir::FailureOr<mlir::Value> copied = llzk::materializeValueCopy(builder, loc, read);
      if (mlir::failed(copied)) {
        return mlir::failure();
      }
      llzk::array::ArrayAccessOpInterface::genWrite(
          builder, loc, destination.getResult(), index, *copied
      );
    }
    return destination.getResult();
  }
};

} // namespace

//===------------------------------------------------------------------===//
// ArrayDialect
//===------------------------------------------------------------------===//

auto llzk::array::ArrayDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/Array/IR/Ops.cpp.inc"
  >();

  // Suppress false positive from `clang-tidy`
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addTypes<
    #define GET_TYPEDEF_LIST
    #include "llzk/Dialect/Array/IR/Types.cpp.inc"
  >();
  // clang-format on
  addInterfaces<LLZKDialectBytecodeInterface<ArrayDialect>, ArrayValueCopyDialectInterface>();
}
