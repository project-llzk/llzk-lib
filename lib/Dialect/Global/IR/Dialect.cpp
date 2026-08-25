//===-- Dialect.cpp - Global value dialect implementation -------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Global/IR/Dialect.h"

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Versioning.h"

// TableGen'd implementation files
#include "llzk/Dialect/Global/IR/Dialect.cpp.inc"

using namespace mlir;

//===------------------------------------------------------------------===//
// GlobalDialect
//===------------------------------------------------------------------===//

namespace {

/// Normalize integer initializer attributes emitted by pre-invariant bytecode.
///
/// Earlier versions serialized array elements as generic integer attributes.
/// Current GlobalDefOp verification requires their types to agree with the
/// declared global type, so upgrade them while loading historical bytecode.
Attribute normalizeLegacyInitializer(Type type, Attribute value) {
  if (auto intValue = llvm::dyn_cast<IntegerAttr>(value)) {
    if (type.isSignlessInteger(1) || llvm::isa<IndexType>(type)) {
      return IntegerAttr::get(type, intValue.getValue());
    }
    if (auto feltType = llvm::dyn_cast<llzk::felt::FeltType>(type)) {
      return llzk::felt::FeltConstAttr::get(value.getContext(), intValue.getValue(), feltType);
    }
  }
  if (auto arrayType = llvm::dyn_cast<llzk::array::ArrayType>(type)) {
    if (auto arrayValue = llvm::dyn_cast<ArrayAttr>(value)) {
      SmallVector<Attribute> elements;
      elements.reserve(arrayValue.size());
      for (Attribute element : arrayValue) {
        elements.push_back(normalizeLegacyInitializer(arrayType.getElementType(), element));
      }
      return ArrayAttr::get(value.getContext(), elements);
    }
  }
  return value;
}

class GlobalDialectBytecodeInterface
    : public llzk::LLZKDialectBytecodeInterface<llzk::global::GlobalDialect> {
  using Base = llzk::LLZKDialectBytecodeInterface<llzk::global::GlobalDialect>;

public:
  using Base::Base;

  LogicalResult upgradeFromVersion(
      Operation *root, const llzk::LLZKDialectVersion & /*current*/,
      const llzk::LLZKDialectVersion & /*requested*/
  ) const final {
    root->walk([](llzk::global::GlobalDefOp global) {
      if (Attribute initialValue = global.getInitialValueAttr()) {
        global.setInitialValueAttr(normalizeLegacyInitializer(global.getType(), initialValue));
      }
    });
    return success();
  }
};

} // namespace

auto llzk::global::GlobalDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/Global/IR/Ops.cpp.inc"
  >();
  // clang-format on
  addInterfaces<GlobalDialectBytecodeInterface>();
}
