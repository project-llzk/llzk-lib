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

/// Normalize initializer attributes emitted by pre-invariant bytecode.
///
/// Earlier versions serialized array elements as generic integer attributes.
/// They also allowed a field-qualified felt initializer and its unqualified
/// declaration (and vice versa) to disagree. Current GlobalDefOp verification
/// requires exact type equality, so apply the same field-adoption rules as the
/// textual parser while loading historical bytecode.
Attribute normalizeLegacyInitializer(Type &type, Attribute value) {
  if (auto feltType = llvm::dyn_cast<llzk::felt::FeltType>(type)) {
    if (auto feltValue = llvm::dyn_cast<llzk::felt::FeltConstAttr>(value)) {
      auto valueType = feltValue.getType();
      if (!feltType.hasField() && valueType.hasField()) {
        type = valueType;
      } else if (feltType.hasField() && !valueType.hasField()) {
        value = llzk::felt::FeltConstAttr::get(value.getContext(), feltValue.getValue(), feltType);
      }
      return value;
    }
    if (auto intValue = llvm::dyn_cast<IntegerAttr>(value)) {
      return llzk::felt::FeltConstAttr::get(value.getContext(), intValue.getValue(), feltType);
    }
  } else if (auto intValue = llvm::dyn_cast<IntegerAttr>(value)) {
    if (type.isSignlessInteger(1) || llvm::isa<IndexType>(type)) {
      return IntegerAttr::get(type, intValue.getValue());
    }
  }

  if (auto arrayType = llvm::dyn_cast<llzk::array::ArrayType>(type)) {
    if (auto arrayValue = llvm::dyn_cast<ArrayAttr>(value)) {
      Type elementType = arrayType.getElementType();
      if (auto feltType = llvm::dyn_cast<llzk::felt::FeltType>(elementType)) {
        auto resolvedFeltType = feltType;
        for (Attribute element : arrayValue) {
          if (auto feltValue = llvm::dyn_cast<llzk::felt::FeltConstAttr>(element)) {
            auto valueType = feltValue.getType();
            if (valueType.hasField()) {
              // Conflicting explicit fields were invalid in textual IR before
              // this invariant. Leave them unchanged for verification to
              // reject rather than silently selecting one of the fields.
              if (resolvedFeltType.hasField() && resolvedFeltType != valueType) {
                return value;
              }
              resolvedFeltType = valueType;
            }
          }
        }
        elementType = resolvedFeltType;
        type = arrayType.cloneWith(elementType);
      }

      SmallVector<Attribute> elements;
      elements.reserve(arrayValue.size());
      for (Attribute element : arrayValue) {
        Type normalizedElementType = elementType;
        elements.push_back(normalizeLegacyInitializer(normalizedElementType, element));
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
        Type type = global.getType();
        global.setInitialValueAttr(normalizeLegacyInitializer(type, initialValue));
        global.setTypeAttr(TypeAttr::get(type));
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
