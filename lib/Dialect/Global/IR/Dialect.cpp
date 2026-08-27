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
#include "llzk/Dialect/String/IR/Types.h"
#include "llzk/Util/TypeHelper.h"

// TableGen'd implementation files
#include "llzk/Dialect/Global/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace llzk;

//===------------------------------------------------------------------===//
// GlobalDialect
//===------------------------------------------------------------------===//

namespace {

/// Normalize initializer attributes emitted by pre-invariant bytecode.
///
/// Earlier versions serialized array elements as generic integer and string attributes.
/// They also allowed a field-qualified felt initializer and its unqualified
/// declaration (and vice versa) to disagree. Current GlobalDefOp verification
/// requires exact type equality, so apply the same field-adoption rules as the
/// textual parser while loading historical bytecode.
FailureOr<Attribute> normalizeLegacyInitializer(Type &type, Attribute value) {
  if (auto feltType = llvm::dyn_cast<felt::FeltType>(type)) {
    if (auto feltValue = llvm::dyn_cast<felt::FeltConstAttr>(value)) {
      auto valueType = feltValue.getType();
      if (!feltType.hasField() && valueType.hasField()) {
        type = valueType;
      } else if (feltType.hasField() && !valueType.hasField()) {
        value = felt::FeltConstAttr::get(value.getContext(), feltValue.getValue(), feltType);
      }
      return value;
    }
    if (auto intValue = llvm::dyn_cast<IntegerAttr>(value)) {
      return felt::FeltConstAttr::get(value.getContext(), intValue.getValue(), feltType);
    }
  } else if (auto intValue = llvm::dyn_cast<IntegerAttr>(value)) {
    APInt intValuePayload = intValue.getValue();
    if (type.isSignlessInteger(1)) {
      if (!intValuePayload.isZero() && !intValuePayload.isOne()) {
        return failure();
      }
      return IntegerAttr::get(type, intValuePayload.trunc(1));
    }
    if (llvm::isa<IndexType>(type)) {
      APInt v = intValue.getValue();
      if (v.getBitWidth() < IndexType::kInternalStorageBitWidth && v.isNegative()) {
        // IntegerAttr stores signless integer value, so a negative narrow literal and its
        // unsigned bit-pattern cannot be distinguished here. Reject the latter case rather
        // than zero-extending a negative value to a different index initializer.
        return failure();
      }
      return forceIntType(intValue, [&value]() {
        return InFlightDiagnosticWrapper::createSilent(value.getContext());
      });
    }
  } else if (auto stringType = llvm::dyn_cast<string::StringType>(type)) {
    if (auto stringValue = llvm::dyn_cast<StringAttr>(value)) {
      return StringAttr::get(stringValue.getValue(), stringType);
    }
  }

  if (auto arrayType = llvm::dyn_cast<array::ArrayType>(type)) {
    if (auto arrayValue = llvm::dyn_cast<ArrayAttr>(value)) {
      Type elementType = arrayType.getElementType();
      if (auto feltType = llvm::dyn_cast<felt::FeltType>(elementType)) {
        auto resolvedFeltType = feltType;
        for (Attribute element : arrayValue) {
          if (auto feltValue = llvm::dyn_cast<felt::FeltConstAttr>(element)) {
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
        FailureOr<Attribute> normalizedElement =
            normalizeLegacyInitializer(normalizedElementType, element);
        if (failed(normalizedElement)) {
          return failure();
        }
        elements.push_back(*normalizedElement);
      }
      return ArrayAttr::get(value.getContext(), elements);
    }
  }
  return value;
}

class GlobalDialectBytecodeInterface : public LLZKDialectBytecodeInterface<global::GlobalDialect> {
  using Base = LLZKDialectBytecodeInterface<global::GlobalDialect>;

public:
  using Base::Base;

  LogicalResult upgradeFromVersion(
      Operation *root, const LLZKDialectVersion & /*current*/,
      const LLZKDialectVersion & /*requested*/
  ) const final {
    auto res = root->walk([](global::GlobalDefOp global) -> WalkResult {
      if (Attribute initialValue = global.getInitialValueAttr()) {
        Type type = global.getType();
        FailureOr<Attribute> normalized = normalizeLegacyInitializer(type, initialValue);
        if (failed(normalized)) {
          return global.emitOpError(
              "contains a legacy initializer that is incompatible with its declared type"
          );
        }
        global.setInitialValueAttr(*normalized);
        global.setTypeAttr(TypeAttr::get(type));
      }
      return WalkResult::advance();
    });
    return failure(res.wasInterrupted());
  }
};

} // namespace

auto global::GlobalDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/Global/IR/Ops.cpp.inc"
  >();
  // clang-format on
  addInterfaces<GlobalDialectBytecodeInterface>();
}
