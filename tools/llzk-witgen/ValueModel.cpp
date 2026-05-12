//===-- ValueModel.cpp - llzk-witgen runtime values ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "ValueModel.h"

#include "Errors.h"

#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/POD/IR/Attrs.h"

#include <mlir/IR/Operation.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace mlir;

namespace llzk::witgen {

/// Require a boolean value from the runtime variant.
llvm::Expected<bool> asBool(const Value &value) {
  if (auto boolValue = std::get_if<bool>(&value)) {
    return *boolValue;
  }
  return makeError("expected i1 value");
}

/// Require an index value from the runtime variant.
llvm::Expected<int64_t> asIndex(const Value &value) {
  if (auto indexValue = std::get_if<int64_t>(&value)) {
    return *indexValue;
  }
  return makeError("expected index value");
}

/// Require a felt value from the runtime variant.
llvm::Expected<llvm::DynamicAPInt> asFelt(const Value &value) {
  if (auto feltValue = std::get_if<llvm::DynamicAPInt>(&value)) {
    return *feltValue;
  }
  return makeError("expected felt value");
}

/// Require an array value from the runtime variant.
llvm::Expected<ArrayValueRef> asArray(const Value &value) {
  if (auto arrayValue = std::get_if<ArrayValueRef>(&value)) {
    return *arrayValue;
  }
  return makeError("expected array value");
}

/// Require a POD value from the runtime variant.
llvm::Expected<PodValueRef> asPod(const Value &value) {
  if (auto podValue = std::get_if<PodValueRef>(&value)) {
    return *podValue;
  }
  return makeError("expected pod value");
}

/// Require a struct value from the runtime variant.
llvm::Expected<StructValueRef> asStruct(const Value &value) {
  if (auto structValue = std::get_if<StructValueRef>(&value)) {
    return *structValue;
  }
  return makeError("expected struct value");
}

/// Build a deterministic default value for a supported LLZK type.
llvm::Expected<Value>
defaultValue(Type type, SymbolTableCollection &tables, Operation *origin, const Field &field) {
  return llvm::TypeSwitch<Type, llvm::Expected<Value>>(type)
      .Case([&](felt::FeltType) -> llvm::Expected<Value> { return field.zero(); })
      .Case([&](IndexType) -> llvm::Expected<Value> { return int64_t(0); })
      .Case([&](IntegerType intType) -> llvm::Expected<Value> {
    if (intType.getWidth() == 1) {
      return false;
    }
    return makeError("only i1 integer values are supported in llzk-witgen");
  })
      .Case([&](array::ArrayType arrayType) -> llvm::Expected<Value> {
    auto arrayValue = std::make_shared<ArrayValue>();
    arrayValue->type = arrayType;
    arrayValue->elements.reserve(arrayType.getNumElements());
    for (int64_t i = 0; i < arrayType.getNumElements(); ++i) {
      auto elem = defaultValue(arrayType.getElementType(), tables, origin, field);
      if (!elem) {
        return elem.takeError();
      }
      arrayValue->elements.push_back(*elem);
    }
    return arrayValue;
  })
      .Case([&](pod::PodType podType) -> llvm::Expected<Value> {
    auto podValue = std::make_shared<PodValue>();
    podValue->type = podType;
    for (pod::RecordAttr record : podType.getRecords()) {
      auto recordValue = defaultValue(record.getType(), tables, origin, field);
      if (!recordValue) {
        return recordValue.takeError();
      }
      podValue->records[record.getName().getValue()] = *recordValue;
    }
    return podValue;
  })
      .Case([&](component::StructType structType) -> llvm::Expected<Value> {
    auto defLookup = structType.getDefinition(tables, origin);
    if (failed(defLookup)) {
      return makeError("could not resolve struct type");
    }
    auto structValue = std::make_shared<StructValue>();
    structValue->type = structType;
    for (component::MemberDefOp member : defLookup->get().getMemberDefs()) {
      auto memberValue = defaultValue(member.getType(), tables, origin, field);
      if (!memberValue) {
        return memberValue.takeError();
      }
      structValue->members[member.getSymName()] = *memberValue;
    }
    return structValue;
  }).Default([&](Type) -> llvm::Expected<Value> {
    return makeError("unsupported type in llzk-witgen");
  });
}

} // namespace llzk::witgen
