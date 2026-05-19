//===-- ValueModel.cpp - llzk-witgen runtime values -------------*- C++ -*-===//
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

#include <limits>
#include <random>

using namespace mlir;

namespace llzk::witgen {

namespace {

/// Draw a uniformly distributed field element in `[0, prime)`.
static llvm::DynamicAPInt randomFieldElement(std::mt19937_64 &rng, const Field &field) {
  const uint64_t prime = toAPSInt(field.prime()).getZExtValue();
  if (prime == 0) {
    return field.zero();
  }
  uint64_t candidate = std::uniform_int_distribution<uint64_t>(0, prime - 1)(rng);
  return field.reduce(candidate);
}

/// Draw a uniformly distributed signed index value.
static int64_t randomIndexValue(std::mt19937_64 &rng) {
  return std::uniform_int_distribution<int64_t>(
      std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()
  )(rng);
}

/// Draw a uniformly distributed boolean value.
static bool randomBoolValue(std::mt19937_64 &rng) {
  return std::uniform_int_distribution<int>(0, 1)(rng) != 0;
}

} // namespace

/// Require a boolean value from the runtime variant.
llvm::Expected<bool> asBool(const WitnessVal &value) {
  if (auto boolValue = std::get_if<bool>(&value)) {
    return *boolValue;
  }
  if (std::holds_alternative<std::monostate>(value)) {
    return makeError("read of uninitialized i1 value");
  }
  return makeError("expected i1 value");
}

/// Require an index value from the runtime variant.
llvm::Expected<int64_t> asIndex(const WitnessVal &value) {
  if (auto indexValue = std::get_if<int64_t>(&value)) {
    return *indexValue;
  }
  if (std::holds_alternative<std::monostate>(value)) {
    return makeError("read of uninitialized index value");
  }
  return makeError("expected index value");
}

/// Require a felt value from the runtime variant.
llvm::Expected<llvm::DynamicAPInt> asFelt(const WitnessVal &value) {
  if (auto feltValue = std::get_if<llvm::DynamicAPInt>(&value)) {
    return *feltValue;
  }
  if (std::holds_alternative<std::monostate>(value)) {
    return makeError("read of uninitialized felt value");
  }
  return makeError("expected felt value");
}

/// Require an array value from the runtime variant.
llvm::Expected<ArrayValueRef> asArray(const WitnessVal &value) {
  if (auto arrayValue = std::get_if<ArrayValueRef>(&value)) {
    return *arrayValue;
  }
  if (std::holds_alternative<std::monostate>(value)) {
    return makeError("read of uninitialized array value");
  }
  return makeError("expected array value");
}

/// Require a POD value from the runtime variant.
llvm::Expected<PodValueRef> asPod(const WitnessVal &value) {
  if (auto podValue = std::get_if<PodValueRef>(&value)) {
    return *podValue;
  }
  if (std::holds_alternative<std::monostate>(value)) {
    return makeError("read of uninitialized POD value");
  }
  return makeError("expected pod value");
}

/// Require a struct value from the runtime variant.
llvm::Expected<StructValueRef> asStruct(const WitnessVal &value) {
  if (auto structValue = std::get_if<StructValueRef>(&value)) {
    return *structValue;
  }
  if (std::holds_alternative<std::monostate>(value)) {
    return makeError("read of uninitialized struct value");
  }
  return makeError("expected struct value");
}

/// Build a default value for a supported LLZK type.
llvm::Expected<WitnessVal>
defaultValue(
    Type type, SymbolTableCollection &tables, Operation *origin, const Field &field,
    UninitializedBehavior behavior, std::mt19937_64 *rng
) {
  return llvm::TypeSwitch<Type, llvm::Expected<WitnessVal>>(type)
      .Case([&](felt::FeltType) -> llvm::Expected<WitnessVal> {
        if (behavior == UninitializedBehavior::Fail) {
          return WitnessVal(std::monostate {});
        }
        if (behavior == UninitializedBehavior::Random) {
          if (!rng) {
            return makeError("missing RNG for random witgen initialization");
          }
          return WitnessVal(randomFieldElement(*rng, field));
        }
        return field.zero();
      })
      .Case([&](IndexType) -> llvm::Expected<WitnessVal> {
        if (behavior == UninitializedBehavior::Fail) {
          return WitnessVal(std::monostate {});
        }
        if (behavior == UninitializedBehavior::Random) {
          if (!rng) {
            return makeError("missing RNG for random witgen initialization");
          }
          return int64_t(randomIndexValue(*rng));
        }
        return int64_t(0);
      })
      .Case([&](IntegerType intType) -> llvm::Expected<WitnessVal> {
    if (intType.getWidth() == 1) {
      if (behavior == UninitializedBehavior::Fail) {
        return WitnessVal(std::monostate {});
      }
      if (behavior == UninitializedBehavior::Random) {
        if (!rng) {
          return makeError("missing RNG for random witgen initialization");
        }
        return randomBoolValue(*rng);
      }
      return false;
    }
    return makeError("only i1 integer values are supported in llzk-witgen");
  })
      .Case([&](array::ArrayType arrayType) -> llvm::Expected<WitnessVal> {
    auto arrayValue = std::make_shared<ArrayValue>();
    arrayValue->type = arrayType;
    arrayValue->elements.reserve(arrayType.getNumElements());
    for (int64_t i = 0; i < arrayType.getNumElements(); ++i) {
      auto elem =
          defaultValue(arrayType.getElementType(), tables, origin, field, behavior, rng);
      if (!elem) {
        return elem.takeError();
      }
      arrayValue->elements.push_back(*elem);
    }
    return arrayValue;
  })
      .Case([&](pod::PodType podType) -> llvm::Expected<WitnessVal> {
    auto podValue = std::make_shared<PodValue>();
    podValue->type = podType;
    for (pod::RecordAttr record : podType.getRecords()) {
      auto recordValue =
          defaultValue(record.getType(), tables, origin, field, behavior, rng);
      if (!recordValue) {
        return recordValue.takeError();
      }
      podValue->records[record.getName().getValue()] = *recordValue;
    }
    return podValue;
  })
      .Case([&](component::StructType structType) -> llvm::Expected<WitnessVal> {
    auto defLookup = structType.getDefinition(tables, origin);
    if (failed(defLookup)) {
      return makeError("could not resolve struct type");
    }
    auto structValue = std::make_shared<StructValue>();
    structValue->type = structType;
    for (component::MemberDefOp member : defLookup->get().getMemberDefs()) {
      auto memberValue =
          defaultValue(member.getType(), tables, origin, field, behavior, rng);
      if (!memberValue) {
        return memberValue.takeError();
      }
      structValue->members[member.getSymName()] = *memberValue;
    }
    return structValue;
  }).Default([&](Type) -> llvm::Expected<WitnessVal> {
    return makeError("unsupported type in llzk-witgen");
  });
}

} // namespace llzk::witgen
