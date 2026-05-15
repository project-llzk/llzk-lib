//===-- JSON.cpp - llzk-witgen JSON conversion helpers ----------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "JSON.h"

#include "Errors.h"
#include "WitnessSelection.h"

#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/POD/IR/Attrs.h"

#include <mlir/IR/Operation.h>

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/raw_ostream.h>

using namespace mlir;

namespace llzk::witgen {

/// Parse an integer-compatible JSON value.
static llvm::Expected<int64_t> jsonToInt(const llvm::json::Value *json) {
  if (std::optional<int64_t> integer = json->getAsInteger()) {
    return *integer;
  }
  if (std::optional<llvm::StringRef> str = json->getAsString()) {
    int64_t value = 0;
    if (!str->getAsInteger(10, value)) {
      return value;
    }
  }
  return makeError("expected integer-compatible JSON value");
}

/// Parse a JSON field element encoded as an integer or decimal string.
static llvm::Expected<llvm::DynamicAPInt>
jsonToFelt(const llvm::json::Value *json, const Field &field) {
  if (std::optional<llvm::StringRef> str = json->getAsString()) {
    return field.reduce(toDynamicAPInt(*str));
  }
  if (std::optional<int64_t> integer = json->getAsInteger()) {
    return field.reduce(*integer);
  }
  return makeError("expected felt value as JSON integer or decimal string");
}

/// Parse a shaped LLZK array from nested JSON arrays.
static llvm::Expected<WitnessVal> parseJSONArray(
    const llvm::json::Value *json, array::ArrayType type, const Field &field, Operation *origin,
    size_t dimIndex = 0
) {
  auto *jsonArray = json->getAsArray();
  if (!jsonArray) {
    return makeError("expected JSON array");
  }

  llvm::ArrayRef<int64_t> shape = type.getShape();
  if (dimIndex >= shape.size()) {
    return makeError("invalid array shape");
  }
  if (jsonArray->size() != static_cast<size_t>(shape[dimIndex])) {
    return makeError("JSON array length does not match LLZK array dimension");
  }

  auto arrayValue = std::make_shared<ArrayValue>();
  arrayValue->type = type;
  if (dimIndex == shape.size() - 1) {
    arrayValue->elements.reserve(jsonArray->size());
    for (const llvm::json::Value &elem : *jsonArray) {
      auto parsed = parseJSONValue(&elem, type.getElementType(), field, origin);
      if (!parsed) {
        return parsed.takeError();
      }
      arrayValue->elements.push_back(*parsed);
    }
    return arrayValue;
  }

  auto subArrayType =
      array::ArrayType::get(type.getElementType(), type.getDimensionSizes().drop_front());
  arrayValue->elements.reserve(jsonArray->size());
  for (const llvm::json::Value &elem : *jsonArray) {
    auto parsed = parseJSONArray(&elem, subArrayType, field, origin, dimIndex + 1);
    if (!parsed) {
      return parsed.takeError();
    }
    auto subArray = asArray(*parsed);
    if (!subArray) {
      return subArray.takeError();
    }
    for (const WitnessVal &subElem : (*subArray)->elements) {
      arrayValue->elements.push_back(subElem);
    }
  }
  return arrayValue;
}

/// Render a field element as a decimal string for JSON output.
static llvm::Expected<llvm::json::Value> feltToJSON(const llvm::DynamicAPInt &value) {
  std::string rendered;
  llvm::raw_string_ostream os(rendered);
  os << value;
  return llvm::json::Value(os.str());
}

/// Serialize a flattened LLZK array into nested JSON arrays.
static llvm::Expected<llvm::json::Value> serializeJSONArray(
    const ArrayValueRef &arrayValue, array::ArrayType type, SymbolTableCollection &tables,
    Operation *origin, SerializationMode mode, size_t dimIndex = 0, size_t flatOffset = 0
) {
  llvm::json::Array jsonArray;
  llvm::ArrayRef<int64_t> shape = type.getShape();
  if (dimIndex == shape.size() - 1) {
    for (int64_t i = 0; i < shape[dimIndex]; ++i) {
      auto elem = serializeJSONValue(
          arrayValue->elements[flatOffset + static_cast<size_t>(i)], type.getElementType(), tables,
          origin, mode
      );
      if (!elem) {
        return elem.takeError();
      }
      jsonArray.push_back(*elem);
    }
    return llvm::json::Value(std::move(jsonArray));
  }

  size_t subArraySize = 1;
  for (size_t i = dimIndex + 1; i < shape.size(); ++i) {
    subArraySize *= static_cast<size_t>(shape[i]);
  }

  auto subArrayType =
      array::ArrayType::get(type.getElementType(), type.getDimensionSizes().drop_front());
  for (int64_t i = 0; i < shape[dimIndex]; ++i) {
    auto subArray = serializeJSONArray(
        arrayValue, subArrayType, tables, origin, mode, dimIndex + 1,
        flatOffset + static_cast<size_t>(i) * subArraySize
    );
    if (!subArray) {
      return subArray.takeError();
    }
    jsonArray.push_back(*subArray);
  }
  return llvm::json::Value(std::move(jsonArray));
}

/// Parse a supported LLZK input type from JSON.
llvm::Expected<WitnessVal>
parseJSONValue(const llvm::json::Value *json, Type type, const Field &field, Operation *origin) {
  return llvm::TypeSwitch<Type, llvm::Expected<WitnessVal>>(type)
      .Case([&](felt::FeltType) -> llvm::Expected<WitnessVal> { return jsonToFelt(json, field); })
      .Case([&](array::ArrayType arrayType) -> llvm::Expected<WitnessVal> {
    return parseJSONArray(json, arrayType, field, origin);
  })
      .Case([&](pod::PodType) -> llvm::Expected<WitnessVal> {
    return makeError("pod JSON inputs are not supported in llzk-witgen v1");
  })
      .Case([&](component::StructType) -> llvm::Expected<WitnessVal> {
    return makeError("struct JSON inputs are not supported in llzk-witgen v1");
  })
      .Case([&](IndexType) -> llvm::Expected<WitnessVal> {
    auto integer = jsonToInt(json);
    if (!integer) {
      return integer.takeError();
    }
    return *integer;
  })
      .Case([&](IntegerType intType) -> llvm::Expected<WitnessVal> {
    if (intType.getWidth() == 1) {
      if (std::optional<bool> boolValue = json->getAsBoolean()) {
        return *boolValue;
      }
      auto integer = jsonToInt(json);
      if (!integer) {
        return integer.takeError();
      }
      return *integer != 0;
    }
    return makeError("only i1 integer JSON inputs are supported");
  }).Default([&](Type) -> llvm::Expected<WitnessVal> {
    return makeError("unsupported input type in llzk-witgen");
  });
}

/// Serialize a supported LLZK runtime value into JSON.
llvm::Expected<llvm::json::Value> serializeJSONValue(
    const WitnessVal &value, Type type, SymbolTableCollection &tables, Operation *origin,
    SerializationMode mode
) {
  return llvm::TypeSwitch<Type, llvm::Expected<llvm::json::Value>>(type)
      .Case([&](felt::FeltType) -> llvm::Expected<llvm::json::Value> {
    auto feltValue = asFelt(value);
    if (!feltValue) {
      return feltValue.takeError();
    }
    return feltToJSON(*feltValue);
  })
      .Case([&](array::ArrayType arrayType) -> llvm::Expected<llvm::json::Value> {
    auto arrayValue = asArray(value);
    if (!arrayValue) {
      return arrayValue.takeError();
    }
    return serializeJSONArray(*arrayValue, arrayType, tables, origin, mode);
  })
      .Case([&](pod::PodType podType) -> llvm::Expected<llvm::json::Value> {
    auto podValue = asPod(value);
    if (!podValue) {
      return podValue.takeError();
    }
    llvm::json::Object result;
    for (pod::RecordAttr record : podType.getRecords()) {
      auto it = (*podValue)->records.find(record.getName().getValue());
      if (it == (*podValue)->records.end()) {
        return makeError("missing POD record during JSON serialization");
      }
      auto serialized = serializeJSONValue(it->second, record.getType(), tables, origin, mode);
      if (!serialized) {
        return serialized.takeError();
      }
      result[record.getName().getValue()] = *serialized;
    }
    return llvm::json::Value(std::move(result));
  })
      .Case([&](component::StructType structType) -> llvm::Expected<llvm::json::Value> {
    auto structValue = asStruct(value);
    if (!structValue) {
      return structValue.takeError();
    }
    auto defLookup = structType.getDefinition(tables, origin);
    if (failed(defLookup)) {
      return makeError("could not resolve struct type during JSON serialization");
    }
      llvm::json::Object result;
      for (component::MemberDefOp member : defLookup->get().getMemberDefs()) {
        auto it = (*structValue)->members.find(member.getSymName());
        if (it == (*structValue)->members.end()) {
          return makeError("missing struct member during JSON serialization");
        }

        if (mode == SerializationMode::PublicOutputsOnly) {
          if (!member.hasPublicAttr()) {
            continue;
          }
        } else {
          if (!memberIsSignal(defLookup->get(), member) &&
              !isa<component::StructType>(member.getType())) {
            continue;
          }
        }

        auto serialized = serializeJSONValue(it->second, member.getType(), tables, origin, mode);
        if (!serialized) {
          return serialized.takeError();
        }
        if (mode == SerializationMode::AllSignals && isa<component::StructType>(member.getType())) {
          auto *object = serialized->getAsObject();
          if (!object || object->empty()) {
            continue;
          }
        }
        result[member.getSymName()] = *serialized;
      }
      return llvm::json::Value(std::move(result));
  })
      .Case([&](IndexType) -> llvm::Expected<llvm::json::Value> {
    auto indexValue = asIndex(value);
    if (!indexValue) {
      return indexValue.takeError();
    }
    return llvm::json::Value(*indexValue);
  })
      .Case([&](IntegerType intType) -> llvm::Expected<llvm::json::Value> {
    if (intType.getWidth() != 1) {
      return makeError("only i1 integer JSON serialization is supported");
    }
    auto boolValue = asBool(value);
    if (!boolValue) {
      return boolValue.takeError();
    }
    return llvm::json::Value(*boolValue);
  }).Default([&](Type) -> llvm::Expected<llvm::json::Value> {
      return makeError("unsupported output type in llzk-witgen");
  });
}

/// Serialize named input values into a JSON object.
llvm::Expected<llvm::json::Object> buildInputsJSONObject(
    ArrayRef<InputBinding> bindings, ArrayRef<WitnessVal> values, SymbolTableCollection &tables,
    Operation *origin
) {
  if (bindings.size() != values.size()) {
    return makeError("input binding count mismatch during witness JSON assembly");
  }

  llvm::json::Object result;
  for (auto [binding, value] : llvm::zip(bindings, values)) {
    auto serialized =
        serializeJSONValue(value, binding.type, tables, origin, SerializationMode::AllSignals);
    if (!serialized) {
      return serialized.takeError();
    }
    result[binding.name] = *serialized;
  }
  return result;
}

/// Extract one nested runtime leaf by path.
llvm::Expected<WitnessVal> extractValueAtPath(
    const WitnessVal &root, Type rootType, ArrayRef<std::string> path, SymbolTableCollection &tables,
    Operation *origin
) {
  if (path.empty()) {
    return root;
  }

  if (auto structType = dyn_cast<component::StructType>(rootType)) {
    auto structValue = asStruct(root);
    if (!structValue) {
      return structValue.takeError();
    }
    auto defLookup = structType.getDefinition(tables, origin);
    if (failed(defLookup)) {
      return makeError("could not resolve struct type while extracting witness value");
    }
    for (component::MemberDefOp member : defLookup->get().getMemberDefs()) {
      if (member.getSymName() != path.front()) {
        continue;
      }
      auto it = (*structValue)->members.find(member.getSymName());
      if (it == (*structValue)->members.end()) {
        return makeError("missing struct member while extracting witness value");
      }
      return extractValueAtPath(it->second, member.getType(), path.drop_front(), tables, origin);
    }
    return makeError("unknown struct member while extracting witness value");
  }

  if (auto podType = dyn_cast<pod::PodType>(rootType)) {
    auto podValue = asPod(root);
    if (!podValue) {
      return podValue.takeError();
    }
    for (pod::RecordAttr record : podType.getRecords()) {
      if (record.getName().getValue() != path.front()) {
        continue;
      }
      auto it = (*podValue)->records.find(record.getName().getValue());
      if (it == (*podValue)->records.end()) {
        return makeError("missing POD record while extracting witness value");
      }
      return extractValueAtPath(it->second, record.getType(), path.drop_front(), tables, origin);
    }
    return makeError("unknown POD record while extracting witness value");
  }

  return makeError("extra witness path components for non-aggregate value");
}

} // namespace llzk::witgen
