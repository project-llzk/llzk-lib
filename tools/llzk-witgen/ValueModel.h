//===-- ValueModel.h - llzk-witgen runtime values ---------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "llzk/Util/Field.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <variant>
#include <vector>

namespace mlir {
class Operation;
class SymbolTableCollection;
} // namespace mlir

namespace llzk::witgen {

struct ArrayValue;
struct PodValue;
struct StructValue;

/// Shared runtime storage for LLZK array values.
using ArrayValueRef = std::shared_ptr<ArrayValue>;

/// Shared runtime storage for LLZK POD values.
using PodValueRef = std::shared_ptr<PodValue>;

/// Shared runtime storage for LLZK struct values.
using StructValueRef = std::shared_ptr<StructValue>;

/// Runtime value representation used by the tool-local interpreter.
using WitnessVal = std::variant<
    std::monostate, bool, int64_t, llvm::DynamicAPInt, ArrayValueRef, PodValueRef, StructValueRef>;

/// Control how witgen materializes uninitialized/default values.
enum class UninitializedBehavior : std::uint8_t {
  Zero,
  Random,
  Fail,
};

/// Materialized array value with flattened element storage.
struct ArrayValue {
  array::ArrayType type;
  std::vector<WitnessVal> elements;
};

/// Materialized POD value keyed by record name.
struct PodValue {
  pod::PodType type;
  llvm::DenseMap<llvm::StringRef, WitnessVal> records;
};

/// Materialized struct value keyed by member name.
struct StructValue {
  component::StructType type;
  llvm::DenseMap<llvm::StringRef, WitnessVal> members;
};

/// Interpret a runtime value as a boolean.
llvm::Expected<bool> asBool(const WitnessVal &value);

/// Interpret a runtime value as an index-sized signed integer.
llvm::Expected<int64_t> asIndex(const WitnessVal &value);

/// Interpret a runtime value as a field element.
llvm::Expected<llvm::DynamicAPInt> asFelt(const WitnessVal &value);

/// Interpret a runtime value as an array reference.
llvm::Expected<ArrayValueRef> asArray(const WitnessVal &value);

/// Interpret a runtime value as a POD reference.
llvm::Expected<PodValueRef> asPod(const WitnessVal &value);

/// Interpret a runtime value as a struct reference.
llvm::Expected<StructValueRef> asStruct(const WitnessVal &value);

/// Build a default value used for `llzk.nondet` and aggregate constructors.
llvm::Expected<WitnessVal> defaultValue(
    mlir::Type type, mlir::SymbolTableCollection &tables, mlir::Operation *origin,
    const llzk::Field &field, UninitializedBehavior behavior, std::mt19937_64 *rng = nullptr
);

} // namespace llzk::witgen
