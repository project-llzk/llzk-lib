//===-- ConstraintScalarFacts.h ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Util/DynamicAPIntHelper.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Matchers.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>

namespace llzk::polymorphic::detail {
/// Evaluate only scalar dependencies demanded by control flow and concrete paths.
class ConstraintScalarFacts {
  mlir::Builder builder;
  llvm::DenseMap<mlir::Value, mlir::Attribute> constants;

  /// Use fold hooks on a disposable operation; emitted scalar expressions stay SSA.
  void fold(mlir::Operation *producer, mlir::Value current) {
    llvm::SmallVector<mlir::Attribute> operands;
    for (mlir::Value operand : producer->getOperands()) {
      operands.push_back(constants.lookup(operand));
    }
    // Casts expose canonicalization rather than fold hooks. Match the same
    // nonnegative, signed-64 representability rule for structural indices.
    if (auto conversion = llvm::dyn_cast<llzk::cast::FeltToIndexOp>(producer)) {
      if (auto constant = llvm::dyn_cast_or_null<felt::FeltConstAttr>(operands.front())) {
        auto reduced = constant.getReducedValue();
        if (!reduced) {
          return;
        }
        auto converted = checkedToInt64(*reduced);
        if (converted && *converted >= 0) {
          constants[current] = builder.getIndexAttr(*converted);
        } else if (!converted) {
          llvm::consumeError(converted.takeError());
        }
      }
      return;
    }
    mlir::Operation *temporary = producer->clone();
    llvm::SmallVector<mlir::OpFoldResult> results;
    if (mlir::succeeded(temporary->fold(operands, results)) &&
        results.size() == producer->getNumResults()) {
      for (auto [result, folded] : llvm::zip_equal(producer->getResults(), results)) {
        constants[result] = llvm::isa<mlir::Attribute>(folded)
                                ? llvm::cast<mlir::Attribute>(folded)
                                : constants.lookup(llvm::cast<mlir::Value>(folded));
      }
    }
    temporary->destroy();
  }

public:
  explicit ConstraintScalarFacts(mlir::MLIRContext *context) : builder(context) {}

  /// Evaluate only the pure SSA dependencies of a requested structural value.
  /// An explicit worklist also handles long iteration-dependent expression chains.
  mlir::Attribute known(mlir::Value value) {
    llvm::SmallVector<std::pair<mlir::Value, bool>> pending {{value, false}};
    llvm::DenseSet<mlir::Value> examined;
    while (!pending.empty()) {
      auto [current, ready] = pending.pop_back_val();
      if (constants.lookup(current)) {
        continue;
      }
      mlir::Attribute literal;
      if (mlir::matchPattern(current, mlir::m_Constant(&literal))) {
        constants[current] = literal;
        continue;
      }
      mlir::Operation *producer = current.getDefiningOp();
      if (!producer || producer->getNumRegions() || !mlir::isMemoryEffectFree(producer)) {
        continue;
      }
      if (!ready) {
        if (!examined.insert(current).second) {
          continue;
        }
        pending.push_back({current, true});
        for (mlir::Value operand : producer->getOperands()) {
          pending.push_back({operand, false});
        }
        continue;
      }
      fold(producer, current);
    }
    return constants.lookup(value);
  }

  /// Require an integer representable by concrete access paths and signed loops.
  std::optional<int64_t> integer(mlir::Value value) {
    auto attr = llvm::dyn_cast_or_null<mlir::IntegerAttr>(known(value));
    if (!attr || !attr.getValue().isSignedIntN(64)) {
      return std::nullopt;
    }
    return attr.getInt();
  }
};

} // namespace llzk::polymorphic::detail
