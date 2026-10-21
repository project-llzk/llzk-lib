//===-- DiscoveryStorage.h --------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/Builders.h>

#include <llvm/ADT/DenseMap.h>

#include <memory>

namespace llzk::polymorphic::detail {

/// Sparse known scalar leaves of a local aggregate, indexed by record/array paths.
using DiscoveryLeaves = llvm::DenseMap<mlir::ArrayAttr, mlir::Attribute>;

/// A scalar constant or a local aggregate identity. Missing leaves are unknown.
struct DiscoveryValue {
  mlir::Attribute constant;
  std::shared_ptr<DiscoveryLeaves> aggregate;

  DiscoveryValue() = default;
  DiscoveryValue(mlir::Attribute value) : constant(value) {}

  bool operator==(const DiscoveryValue &other) const {
    return constant == other.constant && aggregate == other.aggregate;
  }
};

/// Track only local constant aggregate contents. Reads and assignments copy
/// values; loop arguments retain identity. Unknown control flow or escaping
/// storage invalidates facts instead of guessing memory effects.
class DiscoveryStorage {
  mlir::Builder builder;
  llvm::SmallVector<std::weak_ptr<DiscoveryLeaves>> allocations;

  static bool startsWith(mlir::ArrayAttr path, llvm::ArrayRef<mlir::Attribute> prefix) {
    return path.size() >= prefix.size() &&
           llvm::equal(path.getValue().take_front(prefix.size()), prefix);
  }

public:
  explicit DiscoveryStorage(mlir::MLIRContext *context) : builder(context) {}

  /// Allocate an aggregate with no known leaves.
  DiscoveryValue create() {
    if (allocations.size() >= 4096 && allocations.size() % 4096 == 0) {
      llvm::erase_if(allocations, [](const auto &allocation) { return allocation.expired(); });
    }
    DiscoveryValue value;
    value.aggregate = std::make_shared<DiscoveryLeaves>();
    allocations.push_back(value.aggregate);
    return value;
  }

  /// Drop memory facts at an unknown control-flow or call boundary. Scalar SSA
  /// constants already read from storage remain immutable and need no invalidation.
  void invalidate() {
    llvm::erase_if(allocations, [](auto &allocation) {
      if (auto leaves = allocation.lock()) {
        leaves->clear();
        return false;
      }
      return true;
    });
  }

  /// Forget the contents of a potentially modified or escaping local aggregate.
  void forget(const DiscoveryValue &value) {
    if (value.aggregate) {
      value.aggregate->clear();
    }
  }

  /// Read a scalar leaf or snapshot an aggregate subtree into an independent root.
  DiscoveryValue
  read(const DiscoveryValue &source, llvm::ArrayRef<mlir::Attribute> path, bool aggregate) {
    if (!source.aggregate) {
      return {};
    }
    if (!aggregate) {
      return source.aggregate->lookup(builder.getArrayAttr(path));
    }
    auto result = create();
    for (auto [key, value] : *source.aggregate) {
      if (startsWith(key, path)) {
        (*result.aggregate)[builder.getArrayAttr(key.getValue().drop_front(path.size()))] = value;
      }
    }
    return result;
  }

  /// Replace a subtree, clearing old facts even when the new value is unknown.
  void write(
      const DiscoveryValue &destination, llvm::ArrayRef<mlir::Attribute> path,
      const DiscoveryValue &value
  ) {
    if (!destination.aggregate) {
      return;
    }
    DiscoveryLeaves snapshot = value.aggregate ? *value.aggregate : DiscoveryLeaves();
    destination.aggregate->remove_if([path](const auto &entry) {
      return startsWith(entry.first, path);
    });
    if (value.constant) {
      (*destination.aggregate)[builder.getArrayAttr(path)] = value.constant;
    }
    for (auto [key, constant] : snapshot) {
      llvm::SmallVector<mlir::Attribute> target(path);
      llvm::append_range(target, key.getValue());
      (*destination.aggregate)[builder.getArrayAttr(target)] = constant;
    }
  }
};

} // namespace llzk::polymorphic::detail
