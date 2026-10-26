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

/// Scalar contents and concrete array shapes, including nested aggregate roots.
struct DiscoveryAggregate {
  DiscoveryLeaves leaves;
  llvm::DenseMap<mlir::ArrayAttr, llvm::SmallVector<int64_t>> shapes;

  /// Writes may replace nested arrays, but cannot resize the root allocation.
  void invalidate() {
    leaves.clear();
    shapes.remove_if([](const auto &entry) { return !entry.first.empty(); });
  }
};

/// A scalar constant or a local aggregate identity. Missing leaves are unknown.
struct DiscoveryValue {
  mlir::Attribute constant;
  std::shared_ptr<DiscoveryAggregate> aggregate;

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
  /// Periodically reclaim expired roots without scanning on every allocation.
  static constexpr size_t allocationCleanupInterval = 4096;

  mlir::Builder builder;
  llvm::SmallVector<std::weak_ptr<DiscoveryAggregate>> allocations;

  static bool startsWith(mlir::ArrayAttr path, llvm::ArrayRef<mlir::Attribute> prefix) {
    return path.size() >= prefix.size() &&
           llvm::equal(path.getValue().take_front(prefix.size()), prefix);
  }

public:
  explicit DiscoveryStorage(mlir::MLIRContext *context) : builder(context) {}

  /// Allocate an aggregate with no known leaves.
  DiscoveryValue create() {
    if (allocations.size() >= allocationCleanupInterval &&
        allocations.size() % allocationCleanupInterval == 0) {
      llvm::erase_if(allocations, [](const auto &allocation) { return allocation.expired(); });
    }
    DiscoveryValue value;
    value.aggregate = std::make_shared<DiscoveryAggregate>();
    allocations.push_back(value.aggregate);
    return value;
  }

  /// Drop memory facts at an unknown control-flow or call boundary. Scalar SSA
  /// constants already read from storage remain immutable and need no invalidation.
  void invalidate() {
    llvm::erase_if(allocations, [](auto &allocation) {
      if (auto leaves = allocation.lock()) {
        leaves->invalidate();
        return false;
      }
      return true;
    });
  }

  /// Forget the contents of a potentially modified or escaping local aggregate.
  void forget(const DiscoveryValue &value) {
    if (value.aggregate) {
      value.aggregate->invalidate();
    } else {
      // An untracked reference can alias any local allocation.
      invalidate();
    }
  }

  /// Read a scalar leaf or snapshot an aggregate subtree into an independent root.
  DiscoveryValue
  read(const DiscoveryValue &source, llvm::ArrayRef<mlir::Attribute> path, bool aggregate) {
    if (!source.aggregate) {
      return {};
    }
    if (!aggregate) {
      return source.aggregate->leaves.lookup(builder.getArrayAttr(path));
    }
    auto result = create();
    for (auto [key, value] : source.aggregate->leaves) {
      if (startsWith(key, path)) {
        result.aggregate->leaves[builder.getArrayAttr(key.getValue().drop_front(path.size()))] =
            value;
      }
    }
    for (const auto &[key, shape] : source.aggregate->shapes) {
      if (startsWith(key, path)) {
        result.aggregate->shapes[builder.getArrayAttr(key.getValue().drop_front(path.size()))] =
            shape;
      }
    }
    // A prefix of an array's coordinates selects a lower-rank array.
    auto root = source.aggregate->shapes.find(builder.getArrayAttr({}));
    if (root != source.aggregate->shapes.end() && path.size() < root->second.size() &&
        llvm::all_of(path, [](mlir::Attribute part) {
      return llvm::isa<mlir::IntegerAttr>(part);
    })) {
      result.aggregate->shapes[builder.getArrayAttr({})] =
          llvm::SmallVector<int64_t>(llvm::ArrayRef(root->second).drop_front(path.size()));
    }
    return result;
  }

  /// Record the immutable dimensions of a fresh local array.
  void setShape(const DiscoveryValue &value, llvm::ArrayRef<int64_t> shape) {
    value.aggregate->shapes[builder.getArrayAttr({})] = llvm::SmallVector<int64_t>(shape);
  }

  /// Return concrete root dimensions when available, including after value copies.
  llvm::ArrayRef<int64_t> getShape(const DiscoveryValue &value) {
    if (value.aggregate) {
      auto it = value.aggregate->shapes.find(builder.getArrayAttr({}));
      if (it != value.aggregate->shapes.end()) {
        return it->second;
      }
    }
    return {};
  }

  /// Replace a subtree, clearing old facts even when the new value is unknown.
  void write(
      const DiscoveryValue &destination, llvm::ArrayRef<mlir::Attribute> path,
      const DiscoveryValue &value
  ) {
    if (!destination.aggregate) {
      invalidate();
      return;
    }
    DiscoveryAggregate snapshot = value.aggregate ? *value.aggregate : DiscoveryAggregate();
    destination.aggregate->leaves.remove_if([path](const auto &entry) {
      return startsWith(entry.first, path);
    });
    if (value.constant) {
      destination.aggregate->leaves[builder.getArrayAttr(path)] = value.constant;
    }
    destination.aggregate->shapes.remove_if([path](const auto &entry) {
      return startsWith(entry.first, path);
    });
    for (const auto &[key, shape] : snapshot.shapes) {
      llvm::SmallVector<mlir::Attribute> target(path);
      llvm::append_range(target, key.getValue());
      destination.aggregate->shapes[builder.getArrayAttr(target)] = shape;
    }
    for (auto [key, constant] : snapshot.leaves) {
      llvm::SmallVector<mlir::Attribute> target(path);
      llvm::append_range(target, key.getValue());
      destination.aggregate->leaves[builder.getArrayAttr(target)] = constant;
    }
  }
};

} // namespace llzk::polymorphic::detail
