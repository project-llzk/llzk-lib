//===-- ConstraintBindings.h ------------------------------------*- C++ -*-===//
// Part of the LLZK Project, under the Apache License v2.0.
// SPDX-License-Identifier: Apache-2.0

#ifndef LLZK_CONSTRAINT_BINDINGS_H
#define LLZK_CONSTRAINT_BINDINGS_H

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/POD/IR/Types.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

#include <llvm/ADT/DenseMap.h>

#include <variant>

namespace llzk::polymorphic::detail {

/// A concrete subtree of one execution's storage. Types describe its shape;
/// string and integer steps identify records and array coordinates respectively.
struct AccessPath {
  unsigned root;
  mlir::ArrayAttr steps;
  mlir::Type type;
};

/// Scalar expressions are SSA; aggregates are paths, never emitted allocations.
using BoundValue = std::variant<mlir::Value, AccessPath>;
using Environment = llvm::DenseMap<mlir::Value, BoundValue>;

/// Bind scalar leaves without constructing a shadow array or POD object.
/// Function inputs and original storage are immutable roots. Local roots contain
/// independent bindings, including when their scalar values originated in inputs.
class ConstraintBindings {
  struct Root {
    mlir::Value input;
    mlir::Value storage;
    llvm::DenseMap<mlir::ArrayAttr, mlir::Value> leaves;
  };

  mlir::Builder builder;
  llvm::SmallVector<Root> roots;
  uint64_t &steps;
  uint64_t limit;

public:
  using StorageReader = llvm::function_ref<mlir::FailureOr<mlir::Value>(
      mlir::Value, llvm::ArrayRef<mlir::Attribute>, mlir::Type, mlir::Operation *
  )>;

  ConstraintBindings(mlir::MLIRContext *context, uint64_t &work, uint64_t maximum)
      : builder(context), steps(work), limit(maximum) {}

  /// Create a root for a local allocation, function argument, or original storage.
  AccessPath create(mlir::Type type, mlir::Value input = {}, mlir::Value storage = {}) {
    unsigned id = roots.size();
    roots.emplace_back();
    roots.back().input = input;
    roots.back().storage = storage;
    return {id, builder.getArrayAttr({}), type};
  }

  /// Extend an already-validated path with one record or coordinate.
  AccessPath child(AccessPath parent, mlir::Attribute step, mlir::Type type) {
    llvm::SmallVector<mlir::Attribute> path(parent.steps.begin(), parent.steps.end());
    path.push_back(step);
    return {parent.root, builder.getArrayAttr(path), type};
  }

  /// Select one dimension; callers check that the coordinate is static/in-bounds.
  AccessPath element(AccessPath parent, int64_t index) {
    auto type = llvm::cast<array::ArrayType>(parent.type);
    auto shape = type.getShape().drop_front();
    mlir::Type selected =
        shape.empty() ? type.getElementType() : type.cloneWith(shape, type.getElementType());
    return child(parent, builder.getIndexAttr(index), selected);
  }

  /// Resolve one concrete step, including those transferred into a destination.
  mlir::FailureOr<AccessPath> select(AccessPath path, mlir::Attribute step, mlir::Operation *site) {
    if (auto index = llvm::dyn_cast<mlir::IntegerAttr>(step)) {
      auto type = llvm::dyn_cast<array::ArrayType>(path.type);
      if (!type || type.getShape().front() < 0) {
        return site->emitError("constraint flattening: aggregate shape must be static");
      }
      int64_t coordinate = index.getInt();
      if (coordinate < 0 || coordinate >= type.getShape().front()) {
        return site->emitError("constraint flattening: array path index is out of bounds");
      }
      return element(path, coordinate);
    }
    auto type = llvm::dyn_cast<pod::PodType>(path.type);
    auto name = llvm::dyn_cast<mlir::StringAttr>(step);
    if (!type || !name) {
      return site->emitError("constraint flattening: invalid POD path");
    }
    auto record = type.getRecord(name.getValue(), [&] { return site->emitError(); });
    if (mlir::failed(record)) {
      return mlir::failure();
    }
    return child(path, name, *record);
  }

  /// Visit array elements in row-major order, preserving aggregate element types.
  mlir::LogicalResult elements(
      AccessPath path, mlir::Operation *site,
      llvm::function_ref<mlir::LogicalResult(AccessPath)> visit
  ) {
    auto type = llvm::dyn_cast<array::ArrayType>(path.type);
    if (!type) {
      return visit(path);
    }
    int64_t size = type.getShape().front();
    if (size < 0) {
      return site->emitError("constraint flattening: aggregate shape must be static");
    }
    for (int64_t i = 0; i < size; ++i) {
      if (++steps > limit) {
        return site->emitError("constraint flattening: evaluation step limit exceeded");
      }
      if (mlir::failed(elements(element(path, i), site, visit))) {
        return mlir::failure();
      }
    }
    return mlir::success();
  }

  /// Enumerate scalar leaves using types, without allocating aggregate objects.
  mlir::LogicalResult walk(
      AccessPath path, mlir::Operation *site,
      llvm::function_ref<mlir::LogicalResult(AccessPath)> visit
  ) {
    if (llvm::isa<mlir::NoneType>(path.type)) {
      return mlir::success();
    }
    if (llvm::isa<array::ArrayType>(path.type)) {
      return elements(path, site, [&](AccessPath item) { return walk(item, site, visit); });
    }
    if (auto type = llvm::dyn_cast<pod::PodType>(path.type)) {
      for (auto record : type.getRecords()) {
        auto field = child(path, record.getName(), record.getType());
        if (mlir::failed(walk(field, site, visit))) {
          return mlir::failure();
        }
      }
      return mlir::success();
    }
    return visit(path);
  }

  /// Resolve a scalar binding; original storage is read only when needed.
  mlir::FailureOr<mlir::Value> load(AccessPath path, mlir::Operation *site, StorageReader read) {
    auto &leaves = roots[path.root].leaves;
    if (auto found = leaves.find(path.steps); found != leaves.end()) {
      return found->second;
    }
    if (mlir::Value storage = roots[path.root].storage) {
      auto value = read(storage, path.steps.getValue(), path.type, site);
      if (mlir::failed(value)) {
        return mlir::failure();
      }
      leaves[path.steps] = *value;
      return *value;
    }
    site->emitError(
        roots[path.root].input
            ? "constraint flattening: read of uninitialized function input storage"
            : "constraint flattening: read of uninitialized local storage"
    );
    return mlir::failure();
  }

  /// Reject writes based on destination identity, never scalar provenance.
  mlir::LogicalResult writable(AccessPath path, mlir::Operation *site) const {
    const Root &root = roots[path.root];
    if (root.input || root.storage) {
      return site->emitError(
          "constraint flattening: cannot write to function input or original storage"
      );
    }
    return mlir::success();
  }

  /// Bind a scalar leaf during initialization or after checking writability.
  void store(AccessPath path, mlir::Value value) { roots[path.root].leaves[path.steps] = value; }

  /// Snapshot leaves before replacing a subtree, including overlapping copies.
  /// Missing local leaves stay uninitialized; copying does not read them.
  mlir::LogicalResult
  transfer(AccessPath destination, AccessPath source, mlir::Operation *site, StorageReader read) {
    llvm::SmallVector<std::pair<mlir::ArrayAttr, mlir::Value>> snapshot;
    bool original = bool(roots[source.root].storage);
    auto collect = [&](AccessPath leaf) -> mlir::LogicalResult {
      const auto &leaves = roots[leaf.root].leaves;
      if (!leaves.contains(leaf.steps) && !original) {
        return mlir::success();
      }
      auto value = load(leaf, site, read);
      if (mlir::failed(value)) {
        return mlir::failure();
      }
      AccessPath target = destination;
      for (auto step : leaf.steps.getValue().drop_front(source.steps.size())) {
        auto selected = select(target, step, site);
        if (mlir::failed(selected)) {
          return mlir::failure();
        }
        target = *selected;
      }
      snapshot.push_back({target.steps, *value});
      return mlir::success();
    };
    if (mlir::failed(walk(source, site, collect))) {
      return mlir::failure();
    }
    auto prefix = destination.steps.getValue();
    auto &targetLeaves = roots[destination.root].leaves;
    llvm::SmallVector<mlir::ArrayAttr> replaced;
    for (auto [path, value] : targetLeaves) {
      if (path.size() >= prefix.size() &&
          llvm::equal(path.getValue().take_front(prefix.size()), prefix)) {
        replaced.push_back(path);
      }
    }
    for (auto path : replaced) {
      targetLeaves.erase(path);
    }
    for (auto [path, value] : snapshot) {
      targetLeaves[path] = value;
    }
    return mlir::success();
  }
};

} // namespace llzk::polymorphic::detail
#endif
