//===-- UsedFreeFunctions.h -------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Function/IR/Ops.h"

#include <mlir/IR/Attributes.h>

namespace pcl::lowering {

/// Set of free functions that are used by the `@constrain` functions in the circuit.
///
/// Functions are matched by their FQN. This means that a function op with identical FQN
/// to one already inserted in the set will be reported as being already in the set. This is done
/// for handling the analysis step of the stubbed mode, where the IR is cloned to avoid destructive
/// actions affecting the original IR.
///
/// For that reason, this mapping is missing the U in CRUD and we only insert during creation.
class UsedFreeFunctions {
  using M = llvm::DenseMap<mlir::Attribute, llzk::function::FuncDefOp>;

  /// Maps the FQN to the operation representing it.
  M funcs;

  /// Inserts the function into the mapping.
  void insert(llzk::function::FuncDefOp op) { funcs.insert({op.getFullyQualifiedName(), op}); }

  /// Convenience method for the construction of the mapping.
  void insertCallees(
      llvm::SmallVectorImpl<llzk::function::FuncDefOp> &WL, mlir::SymbolTableCollection &tables,
      llzk::function::FuncDefOp f, llvm::function_ref<bool(llzk::function::FuncDefOp)> P = nullptr
  );

public:
  /// Collects the call-graph from the constrain functions (excluding the `@constrain` functions
  /// themselves). Since we only care if a given `function.def` is part of the graph or not we
  /// return the set of vertices.
  UsedFreeFunctions(mlir::ModuleOp op);

  /// Returns whether the function is in the set or not.
  bool contains(llzk::function::FuncDefOp op) const {
    return funcs.contains(op.getFullyQualifiedName());
  }

  /// Removes the function from the mapping.
  ///
  /// As a safety precaution, is only erased if the given function is equal to the
  /// one mapped by the FQN.
  void erase(llzk::function::FuncDefOp op);

  class iterator {
    using It = M::iterator;
    It iter;

  public:
    using difference_type = It::difference_type;
    using value_type = llzk::function::FuncDefOp;
    using pointer = value_type *;
    using reference = value_type &;
    using iterator_category = It::iterator_category;

    iterator() = default;
    iterator(It I) : iter(I) {}

    reference operator*() { return iter->getSecond(); }

    iterator &operator++() {
      ++iter;
      return *this;
    }

    iterator operator++(int) { return iterator(iter++); }

    friend bool operator!=(const iterator &LHS, const iterator &RHS);
  };

  iterator begin() { return iterator(funcs.begin()); }
  iterator end() { return iterator(funcs.end()); }
};

bool operator!=(const UsedFreeFunctions::iterator &LHS, const UsedFreeFunctions::iterator &RHS);
} // namespace pcl::lowering
