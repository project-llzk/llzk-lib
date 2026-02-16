//===-- SymbolLookup.h - Symbol Lookup Functions ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines methods symbol lookup across LLZK operations and
/// included files.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Util/Constants.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>

#include <variant>
#include <vector>

namespace llzk {

template <typename T> class SymbolLookupResult;

using ManagedResources =
    std::shared_ptr<std::pair<mlir::OwningOpRef<mlir::ModuleOp>, mlir::SymbolTableCollection>>;

class SymbolLookupResultUntyped {
public:
  SymbolLookupResultUntyped() : op(nullptr) {}
  SymbolLookupResultUntyped(mlir::Operation *opPtr) : op(opPtr) {}

  SymbolLookupResultUntyped(const SymbolLookupResultUntyped &other)
      : op(other.op), managedResources(other.managedResources),
        includeSymNameStack(other.includeSymNameStack), namespaceStack(other.namespaceStack) {}
  template <typename T> SymbolLookupResultUntyped(const SymbolLookupResult<T> &other);

  SymbolLookupResultUntyped &operator=(const SymbolLookupResultUntyped &other) {
    this->op = other.op;
    this->managedResources = other.managedResources;
    this->includeSymNameStack = other.includeSymNameStack;
    this->namespaceStack = other.namespaceStack;
    return *this;
  }
  template <typename T> SymbolLookupResultUntyped &operator=(const SymbolLookupResult<T> &other);

  SymbolLookupResultUntyped(SymbolLookupResultUntyped &&other)
      : op(other.op), managedResources(std::move(other.managedResources)),
        includeSymNameStack(std::move(other.includeSymNameStack)),
        namespaceStack(std::move(other.namespaceStack)) {
    other.op = nullptr;
  }
  template <typename T> SymbolLookupResultUntyped(SymbolLookupResult<T> &&other);

  SymbolLookupResultUntyped &operator=(SymbolLookupResultUntyped &&other) {
    if (this != &other) {
      this->op = other.op;
      other.op = nullptr;
      this->managedResources = std::move(other.managedResources);
      this->includeSymNameStack = std::move(other.includeSymNameStack);
      this->namespaceStack = std::move(other.namespaceStack);
    }
    return *this;
  }
  template <typename T> SymbolLookupResultUntyped &operator=(SymbolLookupResult<T> &&other);

  /// Access the internal operation.
  mlir::Operation *operator->();
  mlir::Operation &operator*();
  mlir::Operation &operator*() const;
  mlir::Operation *get();
  mlir::Operation *get() const;

  /// True iff the symbol was found.
  operator bool() const;

  /// Return the stack of symbol names from the IncludeOp that were traversed to load this result.
  std::vector<llvm::StringRef> getIncludeSymNames() const { return includeSymNameStack; }

  /// Return the stack of symbol names from either IncludeOp or ModuleOp that were traversed to load
  /// this result.
  llvm::ArrayRef<llvm::StringRef> getNamespace() const { return namespaceStack; }

  /// Return 'true' if at least one IncludeOp was traversed to load this result.
  bool viaInclude() const { return !includeSymNameStack.empty(); }

  mlir::SymbolTableCollection *getSymbolTableCache() {
    if (managedResources) {
      return &managedResources->second;
    } else {
      return nullptr;
    }
  }

  /// True iff the symbol is managed (i.e., loaded via an IncludeOp).
  bool isManaged() const { return managedResources != nullptr; }

  /// Adds a pointer to the set of resources the result has to manage the lifetime of.
  void manage(mlir::OwningOpRef<mlir::ModuleOp> &&ptr, mlir::SymbolTableCollection &&tables);

  /// Adds the symbol name from the IncludeOp that caused the module to be loaded.
  void trackIncludeAsName(llvm::StringRef includeOpSymName);

  /// Adds the symbol name from an IncludeOp or ModuleOp where the op is contained.
  void pushNamespace(llvm::StringRef symName);

  /// Adds the given namespace to the beginning of this result's namespace.
  void prependNamespace(llvm::ArrayRef<llvm::StringRef> ns);

  bool operator==(const SymbolLookupResultUntyped &rhs) const { return op == rhs.op; }

private:
  mlir::Operation *op;
  /// Owns the ModuleOp that contains 'op' if it was loaded via an IncludeOp along with the
  /// SymbolTableCollection for that ModuleOp which should be used for lookups involving 'op'.
  ManagedResources managedResources;
  /// Stack of symbol names from the IncludeOp that were traversed in order to load the Operation.
  std::vector<llvm::StringRef> includeSymNameStack;
  /// Stack of symbol names from the IncludeOp or ModuleOp that were traversed in order to load the
  /// Operation.
  std::vector<llvm::StringRef> namespaceStack;

  friend class Within;
};

template <typename T> class SymbolLookupResult {
public:
  SymbolLookupResult(SymbolLookupResultUntyped &&innerRes) : inner(std::move(innerRes)) {}

  /// Access the internal operation as type T.
  /// Follows the behaviors of llvm::dyn_cast if the internal operation cannot cast to that type.
  T operator->() { return llvm::dyn_cast<T>(*inner); }
  T operator*() { return llvm::dyn_cast<T>(*inner); }
  const T operator*() const { return llvm::dyn_cast<T>(*inner); }
  T get() { return llvm::dyn_cast<T>(inner.get()); }
  T get() const { return llvm::dyn_cast<T>(inner.get()); }

  operator bool() const { return inner && llvm::isa<T>(*inner); }

  /// Return the stack of symbol names from the IncludeOp that were traversed to load this result.
  std::vector<llvm::StringRef> getIncludeSymNames() const { return inner.getIncludeSymNames(); }

  /// Return the stack of symbol names from either IncludeOp or ModuleOp that were traversed to load
  /// this result.
  llvm::ArrayRef<llvm::StringRef> getNamespace() const { return inner.getNamespace(); }

  /// Adds the given namespace to the beginning of this result's namespace.
  void prependNamespace(llvm::ArrayRef<llvm::StringRef> ns) { inner.prependNamespace(ns); }

  /// Return 'true' if at least one IncludeOp was traversed to load this result.
  bool viaInclude() const { return inner.viaInclude(); }

  bool operator==(const SymbolLookupResult<T> &rhs) const { return inner == rhs.inner; }

  /// Return 'true' if the inner resource is managed.
  bool isManaged() const { return inner.isManaged(); }

private:
  SymbolLookupResultUntyped inner;

  friend class Within;
  friend class SymbolLookupResultUntyped;
};

// These methods' definitions need to be here, after the declaration of SymbolLookupResult<T>

template <typename T>
SymbolLookupResultUntyped::SymbolLookupResultUntyped(const SymbolLookupResult<T> &other)
    : SymbolLookupResultUntyped(other.inner) {}

template <typename T>
SymbolLookupResultUntyped &
SymbolLookupResultUntyped::operator=(const SymbolLookupResult<T> &other) {
  *this = other.inner;
  return *this;
}

template <typename T>
SymbolLookupResultUntyped::SymbolLookupResultUntyped(SymbolLookupResult<T> &&other)
    : SymbolLookupResultUntyped(std::move(other.inner)) {}

template <typename T>
SymbolLookupResultUntyped &SymbolLookupResultUntyped::operator=(SymbolLookupResult<T> &&other) {
  *this = std::move(other.inner);
  return *this;
}

class Within {
public:
  /// Lookup within the top-level (root) module
  Within() : from(nullptr) {}
  /// Lookup within the given Operation (cannot be nullptr)
  Within(mlir::Operation *op) : from(op) { assert(op && "cannot lookup within nullptr"); }
  /// Lookup within the Operation of the given result and transfer managed resources
  Within(SymbolLookupResultUntyped &&res) : from(std::move(res)) {}
  /// Lookup within the Operation of the given result and transfer managed resources
  template <typename T> Within(SymbolLookupResult<T> &&res) : Within(std::move(res.inner)) {}

  Within(const Within &) = delete;
  Within(Within &&other) noexcept : from(std::move(other.from)) {}
  Within &operator=(const Within &) = delete;
  Within &operator=(Within &&) noexcept;

  inline static Within root() { return Within(); }

  mlir::FailureOr<SymbolLookupResultUntyped> lookup(
      mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr symbol, mlir::Operation *origin,
      bool reportMissing = true
  ) &&;

private:
  std::variant<mlir::Operation *, SymbolLookupResultUntyped> from;
};

inline mlir::FailureOr<SymbolLookupResultUntyped> lookupSymbolIn(
    mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr symbol, Within &&lookupWithin,
    mlir::Operation *origin, bool reportMissing = true
) {
  return std::move(lookupWithin).lookup(tables, symbol, origin, reportMissing);
}

inline mlir::FailureOr<SymbolLookupResultUntyped> lookupTopLevelSymbol(
    mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr symbol, mlir::Operation *origin,
    bool reportMissing = true
) {
  return Within().lookup(tables, symbol, origin, reportMissing);
}

template <typename T>
inline mlir::FailureOr<SymbolLookupResult<T>> lookupSymbolIn(
    mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr symbol, Within &&lookupWithin,
    mlir::Operation *origin, bool reportMissing = true
) {
  auto found = lookupSymbolIn(tables, symbol, std::move(lookupWithin), origin, reportMissing);
  if (mlir::failed(found)) {
    return mlir::failure(); // lookupSymbolIn() already emits a sufficient error message
  }
  // Keep a copy of the op ptr in case we need it for displaying diagnostics
  mlir::Operation *op = found->get();
  // ... since the untyped result gets moved here into a typed result.
  SymbolLookupResult<T> ret(std::move(*found));
  if (!ret) {
    if (reportMissing) {
      return origin->emitError() << "symbol \"" << symbol << "\" references a '" << op->getName()
                                 << "' but expected a '" << T::getOperationName() << '\'';
    } else {
      return mlir::failure();
    }
  }
  return ret;
}

template <typename T>
inline mlir::FailureOr<SymbolLookupResult<T>> lookupTopLevelSymbol(
    mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr symbol, mlir::Operation *origin,
    bool reportMissing = true
) {
  return lookupSymbolIn<T>(tables, symbol, Within(), origin, reportMissing);
}

} // namespace llzk
