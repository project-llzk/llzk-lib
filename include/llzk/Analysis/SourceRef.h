//===-- SourceRef.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Analysis/AbstractLatticeValue.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/ErrorHelper.h"
#include "llzk/Util/Hash.h"

#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Pass/AnalysisManager.h>

#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/EquivalenceClasses.h>

#include <unordered_set>
#include <vector>

namespace llzk {

/// @brief Defines an index into an LLZK object. For structs, this is a member
/// definition, and for arrays, this is an element index.
/// Effectively a wrapper around a std::variant with extra utility methods.
class SourceRefIndex {
  using IndexRange = std::pair<llvm::DynamicAPInt, llvm::DynamicAPInt>;

public:
  explicit SourceRefIndex(component::MemberDefOp f) : index(f) {}
  explicit SourceRefIndex(SymbolLookupResult<component::MemberDefOp> f) : index(f) {}
  explicit SourceRefIndex(const llvm::DynamicAPInt &i) : index(i) {}
  explicit SourceRefIndex(const llvm::APInt &i) : index(toDynamicAPInt(i)) {}
  explicit SourceRefIndex(int64_t i) : index(llvm::DynamicAPInt(i)) {}
  SourceRefIndex(const llvm::APInt &low, const llvm::APInt &high)
      : index(IndexRange {toDynamicAPInt(low), toDynamicAPInt(high)}) {}
  explicit SourceRefIndex(IndexRange r) : index(r) {}

  bool isMember() const {
    return std::holds_alternative<SymbolLookupResult<component::MemberDefOp>>(index) ||
           std::holds_alternative<component::MemberDefOp>(index);
  }
  component::MemberDefOp getMember() const {
    ensure(isMember(), "SourceRefIndex: member requested but not contained");
    if (std::holds_alternative<component::MemberDefOp>(index)) {
      return std::get<component::MemberDefOp>(index);
    }
    return std::get<SymbolLookupResult<component::MemberDefOp>>(index).get();
  }

  bool isIndex() const { return std::holds_alternative<llvm::DynamicAPInt>(index); }
  llvm::DynamicAPInt getIndex() const {
    ensure(isIndex(), "SourceRefIndex: index requested but not contained");
    return std::get<llvm::DynamicAPInt>(index);
  }

  bool isIndexRange() const { return std::holds_alternative<IndexRange>(index); }
  IndexRange getIndexRange() const {
    ensure(isIndexRange(), "SourceRefIndex: index range requested but not contained");
    return std::get<IndexRange>(index);
  }

  inline void dump() const { print(llvm::errs()); }
  void print(mlir::raw_ostream &os) const;

  inline bool operator==(const SourceRefIndex &rhs) const {
    if (isMember() && rhs.isMember()) {
      // We compare the underlying members, since the member could be in a symbol
      // lookup or not.
      return getMember() == rhs.getMember();
    }
    if (isIndex() && rhs.isIndex()) {
      return getIndex() == rhs.getIndex();
    }
    return index == rhs.index;
  }

  bool operator<(const SourceRefIndex &rhs) const;

  bool operator>(const SourceRefIndex &rhs) const { return rhs < *this; }

  struct Hash {
    size_t operator()(const SourceRefIndex &c) const;
  };

  inline size_t getHash() const { return Hash {}(*this); }

private:
  /// Either:
  /// 1. A member within a struct (possibly as a SymbolLookupResult to be cautious of external
  /// module scopes)
  /// 2. An index into an array
  /// 3. A half-open range of indices into an array, for when we're unsure about a specific index
  /// Likely, this will be from [0, size) at this point.
  std::variant<
      component::MemberDefOp, SymbolLookupResult<component::MemberDefOp>, llvm::DynamicAPInt,
      IndexRange>
      index;
};

static inline mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const SourceRefIndex &rhs) {
  rhs.print(os);
  return os;
}

/// @brief A reference to a "source", which is the base value from which other
/// SSA values are derived.
/// The object may be a reference to an individual felt, felt.const, or a composite type,
/// like an array or an entire struct.
/// - SourceRefs are allowed to reference composite types so that references can be generated
/// for intermediate operations (e.g., readm to read a nested struct).
///
/// These references are relative to a particular function call, so they are either (1) constants,
/// or (2) rooted at a block argument (which is either "self" in @constrain
/// functions or another input) and optionally contain indices into that block
/// argument (e.g., a member reference in a struct or a index into an array).
class SourceRef {
  /**
   * BlockArgument:
   * - If the block arg is 0, then it refers to "self", meaning the signal is internal or an output
   * (public means an output).
   * - Otherwise, it is an input, either public or private.
   *
   * CreateStructOp
   * - For compute functions, the "self" argument is an allocation site.
   *
   * NonDetOp
   * - For constrain functions, a nondet operation is also an "allocation" site,
   *   as it is a value that can be constrained/constrain other values.
   */
  using Root = std::variant<mlir::BlockArgument, component::CreateStructOp, NonDetOp>;
  using Constant =
      std::variant<felt::FeltConstantOp, mlir::arith::ConstantIndexOp, polymorphic::ConstReadOp>;
  using Path = std::vector<SourceRefIndex>;
  struct Rooted {
    Root root;
    Path path;
  };
  // using mutable to reduce constant casts for certain get* functions.
  mutable std::variant<Rooted, Constant> storage;

  template <typename C> bool isConstantType() const {
    if (const Constant *c = std::get_if<Constant>(&storage)) {
      return std::holds_alternative<C>(*c);
    }
    return false;
  }

  template <typename C> C getConstantType() const {
    assert(isConstantType<C>() && "wrong constant type");
    return std::get<C>(std::get<Constant>(storage));
  }

  template <typename R> bool isRootType() const {
    if (const Rooted *r = std::get_if<Rooted>(&storage)) {
      return std::holds_alternative<R>(r->root);
    }
    return false;
  }

  template <typename R> R getRootType() const {
    assert(isRootType<R>() && "wrong root type");
    return std::get<R>(std::get<Rooted>(storage).root);
  }

  const Root &getRoot() const { return std::get<Rooted>(storage).root; }

  const Constant &getConstant() const { return std::get<Constant>(storage); }

  Path &getPath() { return std::get<Rooted>(storage).path; }

public:
  /// Produce all possible SourceRefs that are present starting from the given root.
  static std::vector<SourceRef>
  getAllSourceRefs(mlir::SymbolTableCollection &tables, mlir::ModuleOp mod, const SourceRef &root);

  /// Produce all possible SourceRefs that are present from given struct function.
  static std::vector<SourceRef>
  getAllSourceRefs(component::StructDefOp structDef, function::FuncDefOp fnOp);

  /// Produce all possible SourceRefs from a specific member in a struct.
  /// May produce multiple if the given member is of an aggregate type.
  static std::vector<SourceRef>
  getAllSourceRefs(component::StructDefOp structDef, component::MemberDefOp memberDef);

  /* Rooted constructors */

  SourceRef(mlir::BlockArgument b, Path path = {}) : storage(Rooted {.root = b, .path = path}) {}
  SourceRef(component::CreateStructOp createOp, Path path = {})
      : storage(Rooted {.root = createOp, .path = path}) {}
  SourceRef(NonDetOp nondet, Path path = {}) : storage(Rooted {.root = nondet, .path = path}) {}

  /* Constant constructors */

  explicit SourceRef(felt::FeltConstantOp c) : storage(c) {}
  explicit SourceRef(mlir::arith::ConstantIndexOp c) : storage(c) {}
  explicit SourceRef(polymorphic::ConstReadOp c) : storage(c) {}

  mlir::Type getType() const;

  bool isConstantFelt() const { return isConstantType<felt::FeltConstantOp>(); }
  bool isConstantIndex() const { return isConstantType<mlir::arith::ConstantIndexOp>(); }
  bool isTemplateConstant() const { return isConstantType<polymorphic::ConstReadOp>(); }

  bool isConstant() const { return std::holds_alternative<Constant>(storage); }
  bool isConstantInt() const { return isConstantFelt() || isConstantIndex(); }

  bool isFeltVal() const { return llvm::isa<felt::FeltType>(getType()); }
  bool isIndexVal() const { return llvm::isa<mlir::IndexType>(getType()); }
  bool isIntegerVal() const { return llvm::isa<mlir::IntegerType>(getType()); }
  bool isTypeVarVal() const { return llvm::isa<polymorphic::TypeVarType>(getType()); }
  bool isScalar() const {
    return isConstant() || isFeltVal() || isIndexVal() || isIntegerVal() || isTypeVarVal();
  }

  bool isRooted() const { return std::holds_alternative<Rooted>(storage); }
  bool isBlockArgument() const { return isRootType<mlir::BlockArgument>(); }
  mlir::BlockArgument getBlockArgument() const { return getRootType<mlir::BlockArgument>(); }
  unsigned getInputNum() const { return getBlockArgument().getArgNumber(); }

  bool isCreateStructOp() const { return isRootType<component::CreateStructOp>(); }
  component::CreateStructOp getCreateStructOp() const {
    return getRootType<component::CreateStructOp>();
  }

  bool isNonDetOp() const { return isRootType<NonDetOp>(); }
  NonDetOp getNonDetOp() const { return getRootType<NonDetOp>(); }

  llvm::DynamicAPInt getConstantFeltValue() const {
    llvm::APInt i = getConstantType<felt::FeltConstantOp>().getValue();
    return toDynamicAPInt(i);
  }
  llvm::DynamicAPInt getConstantIndexValue() const {
    return llvm::DynamicAPInt(getConstantType<mlir::arith::ConstantIndexOp>().value());
  }
  llvm::DynamicAPInt getConstantValue() const {
    ensure(
        isConstantFelt() || isConstantIndex(),
        mlir::Twine(mlir::StringRef(__FUNCTION__), " requires a constant int type!")
    );
    return isConstantFelt() ? getConstantFeltValue() : getConstantIndexValue();
  }

  /// @brief Returns true iff `prefix` is a valid prefix of this reference.
  bool isValidPrefix(const SourceRef &prefix) const;

  /// @brief If `prefix` is a valid prefix of this reference, return the suffix that
  /// remains after removing the prefix. I.e., `this` = `prefix` + `suffix`
  /// @param prefix
  /// @return the suffix
  mlir::FailureOr<std::vector<SourceRefIndex>> getSuffix(const SourceRef &prefix) const;

  /// @brief Create a new reference with prefix replaced with other iff prefix is a valid prefix for
  /// this reference. If this reference is a felt.const, the translation will always succeed and
  /// return the felt.const unchanged.
  /// @param prefix
  /// @param other
  /// @return
  mlir::FailureOr<SourceRef> translate(const SourceRef &prefix, const SourceRef &other) const;

  /// @brief Create a new reference that is the immediate prefix of this reference if possible.
  mlir::FailureOr<SourceRef> getParentPrefix() const {
    if (!isRooted() || getPath().empty()) {
      return mlir::failure();
    }
    auto copy = *this;
    copy.getPath().pop_back();
    return copy;
  }

  /// @brief Get all direct children of this SourceRef, assuming this ref is not a scalar.
  std::vector<SourceRef>
  getAllChildren(mlir::SymbolTableCollection &tables, mlir::ModuleOp mod) const;

  SourceRef createChild(const SourceRefIndex &r) const {
    auto copy = *this;
    copy.getPath().push_back(r);
    return copy;
  }

  SourceRef createChild(const SourceRef &other) const {
    assert(other.isConstantIndex());
    return createChild(SourceRefIndex(other.getConstantIndexValue()));
  }

  [[deprecated("Use getPath() instead")]]
  const Path &getPieces() const {
    return std::get<Rooted>(storage).path;
  }
  const Path &getPath() const { return std::get<Rooted>(storage).path; }

  void print(mlir::raw_ostream &os) const;
  void dump() const { print(llvm::errs()); }

  bool operator==(const SourceRef &rhs) const;

  bool operator!=(const SourceRef &rhs) const { return !(*this == rhs); }

  // required for EquivalenceClasses usage
  bool operator<(const SourceRef &rhs) const;

  bool operator>(const SourceRef &rhs) const { return rhs < *this; }

  struct Hash {
    size_t operator()(const SourceRef &val) const;
  };
};

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const SourceRef &rhs);

/* SourceRefSet */

class SourceRefSet : public std::unordered_set<SourceRef, SourceRef::Hash> {
  using Base = std::unordered_set<SourceRef, SourceRef::Hash>;

public:
  using Base::Base;

  SourceRefSet &join(const SourceRefSet &rhs);

  friend mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const SourceRefSet &rhs);
};

static_assert(
    dataflow::ScalarLatticeValue<SourceRefSet>,
    "SourceRefSet must satisfy the ScalarLatticeValue requirements"
);

} // namespace llzk

namespace llvm {

template <> struct DenseMapInfo<llzk::SourceRef> {
  static llzk::SourceRef getEmptyKey() {
    return llzk::SourceRef(mlir::BlockArgument(reinterpret_cast<mlir::detail::ValueImpl *>(1)));
  }
  static inline llzk::SourceRef getTombstoneKey() {
    return llzk::SourceRef(mlir::BlockArgument(reinterpret_cast<mlir::detail::ValueImpl *>(2)));
  }
  static unsigned getHashValue(const llzk::SourceRef &ref) {
    if (ref == getEmptyKey() || ref == getTombstoneKey()) {
      return llvm::hash_value(ref.getBlockArgument().getAsOpaquePointer());
    }
    return llzk::SourceRef::Hash {}(ref);
  }
  static bool isEqual(const llzk::SourceRef &lhs, const llzk::SourceRef &rhs) { return lhs == rhs; }
};

} // namespace llvm
