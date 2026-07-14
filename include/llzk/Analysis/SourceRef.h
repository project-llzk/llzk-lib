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
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/ErrorHelper.h"
#include "llzk/Util/Hash.h"

#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Pass/AnalysisManager.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/EquivalenceClasses.h>
#include <llvm/ADT/TypeSwitch.h>

#include <compare>
#include <unordered_set>
#include <variant>
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
  explicit SourceRefIndex(mlir::StringAttr recordName) : index(recordName) {}
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

  bool isPodRecord() const { return std::holds_alternative<mlir::StringAttr>(index); }
  mlir::StringAttr getPodRecordNameAttr() const {
    ensure(isPodRecord(), "SourceRefIndex: pod record requested but not contained");
    return std::get<mlir::StringAttr>(index);
  }
  llvm::StringRef getPodRecordName() const { return getPodRecordNameAttr().getValue(); }

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

  std::strong_ordering operator<=>(const SourceRefIndex &rhs) const;

  struct Hash {
    size_t operator()(const SourceRefIndex &c) const;
  };

  inline size_t getHash() const { return Hash {}(*this); }

private:
  /// Either:
  /// 1. A member within a struct (possibly as a SymbolLookupResult to be cautious of external
  /// module scopes)
  /// 2. A record within a pod
  /// 3. An index into an array
  /// 4. A half-open range of indices into an array, for when we're unsure about a specific index
  /// Likely, this will be from [0, size) at this point.
  std::variant<
      component::MemberDefOp, SymbolLookupResult<component::MemberDefOp>, mlir::StringAttr,
      llvm::DynamicAPInt, IndexRange>
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
/// or (2) rooted at a value (e.g., `self`, a nondet op, an external call result, an array.new
/// result, an array.read result, or another block argument),
/// and optionally contain indices into root (e.g., a member reference in a struct or a index into
/// an array).
class SourceRef {
public:
  using Path = std::vector<SourceRefIndex>;

private:
  // Sort in the following order:
  // block arg < struct.new < nondet < other rooted result < template const < const index <
  // const felt.
  enum class SortCategory : std::uint8_t {
    BlockArgument,
    CreateStruct,
    NonDet,
    RootResult,
    TemplateConstant,
    ConstantIndex,
    ConstantFelt,
  };

  template <typename OpT> static mlir::Value getSingleResultValue(OpT op) {
    ensure(op, "SourceRef requires a non-null operation");
    ensure(op->getNumResults() == 1, "SourceRef expects a single-result operation");
    return op->getResult(0);
  }

  static mlir::Value getRootResultValue(mlir::OpResult result) {
    ensure(
        !llvm::isa<
            felt::FeltConstantOp, mlir::arith::ConstantIndexOp, mlir::arith::ConstantIntOp,
            polymorphic::ConstReadOp>(result.getOwner()),
        "SourceRef rooted OpResult constructors must not be used for constant values"
    );
    return result;
  }

  template <typename OpT> mlir::FailureOr<OpT> getDefiningOp() const {
    if (auto op = llvm::dyn_cast_if_present<OpT>(value.getDefiningOp())) {
      return op;
    }
    return mlir::failure();
  }

  SourceRef(mlir::Value sourceValue, bool isConstantStorage, Path sourcePath = {})
      : value(sourceValue), path(std::move(sourcePath)), constant(isConstantStorage) {
    ensure(value != nullptr, "SourceRef requires a non-null value");
    ensure(!constant || this->path.empty(), "constant SourceRef cannot have a path");
  }

  Path &getPathMut() { return path; }
  const void *getAsOpaquePointer() const { return value.getAsOpaquePointer(); }
  SortCategory getSortCategory() const;
  llvm::StringRef getTemplateConstantName() const;
  std::strong_ordering compareWithinCategory(const SourceRef &rhs, SortCategory category) const;

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

  SourceRef(mlir::BlockArgument b, Path p = {})
      : SourceRef(b, /*isConstantStorage=*/false, std::move(p)) {}
  SourceRef(component::CreateStructOp createOp, Path p = {})
      : SourceRef(getSingleResultValue(createOp), /*isConstantStorage=*/false, std::move(p)) {}
  SourceRef(NonDetOp nondet, Path p = {})
      : SourceRef(getSingleResultValue(nondet), /*isConstantStorage=*/false, std::move(p)) {}
  SourceRef(mlir::OpResult rootResult, Path p = {})
      : SourceRef(getRootResultValue(rootResult), /*isConstantStorage=*/false, std::move(p)) {}

  /* Constant constructors */

  explicit SourceRef(felt::FeltConstantOp c)
      : SourceRef(getSingleResultValue(c), /*isConstantStorage=*/true) {}
  explicit SourceRef(mlir::arith::ConstantIndexOp c)
      : SourceRef(getSingleResultValue(c), /*isConstantStorage=*/true) {}
  explicit SourceRef(polymorphic::ConstReadOp c)
      : SourceRef(getSingleResultValue(c), /*isConstantStorage=*/true) {}

  mlir::Type getType() const;

  bool isConstantFelt() const {
    return isConstant() && llvm::isa_and_present<felt::FeltConstantOp>(value.getDefiningOp());
  }
  bool isConstantIndex() const {
    return isConstant() &&
           llvm::isa_and_present<mlir::arith::ConstantIndexOp>(value.getDefiningOp());
  }
  bool isTemplateConstant() const {
    return isConstant() && llvm::isa_and_present<polymorphic::ConstReadOp>(value.getDefiningOp());
  }

  bool isConstant() const { return constant; }
  bool isConstantInt() const { return isConstantFelt() || isConstantIndex(); }

  bool isFeltVal() const { return llvm::isa<felt::FeltType>(getType()); }
  bool isIndexVal() const { return llvm::isa<mlir::IndexType>(getType()); }
  bool isIntegerVal() const { return llvm::isa<mlir::IntegerType>(getType()); }
  bool isTypeVarVal() const { return llvm::isa<polymorphic::TypeVarType>(getType()); }
  bool isScalar() const {
    return isConstant() || isFeltVal() || isIndexVal() || isIntegerVal() || isTypeVarVal();
  }

  bool isRooted() const { return !constant; }
  bool isBlockArgument() const { return isRooted() && llvm::isa<mlir::BlockArgument>(value); }
  mlir::FailureOr<mlir::Value> getRoot() const {
    if (isRooted()) {
      return value;
    }
    return mlir::failure();
  }
  mlir::FailureOr<mlir::Value> getConstant() const {
    if (isConstant()) {
      return value;
    }
    return mlir::failure();
  }
  mlir::FailureOr<mlir::BlockArgument> getBlockArgument() const {
    if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value)) {
      return blockArg;
    }
    return mlir::failure();
  }
  mlir::FailureOr<unsigned> getInputNum() const {
    auto blockArg = getBlockArgument();
    if (succeeded(blockArg)) {
      return blockArg->getArgNumber();
    }
    return mlir::failure();
  }

  bool isCreateStructOp() const { return succeeded(getCreateStructOp()); }
  mlir::FailureOr<component::CreateStructOp> getCreateStructOp() const {
    return getDefiningOp<component::CreateStructOp>();
  }

  bool isNonDetOp() const { return succeeded(getNonDetOp()); }
  mlir::FailureOr<NonDetOp> getNonDetOp() const { return getDefiningOp<NonDetOp>(); }

  bool isCallResult() const { return succeeded(getCallOp()); }
  mlir::FailureOr<function::CallOp> getCallOp() const { return getDefiningOp<function::CallOp>(); }

  mlir::FailureOr<llvm::DynamicAPInt> getConstantFeltValue() const {
    auto feltConst = getDefiningOp<felt::FeltConstantOp>();
    if (succeeded(feltConst)) {
      llvm::APInt i = feltConst->getValue();
      return toDynamicAPInt(i);
    }
    return mlir::failure();
  }
  mlir::FailureOr<llvm::DynamicAPInt> getConstantIndexValue() const {
    auto indexConst = getDefiningOp<mlir::arith::ConstantIndexOp>();
    if (succeeded(indexConst)) {
      return llvm::DynamicAPInt(indexConst->value());
    }
    return mlir::failure();
  }
  mlir::FailureOr<llvm::DynamicAPInt> getConstantValue() const {
    auto feltVal = getConstantFeltValue();
    if (succeeded(feltVal)) {
      return *feltVal;
    }
    auto indexVal = getConstantIndexValue();
    if (succeeded(indexVal)) {
      return *indexVal;
    }
    return mlir::failure();
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
    copy.getPathMut().pop_back();
    return copy;
  }

  /// @brief Get all direct children of this SourceRef, assuming this ref is not a scalar.
  std::vector<SourceRef>
  getAllChildren(mlir::SymbolTableCollection &tables, mlir::ModuleOp mod) const;

  mlir::FailureOr<SourceRef> createChild(const SourceRefIndex &r) const {
    if (!isRooted()) {
      return mlir::failure();
    }
    auto copy = *this;
    copy.getPathMut().push_back(r);
    return copy;
  }

  mlir::FailureOr<SourceRef> createChild(const SourceRef &other) const {
    auto idxVal = other.getConstantIndexValue();
    if (failed(idxVal)) {
      return mlir::failure();
    }
    return createChild(SourceRefIndex(*idxVal));
  }

  [[deprecated("Use getPath() instead")]]
  // NOTE: When this function is removed, do not delete it, rewrite as `... = delete`.
  llvm::ArrayRef<SourceRefIndex> getPieces() const {
    return path;
  }
  llvm::ArrayRef<SourceRefIndex> getPath() const { return path; }

  void print(mlir::raw_ostream &os) const;
  void dump() const { print(llvm::errs()); }

  bool operator==(const SourceRef &rhs) const;

  bool operator!=(const SourceRef &rhs) const { return !(*this == rhs); }

  // required for EquivalenceClasses usage
  std::strong_ordering operator<=>(const SourceRef &rhs) const;

  struct Hash {
    size_t operator()(const SourceRef &val) const;
  };

  friend struct llvm::DenseMapInfo<SourceRef>;

private:
  mlir::Value value;
  Path path;
  bool constant;
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
      return llvm::hash_value(ref.getAsOpaquePointer());
    }
    return llzk::SourceRef::Hash {}(ref);
  }
  static bool isEqual(const llzk::SourceRef &lhs, const llzk::SourceRef &rhs) { return lhs == rhs; }
};

} // namespace llvm
