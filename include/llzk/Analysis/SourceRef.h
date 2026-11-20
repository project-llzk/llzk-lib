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
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/EquivalenceClasses.h>
#include <llvm/ADT/TypeSwitch.h>

#include <compare>
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
  explicit SourceRefIndex(int64_t i) : index(llvm::DynamicAPInt(i)) {}
  /// Construct a half-open range; a high bound of ShapedType::kDynamic means unbounded.
  SourceRefIndex(const llvm::DynamicAPInt &low, const llvm::DynamicAPInt &high)
      : index(IndexRange {low, high}) {}
  explicit SourceRefIndex(IndexRange r) : index(r) {}

  /// Select every nonnegative index in a static or dynamic array dimension.
  static SourceRefIndex forArrayDimension(int64_t size);

  /// Whether this range has no finite upper bound.
  bool hasUnboundedUpperBound() const;

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

  /// Return true when these path components select any common storage.
  /// Index ranges are half-open.
  bool overlaps(const SourceRefIndex &rhs) const;

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
  // 99.5% have a path length of 0 or 1, so we use a small vector to avoid heap allocations.
  using Path = llvm::SmallVector<SourceRefIndex, 1>;

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
  /// Numeric identity key: field representatives for specified fields, raw storage otherwise.
  /// Only valid for constant felt or index references; not a fieldless semantic evaluation.
  llvm::DynamicAPInt getConstantIdentityValue() const;
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

  /// Freeze the component currently denoted by this path independently of its mutable slot.
  /// Root SSA handles already have stable identities and are returned unchanged.
  SourceRef captureComponent() const;

  /// Return the underlying storage path for call-entry binding or structural summaries.
  SourceRef withoutComponentCapture() const {
    SourceRef copy = *this;
    copy.capturedComponentDepth = 0;
    return copy;
  }

  /// Whether replacing this possibly ranged prefix would retarget a captured component.
  bool protectsCapturedComponent(const SourceRef &prefix) const;

  /// Return this origin in a particular invocation, retaining its type and path.
  /// Constants remain shared. Repeated call sites truncate recursive contexts so
  /// recursive summaries have a finite set of identities.
  SourceRef inCallContext(mlir::Operation *call) const;

  /// Calls enclosing this origin, ordered from the innermost to the outermost.
  llvm::ArrayRef<mlir::Operation *> getCallContext() const { return callContext; }

  bool isConstantFelt() const {
    return isConstant() && llvm::isa_and_present<felt::FeltConstantOp>(value.getDefiningOp());
  }
  bool isConstantIndex() const {
    return isConstant() &&
           llvm::isa_and_present<mlir::arith::ConstantIndexOp>(value.getDefiningOp());
  }
  /// Return whether this reference originates from a template constant read.
  bool isTemplateConstant() const {
    return isConstant() && llvm::isa_and_present<polymorphic::ConstReadOp>(value.getDefiningOp());
  }
  /// Return whether this reference originates from a read of an immutable global.
  bool isImmutableGlobal() const;

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
  /// Original IR root; a call context further qualifies this reference's identity.
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

  /// Return the field representative, or failure for a nonconstant or unspecified field.
  mlir::FailureOr<llvm::DynamicAPInt> getConstantFeltValue() const {
    auto feltConst = getDefiningOp<felt::FeltConstantOp>();
    if (succeeded(feltConst)) {
      if (auto reduced = feltConst->getValue().getReducedValue()) {
        return *reduced;
      }
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
  /// Reduce a felt constant's declared-field representative into the supplied analysis field.
  /// Fieldless constants use their raw literals instead. Index constants retain integer semantics.
  mlir::FailureOr<llvm::DynamicAPInt> getConstantValue(const Field &field) const;

  /// Return a known numeric value, failing for fieldless felt constants.
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

  /// Return true when both references select overlapping storage at the same path depth.
  bool overlaps(const SourceRef &rhs) const;

  /// Return a copy with ranged array indices narrowed by concrete indices from `rhs`.
  ///
  /// Array indices are paired by dimension order; member and POD-record path components are
  /// ignored when pairing dimensions. A range is replaced only when the corresponding component
  /// in `rhs` is a concrete index contained in that range. Other path components remain unchanged.
  /// For example, narrowing `%self.out[0:4].values[0:8]` with `%arg0[2][5]` produces
  /// `%self.out[2].values[5]`.
  SourceRef narrowRanges(const SourceRef &rhs) const;

  /// @brief If `prefix` is a valid prefix of this reference, return a view of the suffix that
  /// remains after removing the prefix. I.e., `this` = `prefix` + `suffix`.
  ///
  /// The returned view is valid while this reference remains unmodified.
  mlir::FailureOr<llvm::ArrayRef<SourceRefIndex>> getSuffix(const SourceRef &prefix) const;

  /// @brief Return a copy of `prefix` with `suffix` appended when it is rooted.
  ///
  /// Constant references cannot have paths, so their copy is returned unchanged.
  static SourceRef appendSuffix(const SourceRef &prefix, llvm::ArrayRef<SourceRefIndex> suffix);

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
    if (copy.capturedComponentDepth > copy.path.size()) {
      copy.capturedComponentDepth = 0;
      auto ancestor = copy;
      while (!ancestor.path.empty()) {
        if (llvm::isa<component::StructType>(ancestor.getType())) {
          copy.capturedComponentDepth = ancestor.path.size();
          break;
        }
        ancestor.path.pop_back();
      }
    }
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

  /// Print this reference using source-style names. The entry self argument of a
  /// struct constrain function is printed as `%self`.
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
  /// SSA value backing the reference: a root block argument or operation result,
  /// or the result of a constant operation when `constant` is true.
  mlir::Value value;

  /// Member, POD-record, and array selections applied to `value`, from root to leaf.
  /// Empty for constants and references to the entire root value.
  Path path;

  /// Depth of the innermost captured component. Replacing its slot or an ancestor slot must
  /// not retarget this reference; accesses to members below this boundary remain observable.
  /// Zero denotes an ordinary storage path or an immutable root SSA handle.
  size_t capturedComponentDepth = 0;

  /// Enclosing call sites, ordered from innermost to outermost, that qualify the
  /// root's identity. Empty for constants; repeated sites are truncated by inCallContext().
  llvm::SmallVector<mlir::Operation *, 0> callContext;

  /// Whether `value` represents a felt, index, or template constant rather than a root.
  /// Constant references have no path or call context.
  bool constant;
};

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const SourceRef &rhs);

/* SourceRefSet */

class SourceRefSet : public llvm::DenseSet<SourceRef> {
  using Base = llvm::DenseSet<SourceRef>;

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
  static unsigned getHashValue(const llzk::SourceRef &ref) { return llzk::SourceRef::Hash {}(ref); }
  static bool isEqual(const llzk::SourceRef &lhs, const llzk::SourceRef &rhs) { return lhs == rhs; }
};

} // namespace llvm
