//===-- LLZKConversionUtils.h -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Shared utilities for dialect converting transformations.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/Concepts.h"

#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/Twine.h>

namespace llzk {

/// Return a copy of the given argument attribute dictionary with `function.arg_name` set to `name`.
inline mlir::DictionaryAttr
withFunctionArgNameAttr(mlir::DictionaryAttr attrs, llvm::StringRef name) {
  mlir::NamedAttrList newAttrs(attrs);
  newAttrs.set(function::ARG_NAME_ATTR_NAME, mlir::StringAttr::get(attrs.getContext(), name));
  return newAttrs.getDictionary(attrs.getContext());
}

/// Reserve and return a unique function argument name based on `desiredName`.
inline std::string
reserveUniqueFunctionArgName(llvm::StringSet<> &usedNames, llvm::StringRef desiredName) {
  if (!usedNames.contains(desiredName)) {
    usedNames.insert(desiredName);
    return desiredName.str();
  }

  for (unsigned suffix = 1;; ++suffix) {
    std::string candidate = (desiredName + "#" + llvm::Twine(suffix)).str();
    if (!usedNames.contains(candidate)) {
      usedNames.insert(candidate);
      return candidate;
    }
  }
}

/// General helper for converting a `FuncDefOp` by changing its input and/or result types and the
/// associated attributes for those types.
class FunctionTypeConverter {

protected:
  virtual llvm::SmallVector<mlir::Type> convertInputs(mlir::ArrayRef<mlir::Type> origTypes) = 0;
  virtual llvm::SmallVector<mlir::Type> convertResults(mlir::ArrayRef<mlir::Type> origTypes) = 0;

  virtual mlir::ArrayAttr
  convertInputAttrs(mlir::ArrayAttr origAttrs, llvm::SmallVector<mlir::Type> newTypes) = 0;
  virtual mlir::ArrayAttr
  convertResultAttrs(mlir::ArrayAttr origAttrs, llvm::SmallVector<mlir::Type> newTypes) = 0;

  virtual void processBlockArgs(mlir::Block &entryBlock, mlir::RewriterBase &rewriter) = 0;

public:
  virtual ~FunctionTypeConverter() = default;

  void convert(function::FuncDefOp op, mlir::RewriterBase &rewriter) {
    // Update in/out types of the function
    mlir::FunctionType oldTy = op.getFunctionType();
    llvm::SmallVector<mlir::Type> newInputs = convertInputs(oldTy.getInputs());
    llvm::SmallVector<mlir::Type> newResults = convertResults(oldTy.getResults());
    mlir::FunctionType newTy = mlir::FunctionType::get(
        oldTy.getContext(), mlir::TypeRange(newInputs), mlir::TypeRange(newResults)
    );
    if (newTy == oldTy) {
      return; // nothing to change
    }

    // Pre-condition: arg/result count equals corresponding attribute count
    assert(!op.getResAttrsAttr() || op.getResAttrsAttr().size() == op.getNumResults());
    assert(!op.getArgAttrsAttr() || op.getArgAttrsAttr().size() == op.getNumArguments());
    rewriter.modifyOpInPlace(op, [&]() {
      op.setFunctionType(newTy);

      // If any input or result types were added, ensure the attributes are updated too.
      if (mlir::ArrayAttr newArgAttrs = convertInputAttrs(op.getArgAttrsAttr(), newInputs)) {
        op.setArgAttrsAttr(newArgAttrs);
      }
      if (mlir::ArrayAttr newResAttrs = convertResultAttrs(op.getResAttrsAttr(), newResults)) {
        op.setResAttrsAttr(newResAttrs);
      }
    });
    // Post-condition: arg/result count equals corresponding attribute count
    assert(!op.getResAttrsAttr() || op.getResAttrsAttr().size() == op.getNumResults());
    assert(!op.getArgAttrsAttr() || op.getArgAttrsAttr().size() == op.getNumArguments());

    // If the function has a body, ensure the entry block arguments match the function inputs.
    if (mlir::Region *body = op.getCallableRegion()) {
      mlir::Block &entryBlock = body->front();
      if (!std::cmp_equal(entryBlock.getNumArguments(), newInputs.size())) {
        processBlockArgs(entryBlock, rewriter);
        // Post-condition: block args must match function inputs
        assert(std::cmp_equal(entryBlock.getNumArguments(), newInputs.size()));
      }
    }
  }
};

/// Common implementation for handling `MemberWriteOp` and `MemberReadOp` while destructuring
/// an aggregate type (e.g., ArrayType or PodType) stored in a struct member.
///
/// @tparam ImplClass         the concrete subclass (CRTP)
/// @tparam MemberRefOpClass  the concrete op class (must implement `MemberRefOpInterface`)
/// @tparam GenHeaderType     return type of `genHeader()`, used to pass data to `forId()`
/// @tparam IdType            the type used to identify a scalar element within the aggregate
template <
    typename ImplClass, HasInterface<component::MemberRefOpInterface> MemberRefOpClass,
    typename GenHeaderType, typename IdType>
class SplitAggregateInMemberRefOp : public mlir::OpConversionPattern<MemberRefOpClass> {
public:
  /// Scalar member name and type.
  using MemberInfo = std::pair<mlir::StringAttr, mlir::Type>;
  /// Maps a scalar element identifier within the aggregate to its new scalar member info.
  using LocalMemberReplacementMap = llvm::DenseMap<IdType, MemberInfo>;
  /// Maps struct -> original aggregate-type member name -> LocalMemberReplacementMap.
  using MemberReplacementMap = llvm::DenseMap<
      component::StructDefOp, llvm::DenseMap<mlir::StringAttr, LocalMemberReplacementMap>>;

private:
  mlir::SymbolTableCollection &tables;
  const MemberReplacementMap &repMapRef;

  // Static check to ensure the methods are implemented in all subclasses.
  inline static void ensureImplementedAtCompile() {
    static_assert(
        sizeof(MemberRefOpClass) == 0,
        "SplitAggregateInMemberRefOp not implemented for requested type."
    );
  }

protected:
  using OpAdaptor = typename MemberRefOpClass::Adaptor;

  /// Executed at the start of `rewrite()` to (optionally) generate anything that should appear
  /// before the per-scalar operations that will be added by `forId()`.
  static GenHeaderType genHeader(MemberRefOpClass, mlir::ConversionPatternRewriter &) {
    ensureImplementedAtCompile();
    llvm_unreachable("must have concrete instantiation");
  }

  /// Executed for each scalar id in the aggregate type of the original member to generate the
  /// per-scalar operations on the new scalar members.
  static void forId(
      mlir::Location, GenHeaderType &, IdType, MemberInfo, OpAdaptor,
      mlir::ConversionPatternRewriter &
  ) {
    ensureImplementedAtCompile();
    llvm_unreachable("must have concrete instantiation");
  }

public:
  // Suppress false positive from `clang-tidy`
  // NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility)
  SplitAggregateInMemberRefOp(
      mlir::MLIRContext *ctx, mlir::SymbolTableCollection &symTables,
      const MemberReplacementMap &memberRepMap
  )
      : mlir::OpConversionPattern<MemberRefOpClass>(ctx), tables(symTables),
        repMapRef(memberRepMap) {}

  static bool legal(MemberRefOpClass) {
    ensureImplementedAtCompile();
    llvm_unreachable("must have concrete instantiation");
    return false;
  }

  mlir::LogicalResult match(MemberRefOpClass op) const override {
    return mlir::failure(ImplClass::legal(op));
  }

  void rewrite(
      MemberRefOpClass op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override {
    component::StructType tgtStructTy =
        llvm::cast<component::MemberRefOpInterface>(op.getOperation()).getStructType();
    assert(tgtStructTy);
    auto tgtStructDef = tgtStructTy.getDefinition(tables, op);
    assert(mlir::succeeded(tgtStructDef));

    GenHeaderType prefixResult = ImplClass::genHeader(op, rewriter);

    const LocalMemberReplacementMap &idToName =
        repMapRef.at(tgtStructDef->get()).at(op.getMemberNameAttr().getAttr());
    // Split the aggregate member into a series of scalar member ops.
    for (auto [id, newMember] : idToName) {
      ImplClass::forId(op.getLoc(), prefixResult, id, newMember, adaptor, rewriter);
    }
    if constexpr (requires { ImplClass::finalize(op, prefixResult, adaptor, rewriter); }) {
      ImplClass::finalize(op, prefixResult, adaptor, rewriter);
    }
    rewriter.eraseOp(op);
  }
};

} // namespace llzk
