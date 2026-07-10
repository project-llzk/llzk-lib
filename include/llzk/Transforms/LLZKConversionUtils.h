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

#include <optional>
#include <string>

namespace llzk {

/// Return a copy of the given function argument/result attribute dictionary with `attrName` set
/// to `name`.
inline mlir::DictionaryAttr
withFunctionNameAttr(mlir::DictionaryAttr attrs, llvm::StringRef attrName, llvm::StringRef name) {
  mlir::NamedAttrList newAttrs(attrs);
  newAttrs.set(attrName, mlir::StringAttr::get(attrs.getContext(), name));
  return newAttrs.getDictionary(attrs.getContext());
}

/// Return a copy of the given argument attribute dictionary with `function.arg_name` set to
/// `name`.
inline mlir::DictionaryAttr
withFunctionArgNameAttr(mlir::DictionaryAttr attrs, llvm::StringRef name) {
  return withFunctionNameAttr(attrs, function::ARG_NAME_ATTR_NAME, name);
}

/// Return a copy of the given result attribute dictionary with `function.res_name` set to `name`.
inline mlir::DictionaryAttr
withFunctionResNameAttr(mlir::DictionaryAttr attrs, llvm::StringRef name) {
  return withFunctionNameAttr(attrs, function::RES_NAME_ATTR_NAME, name);
}

/// Reserve and return a unique function argument/result name based on `desiredName`.
inline std::string
reserveUniqueAttrName(llvm::StringSet<> &usedNames, llvm::StringRef desiredName) {
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

/// Return the function arg/result attribute at `index` for the given name, if present.
inline std::optional<mlir::StringAttr>
getAttrAtIndexWithName(mlir::ArrayAttr attrs, unsigned index, llvm::StringRef attrName) {
  if (!attrs || index >= attrs.size()) {
    return std::nullopt;
  }
  if (auto dictAttr = llvm::dyn_cast<mlir::DictionaryAttr>(attrs[index])) {
    mlir::Attribute attr = dictAttr.get(attrName);
    if (!attr) {
      return std::nullopt;
    }
    if (auto nameAttr = llvm::dyn_cast<mlir::StringAttr>(attr)) {
      return nameAttr;
    }
  }
  return std::nullopt;
}

/// Cached function arg/result names and split suffixes used while rewriting a function signature.
struct SplitFunctionNameInfo {
  llvm::SmallVector<std::optional<llvm::StringRef>> originalNames;
  llvm::SmallVector<llvm::StringRef> existingNames;
  llvm::SmallVector<llvm::SmallVector<std::string>> splitNameSuffixes;
};

/// Collect function arg/result names and split suffixes from a list of original types.
template <typename GetNameAttrFn, typename GetSplitSuffixesFn>
inline SplitFunctionNameInfo collectSplitFunctionNameInfo(
    mlir::ArrayRef<mlir::Type> origTypes, GetNameAttrFn &&getNameAttr,
    GetSplitSuffixesFn &&getSplitSuffixes
) {
  SplitFunctionNameInfo info;
  info.originalNames.reserve(origTypes.size());
  info.splitNameSuffixes.reserve(origTypes.size());
  for (auto [i, type] : llvm::enumerate(origTypes)) {
    if (std::optional<mlir::StringAttr> nameAttr = getNameAttr(i)) {
      info.originalNames.push_back(nameAttr->getValue());
      info.existingNames.push_back(nameAttr->getValue());
    } else {
      info.originalNames.push_back(std::nullopt);
    }
    info.splitNameSuffixes.push_back(getSplitSuffixes(type));
  }
  return info;
}

/// Expand function arg/result attribute arrays to match a split signature, rewriting name attrs
/// with the provided suffixes where available.
inline mlir::ArrayAttr replicateFunctionNameAttrsAsNeeded(
    mlir::ArrayAttr origAttrs, const llvm::SmallVector<size_t> &originalIdxToSize,
    const llvm::SmallVector<mlir::Type> &newTypes, llvm::StringRef functionNameAttrName,
    llvm::ArrayRef<std::optional<llvm::StringRef>> origNames = {},
    llvm::ArrayRef<llvm::StringRef> existingNames = {},
    llvm::ArrayRef<llvm::SmallVector<std::string>> splitNameSuffixes = {}
) {
  if (!origAttrs) {
    return nullptr;
  }
  assert(originalIdxToSize.size() == origAttrs.size());
  if (originalIdxToSize.size() == newTypes.size()) {
    return nullptr;
  }

  llvm::SmallVector<mlir::Attribute> newAttrs;
  llvm::StringSet<> usedNames;
  if (!origNames.empty()) {
    for (llvm::StringRef name : existingNames) {
      usedNames.insert(name);
    }
  }

  for (auto [i, s] : llvm::enumerate(originalIdxToSize)) {
    mlir::Attribute attr = origAttrs[i];
    if (!origNames.empty() && !splitNameSuffixes.empty() && s != 1 && origNames[i]) {
      assert(i < splitNameSuffixes.size());
      assert(splitNameSuffixes[i].size() == s);
      auto dictAttr = llvm::cast<mlir::DictionaryAttr>(attr);
      llvm::StringRef name = *origNames[i];
      for (llvm::StringRef suffix : splitNameSuffixes[i]) {
        std::string desiredName = (llvm::Twine(name) + suffix).str();
        newAttrs.push_back(withFunctionNameAttr(
            dictAttr, functionNameAttrName, reserveUniqueAttrName(usedNames, desiredName)
        ));
      }
      continue;
    }
    newAttrs.append(s, attr);
  }
  return mlir::ArrayAttr::get(origAttrs.getContext(), newAttrs);
}

/// Rebuild a `function.call` while preserving explicit instantiation state from `oldCall`.
///
/// This helper forwards both template parameters and affine-map instantiation operands from the
/// original call while allowing callers to replace the result types and SSA operands that the new
/// call should use.
inline function::CallOp createCallPreservingInstantiationOperands(
    mlir::Location loc, mlir::TypeRange newResultTypes, function::CallOp oldCall,
    llvm::ArrayRef<mlir::ValueRange> mapOperands, mlir::ValueRange argOperands,
    mlir::ConversionPatternRewriter &rewriter
) {
  llvm::SmallVector<mlir::Attribute> templateParams;
  if (mlir::ArrayAttr templateParamsAttr = oldCall.getTemplateParamsAttr()) {
    templateParams.append(templateParamsAttr.begin(), templateParamsAttr.end());
  }

  if (oldCall.getMapOperands().empty()) {
    return rewriter.create<function::CallOp>(
        loc, newResultTypes, oldCall.getCalleeAttr(), argOperands, templateParams
    );
  }

  return rewriter.create<function::CallOp>(
      loc, newResultTypes, oldCall.getCalleeAttr(), mapOperands, oldCall.getNumDimsPerMapAttr(),
      argOperands, templateParams
  );
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
      bool blockArgsNeedUpdate =
          !std::cmp_equal(entryBlock.getNumArguments(), newInputs.size()) ||
          llvm::any_of(llvm::zip_equal(entryBlock.getArgumentTypes(), newInputs), [](auto pair) {
        return std::get<0>(pair) != std::get<1>(pair);
      });
      if (blockArgsNeedUpdate) {
        processBlockArgs(entryBlock, rewriter);
        // Post-condition: block args must match function inputs in both arity and type.
        assert(std::cmp_equal(entryBlock.getNumArguments(), newInputs.size()));
        for (unsigned i = 0, e = entryBlock.getNumArguments(); i < e; ++i) {
          assert(entryBlock.getArgument(i).getType() == newInputs[i]);
        }
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
    for (const auto &[id, newMember] : idToName) {
      ImplClass::forId(op.getLoc(), prefixResult, id, newMember, adaptor, rewriter);
    }
    if constexpr (requires { ImplClass::finalize(op, prefixResult, adaptor, rewriter); }) {
      ImplClass::finalize(op, prefixResult, adaptor, rewriter);
    }
    rewriter.eraseOp(op);
  }
};

} // namespace llzk
