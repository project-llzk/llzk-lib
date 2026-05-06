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

#include <mlir/IR/PatternMatch.h>

#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/Twine.h>

namespace llzk {

/// Return the `function.arg_name` attribute from the given argument attribute dictionary.
inline mlir::StringAttr getFunctionArgNameAttr(mlir::DictionaryAttr attrs) {
  if (!attrs) {
    return {};
  }
  return llvm::dyn_cast_or_null<mlir::StringAttr>(attrs.get(function::ARG_NAME_ATTR_NAME));
}

/// Return a copy of the given argument attribute dictionary without `function.arg_name`.
inline mlir::DictionaryAttr removeFunctionArgNameAttr(mlir::DictionaryAttr attrs) {
  if (!attrs || !attrs.contains(function::ARG_NAME_ATTR_NAME)) {
    return attrs;
  }
  llvm::SmallVector<mlir::NamedAttribute> newAttrs;
  for (mlir::NamedAttribute attr : attrs) {
    if (attr.getName().getValue() != function::ARG_NAME_ATTR_NAME) {
      newAttrs.push_back(attr);
    }
  }
  return mlir::DictionaryAttr::get(attrs.getContext(), newAttrs);
}

/// Return a copy of the given argument attribute dictionary with `function.arg_name` set to `name`.
inline mlir::DictionaryAttr
setFunctionArgNameAttr(mlir::DictionaryAttr attrs, llvm::StringRef name) {
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

/// Add all valid `function.arg_name` values in `argAttrs` to `usedNames`.
inline void collectFunctionArgNames(mlir::ArrayAttr argAttrs, llvm::StringSet<> &usedNames) {
  if (!argAttrs) {
    return;
  }
  for (mlir::Attribute attr : argAttrs) {
    auto dictAttr = llvm::dyn_cast<mlir::DictionaryAttr>(attr);
    if (mlir::StringAttr argName = getFunctionArgNameAttr(dictAttr)) {
      usedNames.insert(argName.getValue());
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

} // namespace llzk
