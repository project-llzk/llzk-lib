//===-- Ops.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Function/IR/OpTraits.h"
#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/Shared/OpHelpers.h"
#include "llzk/Dialect/Struct/IR/Types.h"

namespace llzk {

namespace component {

/// Only valid/implemented for StructDefOp. Sets the proper `AllowConstraintAttr` and
/// `AllowWitnessAttr` on the functions defined within the StructDefOp.
template <typename TypeClass>
// Suppress false positive from `clang-tidy`
// NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility)
class SetFuncAllowAttrs : public mlir::OpTrait::TraitBase<TypeClass, SetFuncAllowAttrs> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op);
};

} // namespace component

namespace function {
class FuncDefOp;
} // namespace function

} // namespace llzk

// Include TableGen'd declarations
#include "llzk/Dialect/Struct/IR/OpInterfaces.h.inc"

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "llzk/Dialect/Struct/IR/Ops.h.inc"

namespace llzk::component {

mlir::InFlightDiagnostic
genCompareErr(StructDefOp expected, mlir::Operation *origin, const char *aspect);

mlir::LogicalResult checkSelfType(
    mlir::SymbolTableCollection &symbolTable, StructDefOp expectedStruct, mlir::Type actualType,
    mlir::Operation *origin, const char *aspect
);

/// Return true iff the given Operation is nested somewhere within a StructDefOp.
bool isInStruct(mlir::Operation *op);

/// If the given Operation is nested somewhere within a StructDefOp, return a success result
/// containing that StructDefOp. Otherwise emit an error and return a failure result.
mlir::FailureOr<StructDefOp> verifyInStruct(mlir::Operation *op);

/// Return true iff the given Operation is contained within a FuncDefOp with the given name that is
/// itself contained within a StructDefOp.
bool isInStructFunctionNamed(mlir::Operation *op, char const *funcName);

/// Checks if the given Operation is contained within a FuncDefOp with the given name that is itself
/// contained within a StructDefOp, producing an error if not.
template <char const *FuncName, unsigned PrefixLen>
mlir::LogicalResult verifyInStructFunctionNamed(
    mlir::Operation *op, llvm::function_ref<llvm::SmallString<PrefixLen>()> prefix
) {
  return isInStructFunctionNamed(op, FuncName)
             ? mlir::success()
             : op->emitOpError(prefix())
                   << "only valid within a '" << getOperationName<function::FuncDefOp>()
                   << "' named \"@" << FuncName << "\" within a '"
                   << getOperationName<StructDefOp>() << "' definition";
}

/// This class provides a verifier for ops that are expecting to have an ancestor FuncDefOp with the
/// given name.
template <char const *FuncName> struct InStructFunctionNamed {
  template <typename TypeClass> class Impl : public mlir::OpTrait::TraitBase<TypeClass, Impl> {
  public:
    static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
      return verifyInStructFunctionNamed<FuncName, 0>(op, [] { return llvm::SmallString<0>(); });
    }
  };
};

} // namespace llzk::component
