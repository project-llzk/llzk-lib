//===-- SharedImpl.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Common private implementation for poly dialect passes.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Shared/TypeConversionPatterns.h"
#include "llzk/Dialect/Struct/IR/Ops.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "poly-dialect-shared"

namespace llzk::polymorphic::detail {

namespace {

// Default to true if the check is not for that particular operation type.
template <typename Check> inline bool runCheck(mlir::Operation *op, Check check) {
  if (auto specificOp =
          llvm::dyn_cast_if_present<typename llvm::function_traits<Check>::template arg_t<0>>(op)) {
    return check(specificOp);
  }
  return true;
}

} // namespace

/// Return a new `ConversionTarget` allowing all LLZK-required dialects.
mlir::ConversionTarget newBaseTarget(mlir::MLIRContext *ctx);

/// Merge nested array dimensions produced by replacing an array element type.
///
/// If `array<4 x !poly.tvar<@T>>` is rewritten with `@T -> array<8 x index>`, the canonical
/// aggregate shape should become `array<4,8 x index>` rather than an array whose element type
/// is another array because the latter is not allowed in LLZK IR.
array::ArrayType flattenInstantiatedArrayType(array::ArrayType inputTy, mlir::Type convertedElemTy);

/// Build a struct type while representing an empty parameter list as absent.
inline component::StructType
getStructTypeWithParams(mlir::SymbolRefAttr nameRef, mlir::ArrayAttr params) {
  return params && !params.empty() ? component::StructType::get(nameRef, params)
                                   : component::StructType::get(nameRef);
}

/// Build a struct type while representing an empty parameter list as absent.
inline component::StructType getStructTypeWithParams(
    mlir::SymbolRefAttr nameRef, mlir::MLIRContext *ctx, mlir::ArrayRef<mlir::Attribute> params
) {
  return params.empty() ? component::StructType::get(nameRef)
                        : component::StructType::get(nameRef, mlir::ArrayAttr::get(ctx, params));
}

class LegalityCheckCallback {
public:
  virtual ~LegalityCheckCallback() = default;
  virtual void checkStarted() = 0;
  virtual void checkEnded(bool outcome) = 0;
};

class EmptyLegalityCheckCallback : public LegalityCheckCallback {
public:
  void checkStarted() override {}
  void checkEnded(bool) override {}
};

/// Return a new `ConversionTarget` allowing all LLZK-required dialects and defining Op legality
/// based on the given `TypeConverter` for Ops listed in both members of `OpClassesWithStructTypes`
/// and in `AdditionalOpClasses`.
/// Additional legality checks can be included for certain ops that will run along with the default
/// check. For an op to be considered legal all checks (default plus additional checks if any) must
/// return true.
template <typename... AdditionalOpClasses, typename... AdditionalChecks>
mlir::ConversionTarget newConverterDefinedTarget(
    mlir::TypeConverter &tyConv, mlir::MLIRContext *ctx, AdditionalChecks &&...checks
) {
  static EmptyLegalityCheckCallback empty;
  return newConverterDefinedTargetWithCallback<AdditionalOpClasses...>(
      tyConv, ctx, empty, (std::forward<AdditionalChecks>(checks))...
  );
}

/// Return a new `ConversionTarget` allowing all LLZK-required dialects and defining Op legality
/// based on the given `TypeConverter` for Ops listed in both members of `OpClassesWithStructTypes`
/// and in `AdditionalOpClasses`.
/// Additional legality checks can be included for certain ops that will run along with the default
/// check. For an op to be considered legal all checks (default plus additional checks if any) must
/// return true.
template <typename... AdditionalOpClasses, typename... AdditionalChecks>
mlir::ConversionTarget newConverterDefinedTargetWithCallback(
    mlir::TypeConverter &tyConv, mlir::MLIRContext *ctx, LegalityCheckCallback &cb,
    AdditionalChecks &&...checks
) {
  mlir::ConversionTarget target = newBaseTarget(ctx);
  auto inserter = [&](auto... opClasses) {
    target.addDynamicallyLegalOp<decltype(opClasses)...>([&cb, &tyConv,
                                                          &checks...](mlir::Operation *op) {
      LLVM_DEBUG(if (op) {
        llvm::dbgs() << "[newConverterDefinedTarget] checking legality of ";
        op->dump();
      });
      cb.checkStarted();
      auto legality =
          defaultLegalityCheck(tyConv, op) && (runCheck<AdditionalChecks>(op, checks) && ...);

      cb.checkEnded(legality);
      LLVM_DEBUG(if (legality) { llvm::dbgs() << "[newConverterDefinedTarget] is legal\n"; } else {
        llvm::dbgs() << "[newConverterDefinedTarget] is not legal\n";
      });
      return legality;
    });
  };
  std::apply(inserter, OpClassesWithStructTypes.NoGeneralBuilder);
  std::apply(inserter, OpClassesWithStructTypes.WithGeneralBuilder);
  applyToMoreTypes<decltype(inserter), AdditionalOpClasses...>(inserter);
  return target;
}

} // namespace llzk::polymorphic::detail

#undef DEBUG_TYPE
