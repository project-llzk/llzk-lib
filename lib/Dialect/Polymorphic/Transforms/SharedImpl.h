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
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
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

#include <tuple>

#define DEBUG_TYPE "poly-dialect-shared"

namespace llzk::polymorphic::detail {

namespace {

/// Lists all Op classes that may contain a StructType in their results or attributes.
static struct OpClassesWithStructTypes {

  /// Subset that define the general builder function:
  /// `build(OpBuilder&, OperationState&, TypeRange, ValueRange, ArrayRef<NamedAttribute>)`
  const std::tuple<
      // clang-format off
      llzk::array::ArrayLengthOp,
      llzk::array::ReadArrayOp,
      llzk::array::WriteArrayOp,
      llzk::array::InsertArrayOp,
      llzk::array::ExtractArrayOp,
      llzk::constrain::EmitEqualityOp,
      llzk::constrain::EmitContainmentOp,
      llzk::component::MemberDefOp,
      llzk::component::MemberReadOp,
      llzk::component::MemberWriteOp,
      llzk::component::CreateStructOp,
      llzk::function::FuncDefOp,
      llzk::function::ReturnOp,
      llzk::global::GlobalDefOp,
      llzk::global::GlobalReadOp,
      llzk::global::GlobalWriteOp,
      llzk::polymorphic::UnifiableCastOp,
      llzk::polymorphic::ConstReadOp
      // clang-format on
      >
      WithGeneralBuilder {};

  /// Subset that do NOT define the general builder function. These cannot use
  /// `GeneralTypeReplacePattern` and must have a `OpConversionPattern` defined if they need
  /// to be converted. There is a default `OpConversionPattern` defined for each of these if
  /// using `newGeneralRewritePatternSet()`.
  const std::tuple<llzk::function::CallOp, llzk::array::CreateArrayOp> NoGeneralBuilder {};

} OpClassesWithStructTypes;

template <typename I, typename NextOpClass, typename... OtherOpClasses>
inline void applyToMoreTypes(I inserter) {
  std::apply(inserter, std::tuple<NextOpClass, OtherOpClasses...> {});
}
template <typename I> inline void applyToMoreTypes(I inserter) {}

inline bool defaultLegalityCheck(const mlir::TypeConverter &tyConv, mlir::Operation *op) {
  // Check operand types and result types
  if (!tyConv.isLegal(op)) {
    return false;
  }
  // Check type attributes
  // Extend lifetime of temporary to suppress warnings.
  mlir::DictionaryAttr dictAttr = op->getAttrDictionary();
  for (mlir::NamedAttribute n : dictAttr.getValue()) {
    if (mlir::TypeAttr tyAttr = llvm::dyn_cast<mlir::TypeAttr>(n.getValue())) {
      mlir::Type t = tyAttr.getValue();
      if (mlir::FunctionType funcTy = llvm::dyn_cast<mlir::FunctionType>(t)) {
        if (!tyConv.isSignatureLegal(funcTy)) {
          return false;
        }
      } else {
        if (!tyConv.isLegal(t)) {
          return false;
        }
      }
    }
  }
  return true;
}

// Default to true if the check is not for that particular operation type.
template <typename Check> inline bool runCheck(mlir::Operation *op, Check check) {
  if (auto specificOp =
          llvm::dyn_cast_if_present<typename llvm::function_traits<Check>::template arg_t<0>>(op)) {
    return check(specificOp);
  }
  return true;
}

} // namespace

/// Wrapper for PatternRewriter.replaceOpWithNewOp() that automatically copies discardable
/// attributes (i.e., attributes other than those specifically defined as part of the Op in ODS).
template <typename OpClass, typename Rewriter, typename... Args>
inline OpClass replaceOpWithNewOp(Rewriter &rewriter, mlir::Operation *op, Args &&...args) {
  mlir::DictionaryAttr attrs = op->getDiscardableAttrDictionary();
  OpClass newOp = rewriter.template replaceOpWithNewOp<OpClass>(op, std::forward<Args>(args)...);
  newOp->setDiscardableAttrs(attrs);
  return newOp;
}

// NOTE: This pattern will produce a compile error if `OpClass` does not define the general
// `build(OpBuilder&, OperationState&, TypeRange, ValueRange, ArrayRef<NamedAttribute>)` function
// because that function is required by the `replaceOpWithNewOp()` call.
template <typename OpClass>
class GeneralTypeReplacePattern : public mlir::OpConversionPattern<OpClass> {
public:
  using mlir::OpConversionPattern<OpClass>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      OpClass op, OpClass::Adaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override {
    const mlir::TypeConverter *converter = mlir::OpConversionPattern<OpClass>::getTypeConverter();
    assert(converter);
    // Convert result types
    mlir::SmallVector<mlir::Type> newResultTypes;
    if (mlir::failed(converter->convertTypes(op->getResultTypes(), newResultTypes))) {
      return op->emitError("Could not convert Op result types.");
    }
    // ASSERT: 'adaptor.getAttributes()' is empty or subset of 'op->getAttrDictionary()' so the
    // former can be ignored without losing anything.
    assert(
        adaptor.getAttributes().empty() ||
        llvm::all_of(
            adaptor.getAttributes(), [d = op->getAttrDictionary()](mlir::NamedAttribute a) {
      return d.contains(a.getName());
    }
        )
    );
    // Convert any TypeAttr in the attribute list.
    mlir::SmallVector<mlir::NamedAttribute> newAttrs(op->getAttrDictionary().getValue());
    for (mlir::NamedAttribute &n : newAttrs) {
      if (mlir::TypeAttr t = llvm::dyn_cast<mlir::TypeAttr>(n.getValue())) {
        if (mlir::Type newType = converter->convertType(t.getValue())) {
          n.setValue(mlir::TypeAttr::get(newType));
        } else {
          return op->emitError().append("Could not convert type in attribute: ", t);
        }
      }
    }
    // Build a new Op in place of the current one
    replaceOpWithNewOp<OpClass>(
        rewriter, op, mlir::TypeRange(newResultTypes), adaptor.getOperands(),
        mlir::ArrayRef(newAttrs)
    );
    return mlir::success();
  }
};

class CreateArrayOpClassReplacePattern
    : public mlir::OpConversionPattern<llzk::array::CreateArrayOp> {
public:
  using mlir::OpConversionPattern<llzk::array::CreateArrayOp>::OpConversionPattern;

  mlir::LogicalResult match(llzk::array::CreateArrayOp op) const override {
    if (mlir::Type newType = getTypeConverter()->convertType(op.getType())) {
      return mlir::success();
    } else {
      return op->emitError("Could not convert Op result type.");
    }
  }

  void rewrite(
      llzk::array::CreateArrayOp op, OpAdaptor adapter, mlir::ConversionPatternRewriter &rewriter
  ) const override {
    mlir::Type newType = getTypeConverter()->convertType(op.getType());
    assert(
        llvm::isa<llzk::array::ArrayType>(newType) && "CreateArrayOp must produce ArrayType result"
    );
    mlir::DenseI32ArrayAttr numDimsPerMap = op.getNumDimsPerMapAttr();
    if (isNullOrEmpty(numDimsPerMap)) {
      replaceOpWithNewOp<llzk::array::CreateArrayOp>(
          rewriter, op, llvm::cast<llzk::array::ArrayType>(newType), adapter.getElements()
      );
    } else {
      replaceOpWithNewOp<llzk::array::CreateArrayOp>(
          rewriter, op, llvm::cast<llzk::array::ArrayType>(newType), adapter.getMapOperands(),
          numDimsPerMap
      );
    }
  }
};

class CallOpClassReplacePattern : public mlir::OpConversionPattern<llzk::function::CallOp> {
public:
  using mlir::OpConversionPattern<llzk::function::CallOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      llzk::function::CallOp op, OpAdaptor adapter, mlir::ConversionPatternRewriter &rewriter
  ) const override {
    // Convert the result types of the CallOp
    mlir::SmallVector<mlir::Type> newResultTypes;
    if (mlir::failed(getTypeConverter()->convertTypes(op.getResultTypes(), newResultTypes))) {
      return op->emitError("Could not convert Op result types.");
    }
    replaceOpWithNewOp<llzk::function::CallOp>(
        rewriter, op, newResultTypes, op.getCalleeAttr(), adapter.getMapOperands(),
        op.getNumDimsPerMapAttr(), adapter.getArgOperands()
    );
    return mlir::success();
  }
};

/// Return a new `RewritePatternSet` that includes a `GeneralTypeReplacePattern` for all of
/// `OpClassesWithStructTypes.WithGeneralBuilder` and `AdditionalOpClasses`.
/// Note: `GeneralTypeReplacePattern` uses the default benefit (1) so additional patterns with a
/// higher priority can be added for any of the Ops already included and that will take precedence.
template <typename... AdditionalOpClasses>
mlir::RewritePatternSet newGeneralRewritePatternSet(
    mlir::TypeConverter &tyConv, mlir::MLIRContext *ctx, mlir::ConversionTarget &target
) {
  mlir::RewritePatternSet patterns(ctx);
  auto inserter = [&](auto... opClasses) {
    patterns.add<GeneralTypeReplacePattern<decltype(opClasses)>...>(tyConv, ctx);
  };
  std::apply(inserter, OpClassesWithStructTypes.WithGeneralBuilder);
  applyToMoreTypes<decltype(inserter), AdditionalOpClasses...>(inserter);
  // Special cases for ops where GeneralTypeReplacePattern doesn't work
  patterns.add<CreateArrayOpClassReplacePattern, CallOpClassReplacePattern>(tyConv, ctx);
  // Add builtin FunctionType converter
  mlir::populateFunctionOpInterfaceTypeConversionPattern<llzk::function::FuncDefOp>(
      patterns, tyConv
  );
  mlir::scf::populateSCFStructuralTypeConversionsAndLegality(tyConv, patterns, target);
  return patterns;
}

/// Return a new `ConversionTarget` allowing all LLZK-required dialects.
mlir::ConversionTarget newBaseTarget(mlir::MLIRContext *ctx);

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
  mlir::ConversionTarget target = newBaseTarget(ctx);
  auto inserter = [&](auto... opClasses) {
    target.addDynamicallyLegalOp<decltype(opClasses)...>([&tyConv,
                                                          &checks...](mlir::Operation *op) {
      LLVM_DEBUG(if (op) {
        llvm::dbgs() << "[newConverterDefinedTarget] checking legality of ";
        op->dump();
      });
      auto legality =
          defaultLegalityCheck(tyConv, op) && (runCheck<AdditionalChecks>(op, checks) && ...);

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
