//===-- TypeConversionPatterns.h --------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Reusable MLIR dialect conversion functions for LLZK StructType replacement.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"

#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/SmallVector.h>

#include <tuple>

namespace llzk {

/// Check whether an op is legal with respect to the given type converter, including TypeAttr
/// attributes (with special handling for FunctionType stored inside a TypeAttr).
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

/// Wrapper for `PatternRewriter::replaceOpWithNewOp()` that automatically copies discardable
/// attributes (i.e., attributes other than those specifically defined as part of the op in ODS).
template <typename OpClass, typename Rewriter, typename... Args>
inline OpClass replaceOpWithNewOp(Rewriter &rewriter, mlir::Operation *op, Args &&...args) {
  mlir::DictionaryAttr attrs = op->getDiscardableAttrDictionary();
  OpClass newOp = rewriter.template replaceOpWithNewOp<OpClass>(op, std::forward<Args>(args)...);
  newOp->setDiscardableAttrs(attrs);
  return newOp;
}

/// Lists all LLZK op classes that may contain a StructType in their results or attributes.
static struct OpClassesWithStructTypes {

  /// Ops in this subset define the general builder function:
  /// `build(OpBuilder&, OperationState&, TypeRange, ValueRange, ArrayRef<NamedAttribute>)`
  const std::tuple<
      // clang-format off
      array::ArrayLengthOp,
      array::ReadArrayOp,
      array::WriteArrayOp,
      array::InsertArrayOp,
      array::ExtractArrayOp,
      constrain::EmitEqualityOp,
      constrain::EmitContainmentOp,
      component::MemberDefOp,
      component::MemberReadOp,
      component::MemberWriteOp,
      component::CreateStructOp,
      function::FuncDefOp,
      function::ReturnOp,
      global::GlobalDefOp,
      global::GlobalReadOp,
      global::GlobalWriteOp,
      pod::ReadPodOp,
      pod::WritePodOp,
      polymorphic::UnifiableCastOp,
      polymorphic::ConstReadOp
      // clang-format on
      >
      WithGeneralBuilder {};

  /// Ops in this subset do NOT define the general builder function (see above), so they cannot use
  /// `GeneralTypeReplacePattern`. A custom `OpConversionPattern` is needed to convert these ops.
  ///
  /// The `newGeneralRewritePatternSet()` function provides a default `OpConversionPattern` for
  /// each of these with benefit 0, allowing more specific higher-benefit patterns to override.
  const std::tuple<function::CallOp, array::CreateArrayOp, pod::NewPodOp> NoGeneralBuilder {};

} OpClassesWithStructTypes;

namespace {

/// Pattern for ops that define the general builder:
///   `build(OpBuilder&, OperationState&, TypeRange, ValueRange, ArrayRef<NamedAttribute>)`
/// Converts result types and TypeAttr attributes using the provided TypeConverter.
///
/// NOTE: This pattern will produce a compile error if `OpClass` does not define the general
/// builder function because that function is required by the `replaceOpWithNewOp()` call.
template <typename OpClass>
class GeneralTypeReplacePattern : public mlir::OpConversionPattern<OpClass> {
public:
  /// Defaults to benefit 0 so that any op-specific pattern at the standard
  /// benefit (1), or higher, takes priority over this generic fallback.
  GeneralTypeReplacePattern(mlir::TypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<OpClass>(converter, ctx, 0) {}

  mlir::LogicalResult matchAndRewrite(
      OpClass op, typename OpClass::Adaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override {
    const mlir::TypeConverter *converter = mlir::OpConversionPattern<OpClass>::getTypeConverter();
    assert(converter);
    // Convert result types
    mlir::SmallVector<mlir::Type> newResultTypes;
    if (mlir::failed(converter->convertTypes(op->getResultTypes(), newResultTypes))) {
      return op->emitError("Could not convert Op result types.");
    }
    // ASSERT: 'adaptor.getAttributes()' is empty or a subset of 'op->getAttrDictionary()' so the
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

/// Pattern for `CreateArrayOp`, which lacks the general builder.
class CreateArrayOpClassReplacePattern : public mlir::OpConversionPattern<array::CreateArrayOp> {
public:
  /// Defaults to benefit 0 so that any op-specific pattern at the standard
  /// benefit (1), or higher, takes priority over this generic fallback.
  CreateArrayOpClassReplacePattern(mlir::TypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<array::CreateArrayOp>(converter, ctx, 0) {}

  mlir::LogicalResult match(array::CreateArrayOp op) const override {
    if (getTypeConverter()->convertType(op.getType())) {
      return mlir::success();
    }
    return op->emitError("Could not convert Op result type.");
  }

  void rewrite(
      array::CreateArrayOp op, OpAdaptor adapter, mlir::ConversionPatternRewriter &rewriter
  ) const override {
    mlir::Type newType = getTypeConverter()->convertType(op.getType());
    assert(llvm::isa<array::ArrayType>(newType) && "CreateArrayOp must produce ArrayType result");
    mlir::DenseI32ArrayAttr numDimsPerMap = op.getNumDimsPerMapAttr();
    if (isNullOrEmpty(numDimsPerMap)) {
      replaceOpWithNewOp<array::CreateArrayOp>(
          rewriter, op, llvm::cast<array::ArrayType>(newType), adapter.getElements()
      );
    } else {
      replaceOpWithNewOp<array::CreateArrayOp>(
          rewriter, op, llvm::cast<array::ArrayType>(newType), adapter.getMapOperands(),
          numDimsPerMap
      );
    }
  }
};

/// Pattern for `CallOp`. Converts result types only; the callee symbol is left unchanged.
class CallOpClassReplacePattern : public mlir::OpConversionPattern<function::CallOp> {
public:
  CallOpClassReplacePattern(mlir::TypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<function::CallOp>(converter, ctx, 0) {}

  mlir::LogicalResult matchAndRewrite(
      function::CallOp op, OpAdaptor adapter, mlir::ConversionPatternRewriter &rewriter
  ) const override {
    mlir::SmallVector<mlir::Type> newResultTypes;
    if (mlir::failed(getTypeConverter()->convertTypes(op.getResultTypes(), newResultTypes))) {
      return op->emitError("Could not convert Op result types.");
    }
    replaceOpWithNewOp<function::CallOp>(
        rewriter, op, newResultTypes, op.getCalleeAttr(), adapter.getMapOperands(),
        op.getNumDimsPerMapAttr(), adapter.getArgOperands()
    );
    return mlir::success();
  }
};

/// Pattern for `NewPodOp`, which lacks the default builder.
class NewPodOpClassReplacePattern : public mlir::OpConversionPattern<pod::NewPodOp> {
public:
  NewPodOpClassReplacePattern(mlir::TypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<pod::NewPodOp>(converter, ctx, 0) {}

  mlir::LogicalResult matchAndRewrite(
      pod::NewPodOp op, OpAdaptor adapter, mlir::ConversionPatternRewriter &rewriter
  ) const override {
    auto newResultType =
        dyn_cast_if_present<pod::PodType>(getTypeConverter()->convertType(op.getResult().getType())
        );
    if (!newResultType) {
      return op->emitError("Could not convert Op result types.");
    }
    mlir::OperandRangeRange orr = op.getMapOperands();
    mlir::SmallVector<mlir::ValueRange> mapOperands;
    mapOperands.reserve(orr.size());
    mapOperands.insert(mapOperands.end(), orr.begin(), orr.end());
    replaceOpWithNewOp<pod::NewPodOp>(
        rewriter, op, newResultType, mapOperands, op.getNumDimsPerMapAttr(),
        op.getInitializedRecordValues()
    );
    return mlir::success();
  }
};

template <typename I, typename NextOpClass, typename... OtherOpClasses>
inline void applyToMoreTypes(I inserter) {
  std::apply(inserter, std::tuple<NextOpClass, OtherOpClasses...> {});
}
template <typename I> inline void applyToMoreTypes(I) {}

} // namespace

/// Return a new `RewritePatternSet` covering all LLZK op types that may contain a StructType.
/// All fallback patterns use benefit 0 so higher-benefit patterns can override if needed.
///
/// The `GeneralTypeReplacePattern` is used for all classes in `AdditionalOpClasses`.
template <typename... AdditionalOpClasses>
inline mlir::RewritePatternSet newGeneralRewritePatternSet(
    mlir::TypeConverter &tyConv, mlir::MLIRContext *ctx, mlir::ConversionTarget &target
) {
  mlir::RewritePatternSet patterns(ctx);
  auto inserter = [&](auto... opClasses) {
    patterns.add<GeneralTypeReplacePattern<decltype(opClasses)>...>(tyConv, ctx);
  };
  std::apply(inserter, OpClassesWithStructTypes.WithGeneralBuilder);
  applyToMoreTypes<decltype(inserter), AdditionalOpClasses...>(inserter);
  // Special cases for ops where GeneralTypeReplacePattern doesn't work
  patterns.add<
      CreateArrayOpClassReplacePattern, CallOpClassReplacePattern, NewPodOpClassReplacePattern>(
      tyConv, ctx
  );
  // Add builtin FunctionType and SCF op converters
  mlir::populateFunctionOpInterfaceTypeConversionPattern<function::FuncDefOp>(patterns, tyConv);
  mlir::scf::populateSCFStructuralTypeConversionsAndLegality(tyConv, patterns, target);
  return patterns;
}

} // namespace llzk
