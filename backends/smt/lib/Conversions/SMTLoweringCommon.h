//===-- SMTLoweringCommon.h ------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Common private implementation shared by the integer SMT lowerings.
///
/// This layer only contains mechanical lowering infrastructure that is
/// independent of the modular encoding strategy: field selection, type
/// conversion, legality checks, and generic conversion patterns for struct,
/// function, and control-flow scaffolding. The naive and optimized passes then
/// supply their own policy for how felt arithmetic constraints are encoded.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/SMT/IR/SMTOps.h"
#include "llzk/Dialect/SMT/IR/SMTTypes.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/Field.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringMap.h>

#include <utility>

namespace llzk::smt::detail {

using SignalSymbols = llvm::DenseMap<llvm::StringRef, std::pair<mlir::Value, mlir::Value>>;

mlir::FailureOr<FieldRef> resolveSelectedField(mlir::ModuleOp mod, llvm::StringRef fieldName);

class LLZKToSMTTypeConverter : public mlir::TypeConverter {
public:
  explicit LLZKToSMTTypeConverter(mlir::MLIRContext *ctx);
};

bool containsFeltOrStruct(mlir::Type type);

mlir::Operation *convertStructProductToFunc(mlir::Operation *op, mlir::MLIRContext *context);

void configureSMTNoCFBodyConversionTarget(
    mlir::MLIRContext &context, mlir::ConversionTarget &target
);

mlir::Operation *applySMTNoCFBodyConversion(
    mlir::Operation *op, mlir::ConversionTarget &target, mlir::RewritePatternSet &&patterns
);

mlir::LogicalResult validateSupportedSMTMemberAccesses(component::StructDefOp structDef);

template <class From, class To> class BasicConverter : public mlir::OpConversionPattern<From> {
  using mlir::OpConversionPattern<From>::OpConversionPattern;

public:
  mlir::LogicalResult matchAndRewrite(
      From fromOp, typename From::Adaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override {
    rewriter.template replaceOpWithNewOp<To>(fromOp, adaptor.getOperands());
    return mlir::success();
  }
};

class FunctionDefConverter : public mlir::OpConversionPattern<function::FuncDefOp> {
  using mlir::OpConversionPattern<function::FuncDefOp>::OpConversionPattern;

public:
  mlir::LogicalResult matchAndRewrite(
      function::FuncDefOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;
};

class MemberReadConverter : public mlir::OpConversionPattern<component::MemberReadOp> {
public:
  MemberReadConverter(
      mlir::TypeConverter &converter, mlir::MLIRContext *context, const SignalSymbols &signalMap
  );

  mlir::LogicalResult matchAndRewrite(
      component::MemberReadOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;

private:
  SignalSymbols symbols;
};

class StructDefConverter : public mlir::OpConversionPattern<component::StructDefOp> {
  using mlir::OpConversionPattern<component::StructDefOp>::OpConversionPattern;

public:
  mlir::LogicalResult matchAndRewrite(
      component::StructDefOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;
};

class ReturnConverter : public mlir::OpConversionPattern<function::ReturnOp> {
  using mlir::OpConversionPattern<function::ReturnOp>::OpConversionPattern;

public:
  mlir::LogicalResult matchAndRewrite(
      function::ReturnOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;
};

class SCFIfConverter : public mlir::OpConversionPattern<mlir::scf::IfOp> {
  using mlir::OpConversionPattern<mlir::scf::IfOp>::OpConversionPattern;

public:
  mlir::LogicalResult matchAndRewrite(
      mlir::scf::IfOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;
};

class YieldConverter : public mlir::OpConversionPattern<mlir::scf::YieldOp> {
  using mlir::OpConversionPattern<mlir::scf::YieldOp>::OpConversionPattern;

public:
  mlir::LogicalResult matchAndRewrite(
      mlir::scf::YieldOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;
};

class FeltConstConverter : public mlir::OpConversionPattern<felt::FeltConstantOp> {
  using mlir::OpConversionPattern<felt::FeltConstantOp>::OpConversionPattern;

public:
  mlir::LogicalResult matchAndRewrite(
      felt::FeltConstantOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;
};

} // namespace llzk::smt::detail
