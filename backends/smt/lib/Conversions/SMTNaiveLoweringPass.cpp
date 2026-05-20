//===-- SMTNaiveLoweringPass.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a non-optimized SMT lowering for LLZK felt
/// operations. The methodology is intentionally simple: every field-level
/// equality is enforced by reducing both sides modulo the selected prime, and
/// control-flow is preserved for a later SMT CF lowering pass.
///
//===----------------------------------------------------------------------===//

#include "smt/Conversions/ConversionPasses.h"

#include "SMTLoweringCommon.h"

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Bool/IR/Enums.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/Include/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/SMT/IR/SMTDialect.h"
#include "llzk/Dialect/SMT/IR/SMTOps.h"
#include "llzk/Dialect/SMT/IR/SMTTypes.h"
#include "llzk/Dialect/String/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/Field.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/TypeSwitch.h>

#include <string>
#include <utility>

namespace llzk {
namespace smt {
#define GEN_PASS_DECL_SMTNAIVELOWERINGPASS
#define GEN_PASS_DEF_SMTNAIVELOWERINGPASS
#include "smt/Conversions/ConversionPasses.h.inc"
} // namespace smt

using namespace mlir;
using namespace llzk::smt::detail;

namespace {

static smt::IntConstantOp
createSMTIntConstant(OpBuilder &builder, Location loc, MLIRContext *ctx, const llvm::APSInt &value) {
  return builder.create<smt::IntConstantOp>(loc, IntegerAttr::get(ctx, value));
}

static smt::IntConstantOp createSMTPrimeConstant(OpBuilder &builder, Location loc,
                                                 MLIRContext *ctx,
                                                 const llvm::APSInt &prime) {
  return createSMTIntConstant(builder, loc, ctx, prime);
}

static smt::IntConstantOp createSMTZeroConstant(OpBuilder &builder, Location loc,
                                                MLIRContext *ctx) {
  return builder.create<smt::IntConstantOp>(
      loc, IntegerAttr::get(ctx, llvm::APSInt {llvm::APInt {1, 0}})
  );
}

static smt::IntConstantOp createSMTOneConstant(OpBuilder &builder, Location loc,
                                               MLIRContext *ctx) {
  return builder.create<smt::IntConstantOp>(
      loc, IntegerAttr::get(
               ctx, llvm::APSInt(llvm::APInt(64, 1), /*isUnsigned=*/false)
           )
  );
}

static Value createSMTModPrimeExpr(OpBuilder &builder, Location loc, Value value,
                                   MLIRContext *ctx, const llvm::APSInt &prime) {
  auto primeConst = createSMTPrimeConstant(builder, loc, ctx, prime);
  return builder.create<smt::IntModOp>(loc, ValueRange {value, primeConst.getResult()})
      .getResult();
}

class NaiveNonNativeStrategy {
public:
  explicit NaiveNonNativeStrategy(llvm::APSInt prime);

  smt::IntConstantOp createZeroConstant(OpBuilder &builder, Location loc,
                                        MLIRContext *ctx) const;
  smt::IntConstantOp createOneConstant(OpBuilder &builder, Location loc,
                                       MLIRContext *ctx) const;
  Value createModPrimeExpr(OpBuilder &builder, Location loc, Value value,
                           MLIRContext *ctx) const;
  void populatePatterns(RewritePatternSet &patterns, TypeConverter &typeConverter,
                        MLIRContext *context, const SignalSymbols &signalSymbols) const;

private:
  llvm::APSInt prime;
};

class NaiveFeltDivConverter : public OpConversionPattern<felt::DivFeltOp> {
public:
  NaiveFeltDivConverter(
      TypeConverter &typeConverter, MLIRContext *context,
      const NaiveNonNativeStrategy *strategy
  )
      : OpConversionPattern<felt::DivFeltOp>(typeConverter, context, /*benefit=*/2),
        strategy(strategy) {}

  LogicalResult matchAndRewrite(
      felt::DivFeltOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    auto div = rewriter.create<smt::DeclareFunOp>(
        op->getLoc(), smt::IntType::get(getContext()), StringAttr::get(getContext(), "div")
    );
    auto zero = strategy->createZeroConstant(rewriter, op->getLoc(), getContext());
    auto denominatorIsZero =
        rewriter.create<smt::EqOp>(op->getLoc(), adaptor.getRhs(), zero.getResult());
    auto divIsZero = rewriter.create<smt::EqOp>(op->getLoc(), div.getResult(), zero.getResult());
    auto product = rewriter.create<smt::IntMulOp>(
        op->getLoc(), ValueRange {adaptor.getRhs(), div.getResult()}
    );
    auto productMod =
        strategy->createModPrimeExpr(rewriter, op->getLoc(), product.getResult(), getContext());
    auto numeratorMod =
        strategy->createModPrimeExpr(rewriter, op->getLoc(), adaptor.getLhs(), getContext());
    auto productEqualsNumerator =
        rewriter.create<smt::EqOp>(op->getLoc(), productMod, numeratorMod);
    auto divConstraint = rewriter.create<smt::IteOp>(
        op->getLoc(), denominatorIsZero.getResult(), divIsZero.getResult(),
        productEqualsNumerator.getResult()
    );
    rewriter.create<smt::AssertOp>(op->getLoc(), divConstraint.getResult());
    rewriter.replaceOp(op, div.getResult());
    return success();
  }

private:
  const NaiveNonNativeStrategy *strategy;
};

class NaiveFeltInvConverter : public OpConversionPattern<felt::InvFeltOp> {
public:
  NaiveFeltInvConverter(
      TypeConverter &typeConverter, MLIRContext *context,
      const NaiveNonNativeStrategy *strategy
  )
      : OpConversionPattern<felt::InvFeltOp>(typeConverter, context, /*benefit=*/2),
        strategy(strategy) {}

  LogicalResult matchAndRewrite(
      felt::InvFeltOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    auto inv = rewriter.create<smt::DeclareFunOp>(
        op->getLoc(), smt::IntType::get(getContext()), StringAttr::get(getContext(), "inv")
    );
    auto zero = strategy->createZeroConstant(rewriter, op->getLoc(), getContext());
    auto one = strategy->createOneConstant(rewriter, op->getLoc(), getContext());
    auto operandIsZero =
        rewriter.create<smt::EqOp>(op->getLoc(), adaptor.getOperand(), zero.getResult());
    auto invIsZero = rewriter.create<smt::EqOp>(op->getLoc(), inv.getResult(), zero.getResult());
    auto product = rewriter.create<smt::IntMulOp>(
        op->getLoc(), ValueRange {adaptor.getOperand(), inv.getResult()}
    );
    auto productMod =
        strategy->createModPrimeExpr(rewriter, op->getLoc(), product.getResult(), getContext());
    auto productEqualsOne =
        rewriter.create<smt::EqOp>(op->getLoc(), productMod, one.getResult());
    auto invConstraint = rewriter.create<smt::IteOp>(
        op->getLoc(), operandIsZero.getResult(), invIsZero.getResult(),
        productEqualsOne.getResult()
    );
    rewriter.create<smt::AssertOp>(op->getLoc(), invConstraint.getResult());
    rewriter.replaceOp(op, inv.getResult());
    return success();
  }

private:
  const NaiveNonNativeStrategy *strategy;
};

class NaiveMemberWriteConverter : public OpConversionPattern<component::MemberWriteOp> {
public:
  NaiveMemberWriteConverter(
      TypeConverter &typeConverter, MLIRContext *context, const SignalSymbols &signalMap,
      const NaiveNonNativeStrategy *strategy
  )
      : OpConversionPattern<component::MemberWriteOp>(typeConverter, context, /*benefit=*/2),
        symbols(signalMap), strategy(strategy) {}

  LogicalResult matchAndRewrite(
      component::MemberWriteOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (!isa<felt::FeltType>(op.getVal().getType())) {
      op.emitError("SMT lowering currently only supports felt-valued struct.writem");
      return failure();
    }

    auto it = symbols.find(adaptor.getMemberName());
    if (it == symbols.end()) {
      return failure();
    }

    auto [_, witness] = it->second;
    auto witnessMod =
        strategy->createModPrimeExpr(rewriter, op->getLoc(), witness, getContext());
    auto valueMod =
        strategy->createModPrimeExpr(rewriter, op->getLoc(), adaptor.getVal(), getContext());
    auto equal = rewriter.create<smt::EqOp>(op->getLoc(), witnessMod, valueMod);
    rewriter.replaceOpWithNewOp<smt::AssertOp>(op, equal.getResult());
    return success();
  }

private:
  SignalSymbols symbols;
  const NaiveNonNativeStrategy *strategy;
};

class NaiveConstrainConverter : public OpConversionPattern<constrain::EmitEqualityOp> {
public:
  NaiveConstrainConverter(
      TypeConverter &typeConverter, MLIRContext *context,
      const NaiveNonNativeStrategy *strategy
  )
      : OpConversionPattern<constrain::EmitEqualityOp>(typeConverter, context, /*benefit=*/2),
        strategy(strategy) {}

  LogicalResult matchAndRewrite(
      constrain::EmitEqualityOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    auto lhsMod = strategy->createModPrimeExpr(
        rewriter, op->getLoc(), adaptor.getLhs(), getContext()
    );
    auto rhsMod = strategy->createModPrimeExpr(
        rewriter, op->getLoc(), adaptor.getRhs(), getContext()
    );
    auto eq = rewriter.create<smt::EqOp>(op->getLoc(), lhsMod, rhsMod);
    rewriter.replaceOpWithNewOp<smt::AssertOp>(op, eq.getResult());
    return success();
  }

private:
  const NaiveNonNativeStrategy *strategy;
};

class NaiveBoolCmpConverter : public OpConversionPattern<boolean::CmpOp> {
  using OpConversionPattern<boolean::CmpOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      boolean::CmpOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    switch (adaptor.getPredicate()) {
    case boolean::FeltCmpPredicate::EQ:
      rewriter.replaceOpWithNewOp<smt::EqOp>(op, adaptor.getLhs(), adaptor.getRhs());
      return success();
    case boolean::FeltCmpPredicate::NE:
      rewriter.replaceOpWithNewOp<smt::NotOp>(
          op,
          rewriter.create<smt::EqOp>(op.getLoc(), adaptor.getLhs(), adaptor.getRhs()).getResult()
      );
      return success();
    default: {
      static DenseMap<boolean::FeltCmpPredicate, smt::IntPredicate> predicateComparator = {
          {boolean::FeltCmpPredicate::GE, smt::IntPredicate::ge},
          {boolean::FeltCmpPredicate::GT, smt::IntPredicate::gt},
          {boolean::FeltCmpPredicate::LE, smt::IntPredicate::le},
          {boolean::FeltCmpPredicate::LT, smt::IntPredicate::lt}
      };
      rewriter.replaceOpWithNewOp<smt::IntCmpOp>(
          op, predicateComparator[adaptor.getPredicate()], adaptor.getLhs(), adaptor.getRhs()
      );
      return success();
    }
    }
  }
};

/// Bundle the naive non-native lowering policy. This keeps the original
/// modulo-based encoding explicit while making the lowering strategy visible as
/// a separate concept from the SMT integer operations used to materialize it.
NaiveNonNativeStrategy::NaiveNonNativeStrategy(llvm::APSInt prime) : prime(std::move(prime)) {}

smt::IntConstantOp NaiveNonNativeStrategy::createZeroConstant(
    OpBuilder &builder, Location loc, MLIRContext *ctx
) const {
  return createSMTZeroConstant(builder, loc, ctx);
}

smt::IntConstantOp NaiveNonNativeStrategy::createOneConstant(
    OpBuilder &builder, Location loc, MLIRContext *ctx
) const {
  return createSMTOneConstant(builder, loc, ctx);
}

Value NaiveNonNativeStrategy::createModPrimeExpr(
    OpBuilder &builder, Location loc, Value value, MLIRContext *ctx
) const {
  return createSMTModPrimeExpr(builder, loc, value, ctx, prime);
}

void NaiveNonNativeStrategy::populatePatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context,
    const SignalSymbols &signalSymbols
) const {
  patterns.add<
      BasicConverter<felt::AddFeltOp, smt::IntAddOp>,
      BasicConverter<felt::SubFeltOp, smt::IntSubOp>,
      BasicConverter<felt::MulFeltOp, smt::IntMulOp>,
      BasicConverter<felt::NegFeltOp, smt::IntNegOp>,
      BasicConverter<felt::SignedIntDivFeltOp, smt::IntDivOp>,
      BasicConverter<felt::UnsignedModFeltOp, smt::IntModOp>,
      BasicConverter<felt::SignedModFeltOp, smt::IntModOp>, FeltConstConverter,
      FunctionDefConverter, ReturnConverter, SCFIfConverter, YieldConverter,
      NaiveBoolCmpConverter>(typeConverter, context);
  patterns.add<NaiveFeltDivConverter>(typeConverter, context, this);
  patterns.add<NaiveFeltInvConverter>(typeConverter, context, this);
  patterns.add<NaiveConstrainConverter>(typeConverter, context, this);
  patterns.add<NaiveMemberWriteConverter>(typeConverter, context, signalSymbols, this);
  patterns.add<MemberReadConverter>(typeConverter, context, signalSymbols);
}

class SMTNaiveNonNativeLoweringPass
    : public smt::impl::SMTNaiveLoweringPassBase<SMTNaiveNonNativeLoweringPass> {
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<smt::SMTDialect, mlir::func::FuncDialect>();
  }

  Operation *convertBodies(Operation *op, const SignalSymbols &signalSymbols,
                           const NaiveNonNativeStrategy &strategy) {
    if (op == nullptr) {
      return op;
    }

    MLIRContext *context = &getContext();

    LLZKToSMTTypeConverter typeConverter {context};
    RewritePatternSet patterns {context};
    ConversionTarget target {*context};

    configureSMTNoCFBodyConversionTarget(*context, target);
    strategy.populatePatterns(patterns, typeConverter, context, signalSymbols);
    return applySMTNoCFBodyConversion(op, target, std::move(patterns));
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto selectedField = resolveSelectedField(mod, fieldName);
    if (failed(selectedField)) {
      return signalPassFailure();
    }
    auto prime = toAPSInt(selectedField->get().prime());

    mod.walk([this, prime](component::StructDefOp structDef) {
      auto productFunc = structDef.getProductFuncOp();
      if (!productFunc) {
        structDef.emitError("SMT lowering requires a @product function");
        signalPassFailure();
        return WalkResult::interrupt();
      }
      if (failed(validateSupportedSMTMemberAccesses(structDef))) {
        signalPassFailure();
        return WalkResult::interrupt();
      }

      IRRewriter rewriter {&getContext()};
      rewriter.setInsertionPointToStart(&productFunc.getFunctionBody().front());
      auto preamble = productFunc->getLoc();

      SignalSymbols signalSymbols;
      for (auto memberDef : structDef.getMemberDefs()) {
        if (!isa<felt::FeltType>(memberDef.getType())) {
          continue;
        }

        std::string constraintName = memberDef.getSymName().str() + "_c";
        std::string witnessName = memberDef.getSymName().str() + "_w";
        auto constraintSym = rewriter.create<smt::DeclareFunOp>(
            preamble, smt::IntType::get(&getContext()),
            StringAttr::get(&getContext(), constraintName)
        );
        auto witnessSym = rewriter.create<smt::DeclareFunOp>(
            preamble, smt::IntType::get(&getContext()),
            StringAttr::get(&getContext(), witnessName)
        );
        signalSymbols[memberDef.getSymName()] = {constraintSym.getResult(), witnessSym.getResult()};
      }

      NaiveNonNativeStrategy strategy {prime};
      auto *result = convertStructProductToFunc(
          convertBodies(structDef, signalSymbols, strategy), &getContext()
      );
      if (result == nullptr) {
        signalPassFailure();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }
};

} // namespace

namespace smt {

std::unique_ptr<mlir::Pass> createSMTNaiveLoweringPass() {
  return std::make_unique<SMTNaiveNonNativeLoweringPass>();
}

} // namespace smt

} // namespace llzk
