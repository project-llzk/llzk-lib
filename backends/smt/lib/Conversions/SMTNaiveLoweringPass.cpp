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

#include "SMTLoweringCommon.h"
#include "smt/Conversions/ConversionPasses.h"

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

static llvm::APSInt getSignedFeltThreshold(const llvm::APSInt &prime) {
  llvm::APSInt two(llvm::APInt(prime.getBitWidth(), 2), prime.isUnsigned());
  llvm::APSInt one(llvm::APInt(prime.getBitWidth(), 1), prime.isUnsigned());
  llvm::APSInt threshold = prime / two;
  threshold += one;
  return threshold;
}

static smt::IntConstantOp createSMTIntConstant(
    OpBuilder &builder, Location loc, MLIRContext *ctx, const llvm::APSInt &value
) {
  return builder.create<smt::IntConstantOp>(loc, IntegerAttr::get(ctx, value));
}

static smt::IntConstantOp createSMTPrimeConstant(
    OpBuilder &builder, Location loc, MLIRContext *ctx, const llvm::APSInt &prime
) {
  return createSMTIntConstant(builder, loc, ctx, prime);
}

static smt::IntConstantOp
createSMTZeroConstant(OpBuilder &builder, Location loc, MLIRContext *ctx) {
  return builder.create<smt::IntConstantOp>(
      loc, IntegerAttr::get(ctx, llvm::APSInt {llvm::APInt {1, 0}})
  );
}

static smt::IntConstantOp createSMTOneConstant(OpBuilder &builder, Location loc, MLIRContext *ctx) {
  return builder.create<smt::IntConstantOp>(
      loc, IntegerAttr::get(ctx, llvm::APSInt(llvm::APInt(64, 1), /*isUnsigned=*/false))
  );
}

static Value createSMTModPrimeExpr(
    OpBuilder &builder, Location loc, Value value, MLIRContext *ctx, const llvm::APSInt &prime
) {
  auto primeConst = createSMTPrimeConstant(builder, loc, ctx, prime);
  return builder.create<smt::IntModOp>(loc, ValueRange {value, primeConst.getResult()}).getResult();
}

class NaiveNonNativeStrategy {
public:
  explicit NaiveNonNativeStrategy(llvm::APSInt fieldPrime);

  smt::IntConstantOp createZeroConstant(OpBuilder &builder, Location loc, MLIRContext *ctx) const;
  smt::IntConstantOp createOneConstant(OpBuilder &builder, Location loc, MLIRContext *ctx) const;
  Value createModPrimeExpr(OpBuilder &builder, Location loc, Value value, MLIRContext *ctx) const;
  Value emitSignedIntDivisionValue(
      OpBuilder &builder, Location loc, Value lhs, Value rhs, MLIRContext *ctx
  ) const;
  Value emitSignedModValue(
      OpBuilder &builder, Location loc, Value lhs, Value rhs, MLIRContext *ctx
  ) const;
  std::string getFreshName(StringRef baseName) const;
  void populatePatterns(
      RewritePatternSet &patterns, TypeConverter &converter, MLIRContext *context,
      const SignalSymbols &signalSymbols
  ) const;

private:
  Value createSignedFeltExpr(OpBuilder &builder, Location loc, Value value, MLIRContext *ctx) const;
  Value createAbsExpr(OpBuilder &builder, Location loc, Value value, MLIRContext *ctx) const;
  Value createTruncatingSignedDivExpr(
      OpBuilder &builder, Location loc, Value lhs, Value rhs, MLIRContext *ctx
  ) const;
  llvm::APSInt prime;
  mutable llvm::StringMap<unsigned> freshSymbolCounts;
};

class NaiveFeltDivConverter : public OpConversionPattern<felt::DivFeltOp> {
public:
  NaiveFeltDivConverter(
      TypeConverter &converter, MLIRContext *context, const NaiveNonNativeStrategy *loweringStrategy
  )
      : OpConversionPattern<felt::DivFeltOp>(converter, context, /*benefit=*/2),
        strategy(loweringStrategy) {}

  LogicalResult matchAndRewrite(
      felt::DivFeltOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    auto div = rewriter.create<smt::DeclareFunOp>(
        op->getLoc(), smt::IntType::get(getContext()),
        StringAttr::get(getContext(), strategy->getFreshName("felt_div"))
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
      TypeConverter &converter, MLIRContext *context, const NaiveNonNativeStrategy *loweringStrategy
  )
      : OpConversionPattern<felt::InvFeltOp>(converter, context, /*benefit=*/2),
        strategy(loweringStrategy) {}

  LogicalResult matchAndRewrite(
      felt::InvFeltOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    auto inv = rewriter.create<smt::DeclareFunOp>(
        op->getLoc(), smt::IntType::get(getContext()),
        StringAttr::get(getContext(), strategy->getFreshName("inv"))
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
    auto productEqualsOne = rewriter.create<smt::EqOp>(op->getLoc(), productMod, one.getResult());
    auto invConstraint = rewriter.create<smt::IteOp>(
        op->getLoc(), operandIsZero.getResult(), invIsZero.getResult(), productEqualsOne.getResult()
    );
    rewriter.create<smt::AssertOp>(op->getLoc(), invConstraint.getResult());
    rewriter.replaceOp(op, inv.getResult());
    return success();
  }

private:
  const NaiveNonNativeStrategy *strategy;
};

class NaiveSignedIntDivConverter : public OpConversionPattern<felt::SignedIntDivFeltOp> {
public:
  NaiveSignedIntDivConverter(
      TypeConverter &converter, MLIRContext *context, const NaiveNonNativeStrategy *loweringStrategy
  )
      : OpConversionPattern<felt::SignedIntDivFeltOp>(converter, context, /*benefit=*/2),
        strategy(loweringStrategy) {}

  LogicalResult matchAndRewrite(
      felt::SignedIntDivFeltOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    rewriter.replaceOp(
        op, strategy->emitSignedIntDivisionValue(
                rewriter, op.getLoc(), adaptor.getLhs(), adaptor.getRhs(), getContext()
            )
    );
    return success();
  }

private:
  const NaiveNonNativeStrategy *strategy;
};

class NaiveSignedModConverter : public OpConversionPattern<felt::SignedModFeltOp> {
public:
  NaiveSignedModConverter(
      TypeConverter &converter, MLIRContext *context, const NaiveNonNativeStrategy *loweringStrategy
  )
      : OpConversionPattern<felt::SignedModFeltOp>(converter, context, /*benefit=*/2),
        strategy(loweringStrategy) {}

  LogicalResult matchAndRewrite(
      felt::SignedModFeltOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    rewriter.replaceOp(
        op, strategy->emitSignedModValue(
                rewriter, op.getLoc(), adaptor.getLhs(), adaptor.getRhs(), getContext()
            )
    );
    return success();
  }

private:
  const NaiveNonNativeStrategy *strategy;
};

class NaiveMemberWriteConverter : public OpConversionPattern<component::MemberWriteOp> {
public:
  NaiveMemberWriteConverter(
      TypeConverter &converter, MLIRContext *context, const SignalSymbols &signalMap,
      const NaiveNonNativeStrategy *loweringStrategy
  )
      : OpConversionPattern<component::MemberWriteOp>(converter, context, /*benefit=*/2),
        symbols(signalMap), strategy(loweringStrategy) {}

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
    auto witnessMod = strategy->createModPrimeExpr(rewriter, op->getLoc(), witness, getContext());
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
      TypeConverter &converter, MLIRContext *context, const NaiveNonNativeStrategy *loweringStrategy
  )
      : OpConversionPattern<constrain::EmitEqualityOp>(converter, context, /*benefit=*/2),
        strategy(loweringStrategy) {}

  LogicalResult matchAndRewrite(
      constrain::EmitEqualityOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    auto lhsMod =
        strategy->createModPrimeExpr(rewriter, op->getLoc(), adaptor.getLhs(), getContext());
    auto rhsMod =
        strategy->createModPrimeExpr(rewriter, op->getLoc(), adaptor.getRhs(), getContext());
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
NaiveNonNativeStrategy::NaiveNonNativeStrategy(llvm::APSInt fieldPrime)
    : prime(std::move(fieldPrime)) {}

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

Value NaiveNonNativeStrategy::createSignedFeltExpr(
    OpBuilder &builder, Location loc, Value value, MLIRContext *ctx
) const {
  Value canonical = createModPrimeExpr(builder, loc, value, ctx);
  Value threshold =
      createSMTIntConstant(builder, loc, ctx, getSignedFeltThreshold(prime)).getResult();
  auto inNonNegativeRange =
      builder.create<smt::IntCmpOp>(loc, smt::IntPredicate::lt, canonical, threshold);
  Value primeValue = createSMTPrimeConstant(builder, loc, ctx, prime).getResult();
  Value negativeRepresentative =
      builder.create<smt::IntSubOp>(loc, canonical, primeValue).getResult();
  return builder
      .create<smt::IteOp>(loc, inNonNegativeRange.getResult(), canonical, negativeRepresentative)
      .getResult();
}

Value NaiveNonNativeStrategy::createAbsExpr(
    OpBuilder &builder, Location loc, Value value, MLIRContext *ctx
) const {
  Value zero = createZeroConstant(builder, loc, ctx).getResult();
  auto isNegative = builder.create<smt::IntCmpOp>(loc, smt::IntPredicate::lt, value, zero);
  Value negated = builder.create<smt::IntSubOp>(loc, zero, value).getResult();
  return builder.create<smt::IteOp>(loc, isNegative.getResult(), negated, value).getResult();
}

Value NaiveNonNativeStrategy::createTruncatingSignedDivExpr(
    OpBuilder &builder, Location loc, Value lhs, Value rhs, MLIRContext *ctx
) const {
  Value zero = createZeroConstant(builder, loc, ctx).getResult();
  auto lhsNeg = builder.create<smt::IntCmpOp>(loc, smt::IntPredicate::lt, lhs, zero);
  auto rhsNeg = builder.create<smt::IntCmpOp>(loc, smt::IntPredicate::lt, rhs, zero);
  Value lhsAbs = createAbsExpr(builder, loc, lhs, ctx);
  Value rhsAbs = createAbsExpr(builder, loc, rhs, ctx);
  Value absQuotient = builder.create<smt::IntDivOp>(loc, lhsAbs, rhsAbs).getResult();
  Value signsDiffer =
      builder.create<smt::XOrOp>(loc, ValueRange {lhsNeg.getResult(), rhsNeg.getResult()})
          .getResult();
  Value negatedQuotient = builder.create<smt::IntSubOp>(loc, zero, absQuotient).getResult();
  return builder.create<smt::IteOp>(loc, signsDiffer, negatedQuotient, absQuotient).getResult();
}

Value NaiveNonNativeStrategy::emitSignedIntDivisionValue(
    OpBuilder &builder, Location loc, Value lhs, Value rhs, MLIRContext *ctx
) const {
  // `felt.sintdiv` interprets the felt operands as signed integers, divides
  // with truncation toward zero, and then converts the quotient back to a felt.
  Value signedLhs = createSignedFeltExpr(builder, loc, lhs, ctx);
  Value signedRhs = createSignedFeltExpr(builder, loc, rhs, ctx);
  Value signedQuotient = createTruncatingSignedDivExpr(builder, loc, signedLhs, signedRhs, ctx);
  return createModPrimeExpr(builder, loc, signedQuotient, ctx);
}

Value NaiveNonNativeStrategy::emitSignedModValue(
    OpBuilder &builder, Location loc, Value lhs, Value rhs, MLIRContext *ctx
) const {
  // `felt.smod` uses the remainder paired with `felt.sintdiv`: compute
  // q = trunc(lhs / rhs) in the signed embedding, then reduce lhs - q * rhs
  // back into the field.
  Value signedLhs = createSignedFeltExpr(builder, loc, lhs, ctx);
  Value signedRhs = createSignedFeltExpr(builder, loc, rhs, ctx);
  Value signedQuotient = createTruncatingSignedDivExpr(builder, loc, signedLhs, signedRhs, ctx);
  Value product =
      builder.create<smt::IntMulOp>(loc, ValueRange {signedQuotient, signedRhs}).getResult();
  Value signedRemainder = builder.create<smt::IntSubOp>(loc, signedLhs, product).getResult();
  return createModPrimeExpr(builder, loc, signedRemainder, ctx);
}

std::string NaiveNonNativeStrategy::getFreshName(StringRef baseName) const {
  unsigned count = freshSymbolCounts[baseName]++;
  if (count == 0) {
    return baseName.str();
  }

  std::string uniqueName(baseName);
  uniqueName += "_";
  uniqueName += std::to_string(count);
  return uniqueName;
}

void NaiveNonNativeStrategy::populatePatterns(
    RewritePatternSet &patterns, TypeConverter &converter, MLIRContext *context,
    const SignalSymbols &signalSymbols
) const {
  patterns.add<
      BasicConverter<felt::AddFeltOp, smt::IntAddOp>,
      BasicConverter<felt::SubFeltOp, smt::IntSubOp>,
      BasicConverter<felt::MulFeltOp, smt::IntMulOp>,
      BasicConverter<felt::NegFeltOp, smt::IntNegOp>,
      BasicConverter<felt::UnsignedModFeltOp, smt::IntModOp>, FeltConstConverter,
      FunctionDefConverter, ReturnConverter, SCFIfConverter, YieldConverter, NaiveBoolCmpConverter>(
      converter, context
  );
  patterns.add<NaiveFeltDivConverter>(converter, context, this);
  patterns.add<NaiveFeltInvConverter>(converter, context, this);
  patterns.add<NaiveSignedIntDivConverter>(converter, context, this);
  patterns.add<NaiveSignedModConverter>(converter, context, this);
  patterns.add<NaiveConstrainConverter>(converter, context, this);
  patterns.add<NaiveMemberWriteConverter>(converter, context, signalSymbols, this);
  patterns.add<MemberReadConverter>(converter, context, signalSymbols);
}

class SMTNaiveNonNativeLoweringPass
    : public smt::impl::SMTNaiveLoweringPassBase<SMTNaiveNonNativeLoweringPass> {
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<smt::SMTDialect, mlir::func::FuncDialect>();
  }

  Operation *convertBodies(
      Operation *op, const SignalSymbols &signalSymbols, const NaiveNonNativeStrategy &strategy
  ) {
    if (op == nullptr) {
      return op;
    }

    MLIRContext *context = &getContext();

    LLZKToSMTTypeConverter typeConverter {context};
    RewritePatternSet patterns {context};
    ConversionTarget target {*context};

    configureSMTNoCFBodyConversionTarget(target);
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

    auto walkResult = mod.walk([this, prime](component::StructDefOp structDef) {
      auto productFunc = structDef.getProductFuncOp();
      if (!productFunc) {
        structDef.emitError("SMT lowering requires a @product function");
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
            preamble, smt::IntType::get(&getContext()), StringAttr::get(&getContext(), witnessName)
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

    if (!walkResult.wasInterrupted()) {
      // Remove `llzk.main` attribute because `convertStructProductToFunc()` above deleted structs.
      mod->removeAttr(MAIN_ATTR_NAME);
    }
  }
};

} // namespace

namespace smt {

std::unique_ptr<mlir::Pass> createSMTNaiveLoweringPass() {
  return std::make_unique<SMTNaiveNonNativeLoweringPass>();
}

} // namespace smt

} // namespace llzk
