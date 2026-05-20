//===-- SMTLoweringPass.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-smt-lowering` pass.
///
//===----------------------------------------------------------------------===//

#include "SMTLoweringCommon.h"
#include "smt/Conversions/ConversionPasses.h"

#include "llzk/Analysis/IntervalAnalysis.h"
#include "llzk/Analysis/SourceRef.h"
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
#include "llzk/Util/Constants.h"
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

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace llzk {
namespace smt {
#define GEN_PASS_DECL_SMTLOWERINGPASS
#define GEN_PASS_DEF_SMTLOWERINGPASS
#include "smt/Conversions/ConversionPasses.h.inc"
} // namespace smt

using namespace mlir;
using namespace llzk::smt::detail;

namespace {

/// Decide when modular reduction can be avoided or must be reintroduced.
///
/// This helper owns the interval-analysis view of the current lowering target.
/// It computes the range-based facts that drive the optimized non-native
/// encoding, but it does not emit SMT operations itself. Keeping that reasoning
/// separate makes it possible to reuse the same modulo-elimination methodology
/// for other non-native target theories in the future.
class ModularReasoner {
public:
  static constexpr uint64_t explicitReductionQuotientThreshold = 1000;

  enum class ReductionKind {
    Direct,
    NativeMod,
    // Given a constraint of the form e % p = 0, `ExplicitWitness` reduces the constraint to
    // e = q*p + r where q, r are fresh variables. It further range constrains r to lie in [0, p-1]
    // and also constrains the quotient based on the unreduced range on e.
    ExplicitWitness,
  };

  struct ReductionPlan {
    ReductionKind kind;
    std::optional<UnreducedInterval> quotientRange;
  };

  ModularReasoner(
      const Field &selectedField, DataFlowSolver &dataflowSolver,
      const StructIntervals *intervals = nullptr
  )
      : field(selectedField), prime(toAPSInt(selectedField.prime())),
        primeDynamic(toDynamicAPInt(prime)), solver(dataflowSolver) {
    if (!intervals) {
      return;
    }

    captureMemberRanges(intervals->getComputeIntervals(), witnessRanges);
    captureMemberRanges(intervals->getConstrainIntervals(), constraintRanges);
  }

  bool isScalarFeltType(Type type) const { return isa<felt::FeltType>(type); }

  UnreducedInterval getDefaultFeltRange() const {
    return UnreducedInterval(field.get().zero(), field.get().maxVal());
  }

  UnreducedInterval getScalarValueRange(Value value) const {
    if (const auto *lattice = solver.lookupState<IntervalAnalysisLattice>(value)) {
      const ExpressionValue &expr = lattice->getValue().getScalarValue();
      if (expr.hasUnreducedInterval()) {
        return expr.getUnreducedInterval();
      }
      if (expr.getExpr() != nullptr) {
        return expr.getInterval().firstUnreduced();
      }
    }
    return getDefaultFeltRange();
  }

  UnreducedInterval getWitnessMemberRange(StringRef memberName) const {
    return lookupMemberRange(witnessRanges, memberName);
  }

  UnreducedInterval getConstraintMemberRange(StringRef memberName) const {
    return lookupMemberRange(constraintRanges, memberName);
  }

  bool unionWidthLessThanPrime(const UnreducedInterval &lhs, const UnreducedInterval &rhs) const {
    return rangeSpanLessThanPrime(lhs.doUnion(rhs));
  }

  bool spansModulusBoundary(const UnreducedInterval &range) const {
    return floorDiv(range.getLHS()) != floorDiv(range.getRHS());
  }

  bool sameResidueWindow(const UnreducedInterval &lhs, const UnreducedInterval &rhs) const {
    llvm::DynamicAPInt lhsLow = floorDiv(lhs.getLHS());
    llvm::DynamicAPInt lhsHigh = floorDiv(lhs.getRHS());
    llvm::DynamicAPInt rhsLow = floorDiv(rhs.getLHS());
    llvm::DynamicAPInt rhsHigh = floorDiv(rhs.getRHS());
    return lhsLow == lhsHigh && lhsLow == rhsLow && lhsLow == rhsHigh;
  }

  bool isCanonical(const UnreducedInterval &range) const {
    llvm::DynamicAPInt zero(0);
    return range.getLHS() >= zero && range.getRHS() < primeDynamic;
  }

  bool maybeContainsZeroResidue(const UnreducedInterval &range) const {
    llvm::DynamicAPInt firstMultiple = ceilDiv(range.getLHS()) * primeDynamic;
    return firstMultiple <= range.getRHS();
  }

  ReductionPlan planCanonicalization(const UnreducedInterval &range) const {
    if (isCanonical(range)) {
      return {ReductionKind::Direct, std::nullopt};
    }

    if (!shouldUseExplicitReduction(range)) {
      return {ReductionKind::NativeMod, std::nullopt};
    }

    return {ReductionKind::ExplicitWitness, getQuotientRange(range)};
  }

  ReductionPlan
  planCongruence(const UnreducedInterval &lhsRange, const UnreducedInterval &rhsRange) const {
    if (explicitReductionQuotientThreshold > 0 && unionWidthLessThanPrime(lhsRange, rhsRange)) {
      return {ReductionKind::Direct, std::nullopt};
    }

    auto diffRange = lhsRange - rhsRange;
    if (!shouldUseExplicitReduction(diffRange)) {
      return {ReductionKind::NativeMod, std::nullopt};
    }

    return {ReductionKind::ExplicitWitness, getQuotientRange(diffRange)};
  }

  const llvm::APSInt &getPrime() const { return prime; }

private:
  std::reference_wrapper<const Field> field;
  llvm::APSInt prime;
  llvm::DynamicAPInt primeDynamic;
  DataFlowSolver &solver;
  llvm::StringMap<UnreducedInterval> witnessRanges;
  llvm::StringMap<UnreducedInterval> constraintRanges;

  static bool isDirectMemberRef(const SourceRef &ref) {
    return ref.isRooted() && ref.getPath().size() == 1 && ref.getPath().front().isMember();
  }

  void captureMemberRanges(
      const llvm::MapVector<SourceRef, Interval> &reducedRanges,
      llvm::StringMap<UnreducedInterval> &out
  ) {
    for (const auto &[ref, reduced] : reducedRanges) {
      if (!isDirectMemberRef(ref)) {
        continue;
      }

      auto member = ref.getPath().front().getMember();
      StringRef memberName = member.getSymName();
      // Member symbols denote stored felt values. Use the reduced member interval lifted
      // to a single residue window, rather than the producer expression's unreduced interval.
      // Otherwise writes like `1 - x * inv` incorrectly widen the stored witness symbol.
      UnreducedInterval range = reduced.firstUnreduced();
      auto [it, inserted] = out.try_emplace(memberName, range);
      if (!inserted) {
        it->second = range;
      }
    }
  }

  UnreducedInterval
  lookupMemberRange(const llvm::StringMap<UnreducedInterval> &ranges, StringRef memberName) const {
    if (auto it = ranges.find(memberName); it != ranges.end()) {
      return it->second;
    }
    return getDefaultFeltRange();
  }

  bool rangeSpanLessThanPrime(const UnreducedInterval &range) const {
    return range.getRHS() - range.getLHS() < primeDynamic;
  }

  UnreducedInterval getQuotientRange(const UnreducedInterval &range) const {
    return UnreducedInterval(floorDiv(range.getLHS()), floorDiv(range.getRHS()));
  }

  bool shouldUseExplicitReduction(const UnreducedInterval &range) const {
    auto quotientRange = getQuotientRange(range);
    auto quotientWidth = quotientRange.getRHS() - quotientRange.getLHS();
    return quotientWidth < llvm::DynamicAPInt(explicitReductionQuotientThreshold);
  }

  llvm::DynamicAPInt floorDiv(const llvm::DynamicAPInt &value) const {
    llvm::APSInt lhs = toAPSInt(value);
    llvm::APSInt rhs = toAPSInt(primeDynamic);
    unsigned width = std::max(lhs.getBitWidth(), rhs.getBitWidth()) + 1;
    lhs = lhs.extend(width);
    rhs = rhs.extend(width);
    lhs.setIsSigned(true);
    rhs.setIsSigned(true);
    llvm::APSInt quotient = lhs / rhs;
    llvm::APSInt remainder = lhs % rhs;
    if (remainder < 0) {
      quotient -= llvm::APSInt(llvm::APInt(width, 1), /*isUnsigned=*/false);
    }
    return toDynamicAPInt(quotient);
  }

  llvm::DynamicAPInt ceilDiv(const llvm::DynamicAPInt &value) const { return -floorDiv(-value); }

  static std::string makeName(StringRef prefix, StringRef suffix) {
    std::string name(prefix);
    name += suffix;
    return name;
  }
};

/// Theory-neutral primitive emitter interface used by non-native encoders.
class NonNativeTheoryEmitter {
public:
  virtual ~NonNativeTheoryEmitter() = default;

  virtual void emitRangeConstraint(
      OpBuilder &builder, Location loc, Value value, const UnreducedInterval &range
  ) const = 0;
  virtual Value emitFreshSymbol(OpBuilder &builder, Location loc, StringRef name) const = 0;
  virtual Value
  emitConstant(OpBuilder &builder, Location loc, const llvm::DynamicAPInt &value) const = 0;
  virtual Value emitSub(OpBuilder &builder, Location loc, Value lhs, Value rhs) const = 0;
  virtual Value emitAdd(OpBuilder &builder, Location loc, Value lhs, Value rhs) const = 0;
  virtual Value emitMul(OpBuilder &builder, Location loc, Value lhs, Value rhs) const = 0;
  virtual Value emitModPrime(OpBuilder &builder, Location loc, Value value) const = 0;
  virtual Value emitPrimeMultiple(OpBuilder &builder, Location loc, Value factor) const = 0;
  virtual Value emitOrderedComparison(
      OpBuilder &builder, Location loc, boolean::FeltCmpPredicate predicate, Value lhs, Value rhs
  ) const = 0;
};

/// Emit primitive integer-theory terms for the optimized non-native encoding.
///
/// This layer only builds integer-sorted values and
/// arithmetic fragments. Higher-level non-native encoding structure lives above
/// this emitter.
class SMTIntTheoryEmitter : public NonNativeTheoryEmitter {
public:
  SMTIntTheoryEmitter(MLIRContext *context, const ModularReasoner &reasoner)
      : ctx(context), reasoner(reasoner) {}

  void emitRangeConstraint(
      OpBuilder &builder, Location loc, Value value, const UnreducedInterval &range
  ) const override {
    auto lower = createIntConstant(builder, loc, range.getLHS());
    auto upper = createIntConstant(builder, loc, range.getRHS());
    auto lowerBound =
        builder.create<smt::IntCmpOp>(loc, smt::IntPredicate::ge, value, lower.getResult());
    auto upperBound =
        builder.create<smt::IntCmpOp>(loc, smt::IntPredicate::le, value, upper.getResult());
    // Assert the lower bound of the canonical/unreduced interval for this symbol.
    builder.create<smt::AssertOp>(loc, lowerBound.getResult());
    // Assert the upper bound of the canonical/unreduced interval for this symbol.
    builder.create<smt::AssertOp>(loc, upperBound.getResult());
  }

  Value emitFreshSymbol(OpBuilder &builder, Location loc, StringRef name) const override {
    return builder
        .create<smt::DeclareFunOp>(loc, smt::IntType::get(ctx), StringAttr::get(ctx, name))
        .getResult();
  }

  Value
  emitConstant(OpBuilder &builder, Location loc, const llvm::DynamicAPInt &value) const override {
    return createIntConstant(builder, loc, value).getResult();
  }

  Value emitSub(OpBuilder &builder, Location loc, Value lhs, Value rhs) const override {
    return builder.create<smt::IntSubOp>(loc, lhs, rhs).getResult();
  }

  Value emitAdd(OpBuilder &builder, Location loc, Value lhs, Value rhs) const override {
    return builder.create<smt::IntAddOp>(loc, ValueRange {lhs, rhs}).getResult();
  }

  Value emitMul(OpBuilder &builder, Location loc, Value lhs, Value rhs) const override {
    return builder.create<smt::IntMulOp>(loc, ValueRange {lhs, rhs}).getResult();
  }

  Value emitModPrime(OpBuilder &builder, Location loc, Value value) const override {
    auto primeConst = createPrimeConstant(builder, loc);
    return builder.create<smt::IntModOp>(loc, ValueRange {value, primeConst.getResult()})
        .getResult();
  }

  Value emitPrimeMultiple(OpBuilder &builder, Location loc, Value factor) const override {
    auto primeConst = createPrimeConstant(builder, loc);
    return emitMul(builder, loc, factor, primeConst.getResult());
  }

  Value emitOrderedComparison(
      OpBuilder &builder, Location loc, boolean::FeltCmpPredicate predicate, Value lhs, Value rhs
  ) const override {
    static DenseMap<boolean::FeltCmpPredicate, smt::IntPredicate> predicateComparator = {
        {boolean::FeltCmpPredicate::GE, smt::IntPredicate::ge},
        {boolean::FeltCmpPredicate::GT, smt::IntPredicate::gt},
        {boolean::FeltCmpPredicate::LE, smt::IntPredicate::le},
        {boolean::FeltCmpPredicate::LT, smt::IntPredicate::lt}
    };
    return builder.create<smt::IntCmpOp>(loc, predicateComparator[predicate], lhs, rhs).getResult();
  }

private:
  MLIRContext *ctx;
  const ModularReasoner &reasoner;

  smt::IntConstantOp createPrimeConstant(OpBuilder &builder, Location loc) const {
    return builder.create<smt::IntConstantOp>(loc, IntegerAttr::get(ctx, reasoner.getPrime()));
  }

  smt::IntConstantOp
  createIntConstant(OpBuilder &builder, Location loc, const llvm::DynamicAPInt &value) const {
    return builder.create<smt::IntConstantOp>(loc, IntegerAttr::get(ctx, toAPSInt(value)));
  }
};

} // namespace

/// Lower felt operations by combining interval-guided modular reasoning with a
/// theory-neutral primitive emitter.
///
/// The core idea is:
/// 1. Use interval analysis to decide whether modular reduction can be dropped.
/// 2. If reduction is still needed, choose between native `mod p` and an
///    explicit quotient witness based on quotient width.
/// 3. Compose those choices into equality, comparison, division, and inverse
///    encodings using generic SMT structure such as `eq`, `ite`, and `assert`.
///
/// This class owns the encoding policy. The theory emitter beneath it only
/// knows how to build primitive target-theory terms.
class OptimizedNonNativeStrategy {
public:
  OptimizedNonNativeStrategy(
      MLIRContext *context, const Field &selectedField, DataFlowSolver &dataflowSolver,
      const StructIntervals *intervals = nullptr
  );

  bool isScalarFeltType(Type type) const;
  UnreducedInterval getDefaultFeltRange() const;
  UnreducedInterval getScalarValueRange(Value value) const;
  UnreducedInterval getConstraintMemberRange(StringRef memberName) const;
  UnreducedInterval getWitnessMemberRange(StringRef memberName) const;
  void emitRangeConstraint(
      OpBuilder &builder, Location loc, Value value, const UnreducedInterval &range
  ) const;
  bool maybeContainsZeroResidue(const UnreducedInterval &range) const;
  bool spansModulusBoundary(const UnreducedInterval &range) const;
  bool sameResidueWindow(const UnreducedInterval &lhs, const UnreducedInterval &rhs) const;
  Value canonicalizeValue(
      OpBuilder &builder, Location loc, Value value, const UnreducedInterval &range,
      StringRef prefix
  ) const;
  Value buildCanonicalEqualityPredicate(
      OpBuilder &builder, Location loc, Value lhs, const UnreducedInterval &lhsRange, Value rhs,
      const UnreducedInterval &rhsRange, StringRef prefix
  ) const;
  Value buildCongruenceEqualityPredicate(
      OpBuilder &builder, Location loc, Value lhs, const UnreducedInterval &lhsRange, Value rhs,
      const UnreducedInterval &rhsRange, StringRef prefix
  ) const;
  Value emitOrderedComparisonPredicate(
      OpBuilder &builder, Location loc, boolean::FeltCmpPredicate predicate, Value lhs,
      const UnreducedInterval &lhsRange, Value rhs, const UnreducedInterval &rhsRange,
      StringRef prefix
  ) const;
  void emitCongruenceEqualityAssertion(
      OpBuilder &builder, Location loc, Value lhs, const UnreducedInterval &lhsRange, Value rhs,
      const UnreducedInterval &rhsRange, StringRef prefix
  ) const;
  Value emitDivisionValue(
      OpBuilder &builder, Location loc, Value numerator, const UnreducedInterval &numeratorRange,
      Value denominator, const UnreducedInterval &denominatorRange,
      const UnreducedInterval &resultRange
  ) const;
  Value emitInverseValue(
      OpBuilder &builder, Location loc, Value operand, const UnreducedInterval &operandRange
  ) const;
  void populatePatterns(
      RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context,
      const SignalSymbols &signalSymbols
  ) const;

private:
  ModularReasoner reasoner;
  std::unique_ptr<NonNativeTheoryEmitter> emitter;
};

class FeltDivConverter : public OpConversionPattern<felt::DivFeltOp> {
public:
  FeltDivConverter(
      TypeConverter &typeConverter, MLIRContext *context, const OptimizedNonNativeStrategy *strategy
  )
      : OpConversionPattern<felt::DivFeltOp>(typeConverter, context, /*benefit=*/2),
        strategy(strategy) {}

  LogicalResult matchAndRewrite(
      felt::DivFeltOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    auto divRange = strategy->getScalarValueRange(op.getResult());
    auto lhsRange = strategy->getScalarValueRange(op.getLhs());
    auto rhsRange = strategy->getScalarValueRange(op.getRhs());
    auto div = strategy->emitDivisionValue(
        rewriter, op.getLoc(), adaptor.getLhs(), lhsRange, adaptor.getRhs(), rhsRange, divRange
    );
    rewriter.replaceOp(op, div);

    return success();
  }

private:
  const OptimizedNonNativeStrategy *strategy;
};

class FeltInvConverter : public OpConversionPattern<felt::InvFeltOp> {
public:
  FeltInvConverter(
      TypeConverter &typeConverter, MLIRContext *context, const OptimizedNonNativeStrategy *strategy
  )
      : OpConversionPattern<felt::InvFeltOp>(typeConverter, context, /*benefit=*/2),
        strategy(strategy) {}

  LogicalResult matchAndRewrite(
      felt::InvFeltOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    auto operandRange = strategy->getScalarValueRange(op.getOperand());
    auto inv =
        strategy->emitInverseValue(rewriter, op.getLoc(), adaptor.getOperand(), operandRange);
    rewriter.replaceOp(op, inv);

    return success();
  }

private:
  const OptimizedNonNativeStrategy *strategy;
};

class MemberWriteConverter : public OpConversionPattern<component::MemberWriteOp> {
public:
  MemberWriteConverter(
      TypeConverter &typeConverter, MLIRContext *context, const SignalSymbols &signalMap,
      const OptimizedNonNativeStrategy *strategy
  )
      : OpConversionPattern<component::MemberWriteOp>(typeConverter, context, /*benefit=*/2),
        symbols(signalMap), strategy(strategy) {}

  LogicalResult matchAndRewrite(
      component::MemberWriteOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (!strategy->isScalarFeltType(op.getVal().getType())) {
      op.emitError("SMT lowering currently only supports felt-valued struct.writem");
      return failure();
    }

    auto it = symbols.find(adaptor.getMemberName());
    if (it == symbols.end()) {
      return failure();
    }

    auto [_, witness] = it->second;
    auto witnessRange = strategy->getWitnessMemberRange(adaptor.getMemberName());
    auto valueRange = strategy->getScalarValueRange(op.getVal());
    strategy->emitCongruenceEqualityAssertion(
        rewriter, op.getLoc(), witness, witnessRange, adaptor.getVal(), valueRange, "member_write"
    );
    rewriter.eraseOp(op);

    return success();
  }

private:
  SignalSymbols symbols;
  const OptimizedNonNativeStrategy *strategy;
};

class ConstrainConverter : public OpConversionPattern<constrain::EmitEqualityOp> {
public:
  ConstrainConverter(
      TypeConverter &typeConverter, MLIRContext *context, const OptimizedNonNativeStrategy *strategy
  )
      : OpConversionPattern<constrain::EmitEqualityOp>(typeConverter, context, /*benefit=*/2),
        strategy(strategy) {}

  LogicalResult matchAndRewrite(
      constrain::EmitEqualityOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    auto lhsRange = strategy->getScalarValueRange(op.getLhs());
    auto rhsRange = strategy->getScalarValueRange(op.getRhs());
    strategy->emitCongruenceEqualityAssertion(
        rewriter, op.getLoc(), adaptor.getLhs(), lhsRange, adaptor.getRhs(), rhsRange,
        "constrain_eq"
    );
    rewriter.eraseOp(op);
    return success();
  }

private:
  const OptimizedNonNativeStrategy *strategy;
};

class BoolCmpConverter : public OpConversionPattern<boolean::CmpOp> {
public:
  BoolCmpConverter(
      TypeConverter &typeConverter, MLIRContext *context, const OptimizedNonNativeStrategy *strategy
  )
      : OpConversionPattern<boolean::CmpOp>(typeConverter, context), strategy(strategy) {}

  LogicalResult matchAndRewrite(
      boolean::CmpOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    auto lhsRange = strategy->getScalarValueRange(op.getLhs());
    auto rhsRange = strategy->getScalarValueRange(op.getRhs());

    switch (adaptor.getPredicate()) {
    case boolean::FeltCmpPredicate::EQ: {
      auto eq = strategy->buildCanonicalEqualityPredicate(
          rewriter, op.getLoc(), adaptor.getLhs(), lhsRange, adaptor.getRhs(), rhsRange,
          "bool_cmp_eq"
      );
      rewriter.replaceOp(op, eq);
      return success();
    }
    case boolean::FeltCmpPredicate::NE: {
      auto eq = strategy->buildCanonicalEqualityPredicate(
          rewriter, op.getLoc(), adaptor.getLhs(), lhsRange, adaptor.getRhs(), rhsRange,
          "bool_cmp_ne"
      );
      rewriter.replaceOp(op, rewriter.create<smt::NotOp>(op.getLoc(), eq).getResult());
      return success();
    }
    default: {
      Value cmp = strategy->emitOrderedComparisonPredicate(
          rewriter, op.getLoc(), adaptor.getPredicate(), adaptor.getLhs(), lhsRange,
          adaptor.getRhs(), rhsRange, "bool_cmp_ordered"
      );
      rewriter.replaceOp(op, cmp);
      return success();
    }
    }
  }

private:
  const OptimizedNonNativeStrategy *strategy;
};

/// Bundle the optimized non-native lowering policy with the current SMT int
/// emitter. The strategy decides which range-based encoding shape should be
/// used, while the emitter remains responsible for spelling that choice in the
/// target SMT integer theory.
OptimizedNonNativeStrategy::OptimizedNonNativeStrategy(
    MLIRContext *context, const Field &selectedField, DataFlowSolver &dataflowSolver,
    const StructIntervals *intervals
)
    : reasoner(selectedField, dataflowSolver, intervals),
      emitter(std::make_unique<SMTIntTheoryEmitter>(context, reasoner)) {}

bool OptimizedNonNativeStrategy::isScalarFeltType(Type type) const {
  return reasoner.isScalarFeltType(type);
}

UnreducedInterval OptimizedNonNativeStrategy::getDefaultFeltRange() const {
  return reasoner.getDefaultFeltRange();
}

UnreducedInterval OptimizedNonNativeStrategy::getScalarValueRange(Value value) const {
  return reasoner.getScalarValueRange(value);
}

UnreducedInterval OptimizedNonNativeStrategy::getConstraintMemberRange(StringRef memberName) const {
  return reasoner.getConstraintMemberRange(memberName);
}

UnreducedInterval OptimizedNonNativeStrategy::getWitnessMemberRange(StringRef memberName) const {
  return reasoner.getWitnessMemberRange(memberName);
}

void OptimizedNonNativeStrategy::emitRangeConstraint(
    OpBuilder &builder, Location loc, Value value, const UnreducedInterval &range
) const {
  emitter->emitRangeConstraint(builder, loc, value, range);
}

bool OptimizedNonNativeStrategy::maybeContainsZeroResidue(const UnreducedInterval &range) const {
  return reasoner.maybeContainsZeroResidue(range);
}

bool OptimizedNonNativeStrategy::spansModulusBoundary(const UnreducedInterval &range) const {
  return reasoner.spansModulusBoundary(range);
}

bool OptimizedNonNativeStrategy::sameResidueWindow(
    const UnreducedInterval &lhs, const UnreducedInterval &rhs
) const {
  return reasoner.sameResidueWindow(lhs, rhs);
}

Value OptimizedNonNativeStrategy::canonicalizeValue(
    OpBuilder &builder, Location loc, Value value, const UnreducedInterval &range, StringRef prefix
) const {
  // Canonicalization produces the representative in [0, p-1]. We keep the
  // value unchanged when it is already canonical, otherwise either use native
  // `mod p` or materialize quotient/remainder witnesses depending on the
  // range-derived reduction plan.
  auto plan = reasoner.planCanonicalization(range);
  switch (plan.kind) {
  case ModularReasoner::ReductionKind::Direct:
    return value;
  case ModularReasoner::ReductionKind::NativeMod:
    return emitter->emitModPrime(builder, loc, value);
  case ModularReasoner::ReductionKind::ExplicitWitness: {
    assert(plan.quotientRange.has_value() && "explicit canonicalization requires quotient range");
    std::string quotientName = (prefix + Twine("_q")).str();
    std::string remainderName = (prefix + Twine("_n")).str();
    Value quotient = emitter->emitFreshSymbol(builder, loc, quotientName);
    Value remainder = emitter->emitFreshSymbol(builder, loc, remainderName);
    emitRangeConstraint(builder, loc, quotient, *plan.quotientRange);
    emitRangeConstraint(builder, loc, remainder, getDefaultFeltRange());
    Value qTimesP = emitter->emitPrimeMultiple(builder, loc, quotient);
    Value reconstructed = emitter->emitAdd(builder, loc, qTimesP, remainder);
    Value eq = builder.create<smt::EqOp>(loc, value, reconstructed).getResult();
    // Assert value = q * p + r, where r is the canonical representative in [0, p-1].
    builder.create<smt::AssertOp>(loc, eq);
    return remainder;
  }
  }
  llvm_unreachable("unknown reduction kind");
}

Value OptimizedNonNativeStrategy::buildCanonicalEqualityPredicate(
    OpBuilder &builder, Location loc, Value lhs, const UnreducedInterval &lhsRange, Value rhs,
    const UnreducedInterval &rhsRange, StringRef prefix
) const {
  if (reasoner.unionWidthLessThanPrime(lhsRange, rhsRange)) {
    return builder.create<smt::EqOp>(loc, lhs, rhs).getResult();
  }

  Value lhsCanonical =
      canonicalizeValue(builder, loc, lhs, lhsRange, (prefix + Twine("_lhs")).str());
  Value rhsCanonical =
      canonicalizeValue(builder, loc, rhs, rhsRange, (prefix + Twine("_rhs")).str());
  return builder.create<smt::EqOp>(loc, lhsCanonical, rhsCanonical).getResult();
}

Value OptimizedNonNativeStrategy::buildCongruenceEqualityPredicate(
    OpBuilder &builder, Location loc, Value lhs, const UnreducedInterval &lhsRange, Value rhs,
    const UnreducedInterval &rhsRange, StringRef prefix
) const {
  // Congruence lowering starts from lhs - rhs ≡ 0 (mod p). When the two ranges
  // are narrow enough we can compare directly; otherwise we either keep a
  // native `mod p` constraint or introduce an explicit quotient witness.
  auto plan = reasoner.planCongruence(lhsRange, rhsRange);
  if (plan.kind == ModularReasoner::ReductionKind::Direct) {
    return builder.create<smt::EqOp>(loc, lhs, rhs).getResult();
  }

  Value diff = emitter->emitSub(builder, loc, lhs, rhs);
  switch (plan.kind) {
  case ModularReasoner::ReductionKind::NativeMod: {
    Value zero = emitter->emitConstant(builder, loc, llvm::DynamicAPInt(0));
    Value reducedDiff = emitter->emitModPrime(builder, loc, diff);
    return builder.create<smt::EqOp>(loc, reducedDiff, zero).getResult();
  }
  case ModularReasoner::ReductionKind::ExplicitWitness: {
    assert(plan.quotientRange.has_value() && "explicit congruence requires quotient range");
    std::string quotientName = (prefix + Twine("_q")).str();
    Value quotient = emitter->emitFreshSymbol(builder, loc, quotientName);
    emitRangeConstraint(builder, loc, quotient, *plan.quotientRange);
    Value qTimesP = emitter->emitPrimeMultiple(builder, loc, quotient);
    return builder.create<smt::EqOp>(loc, diff, qTimesP).getResult();
  }
  case ModularReasoner::ReductionKind::Direct:
    llvm_unreachable("direct reductions handled above");
  }
  llvm_unreachable("unknown reduction kind");
}

Value OptimizedNonNativeStrategy::emitOrderedComparisonPredicate(
    OpBuilder &builder, Location loc, boolean::FeltCmpPredicate predicate, Value lhs,
    const UnreducedInterval &lhsRange, Value rhs, const UnreducedInterval &rhsRange,
    StringRef prefix
) const {
  // Ordered field comparisons are only safe to interpret directly when both
  // operands stay within the same residue window. Otherwise compare the
  // canonical representatives instead.
  Value lhsToCompare = lhs;
  Value rhsToCompare = rhs;
  if (spansModulusBoundary(lhsRange) || spansModulusBoundary(rhsRange) ||
      !sameResidueWindow(lhsRange, rhsRange)) {
    lhsToCompare = canonicalizeValue(builder, loc, lhs, lhsRange, (prefix + Twine("_lhs")).str());
    rhsToCompare = canonicalizeValue(builder, loc, rhs, rhsRange, (prefix + Twine("_rhs")).str());
  }

  return emitter->emitOrderedComparison(builder, loc, predicate, lhsToCompare, rhsToCompare);
}

void OptimizedNonNativeStrategy::emitCongruenceEqualityAssertion(
    OpBuilder &builder, Location loc, Value lhs, const UnreducedInterval &lhsRange, Value rhs,
    const UnreducedInterval &rhsRange, StringRef prefix
) const {
  Value predicate =
      buildCongruenceEqualityPredicate(builder, loc, lhs, lhsRange, rhs, rhsRange, prefix);
  // Assert lhs ≡ rhs (mod p) using the selected congruence encoding plan.
  builder.create<smt::AssertOp>(loc, predicate);
}

Value OptimizedNonNativeStrategy::emitDivisionValue(
    OpBuilder &builder, Location loc, Value numerator, const UnreducedInterval &numeratorRange,
    Value denominator, const UnreducedInterval &denominatorRange,
    const UnreducedInterval &resultRange
) const {
  // Note: this lowering inherits the current inverse convention used by the
  // non-native encoding task notes and interval reasoning: a zero denominator
  // forces the division result to zero. That is in some tension with the felt
  // dialect documentation, which says the divisor must be non-zero.
  // `felt.div x y` is encoded by a fresh witness `d` satisfying y * d ≡ x
  // (mod p). If the denominator may be zero modulo p, we preserve the LLZK
  // semantics with a branch that forces the result to zero in that case.
  Value div = emitter->emitFreshSymbol(builder, loc, "div");
  UnreducedInterval zeroRange(0, 0);
  emitRangeConstraint(builder, loc, div, resultRange);

  Value zero = emitter->emitConstant(builder, loc, llvm::DynamicAPInt(0));
  Value product = emitter->emitMul(builder, loc, denominator, div);
  UnreducedInterval productRange = denominatorRange * resultRange;
  Value productEqualsNumerator = buildCongruenceEqualityPredicate(
      builder, loc, product, productRange, numerator, numeratorRange, "felt_div"
  );

  if (!maybeContainsZeroResidue(denominatorRange)) {
    // Assert denominator * div ≡ numerator (mod p) when the denominator is provably nonzero.
    builder.create<smt::AssertOp>(loc, productEqualsNumerator);
    return div;
  }

  Value denominatorIsZero = buildCanonicalEqualityPredicate(
      builder, loc, denominator, denominatorRange, zero, zeroRange, "felt_div_denom_zero"
  );
  Value divIsZero = buildCanonicalEqualityPredicate(
      builder, loc, div, resultRange, zero, zeroRange, "felt_div_result_zero"
  );
  // Build `ite denominator == 0 then div = 0 else denominator * div ≡ numerator (mod p)`.
  Value divConstraint =
      builder.create<smt::IteOp>(loc, denominatorIsZero, divIsZero, productEqualsNumerator)
          .getResult();
  // Assert the LLZK field-division semantics with an explicit zero-denominator branch.
  builder.create<smt::AssertOp>(loc, divConstraint);
  return div;
}

Value OptimizedNonNativeStrategy::emitInverseValue(
    OpBuilder &builder, Location loc, Value operand, const UnreducedInterval &operandRange
) const {
  // Note: the current lowering treats `inv(0)` as 0 and otherwise constrains
  // `x * inv(x) ≡ 1 (mod p)`. This matches the present analysis/task
  // convention, but it is in tension with the felt dialect docs that require
  // non-zero divisors for field division.
  // `felt.inv x` is the special case x * inv ≡ 1 (mod p), again with the LLZK
  // zero-denominator convention that a zero residue forces the result to zero.
  Value inv = emitter->emitFreshSymbol(builder, loc, "inv");
  UnreducedInterval invRange = getDefaultFeltRange();
  UnreducedInterval zeroRange(0, 0);
  UnreducedInterval oneRange(1, 1);
  emitRangeConstraint(builder, loc, inv, invRange);

  Value zero = emitter->emitConstant(builder, loc, llvm::DynamicAPInt(0));
  Value one = emitter->emitConstant(builder, loc, llvm::DynamicAPInt(1));
  Value product = emitter->emitMul(builder, loc, operand, inv);
  UnreducedInterval productRange = operandRange * invRange;
  Value productEqualsOne = buildCongruenceEqualityPredicate(
      builder, loc, product, productRange, one, oneRange, "felt_inv"
  );

  if (!maybeContainsZeroResidue(operandRange)) {
    // Assert operand * inv ≡ 1 (mod p) when the operand is provably nonzero.
    builder.create<smt::AssertOp>(loc, productEqualsOne);
    return inv;
  }

  Value operandIsZero = buildCanonicalEqualityPredicate(
      builder, loc, operand, operandRange, zero, zeroRange, "felt_inv_operand_zero"
  );
  Value invIsZero = buildCanonicalEqualityPredicate(
      builder, loc, inv, invRange, zero, zeroRange, "felt_inv_result_zero"
  );
  // Build `ite operand == 0 then inv = 0 else operand * inv ≡ 1 (mod p)`.
  Value invConstraint =
      builder.create<smt::IteOp>(loc, operandIsZero, invIsZero, productEqualsOne).getResult();
  // Assert the LLZK inverse semantics with an explicit zero-operand branch.
  builder.create<smt::AssertOp>(loc, invConstraint);
  return inv;
}

void OptimizedNonNativeStrategy::populatePatterns(
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
      BasicConverter<felt::SignedModFeltOp, smt::IntModOp>, FeltConstConverter, ReturnConverter,
      SCFIfConverter, YieldConverter>(typeConverter, context);
  patterns.add<FunctionDefConverter>(typeConverter, context);
  patterns.add<BoolCmpConverter>(typeConverter, context, this);
  patterns.add<FeltDivConverter>(typeConverter, context, this);
  patterns.add<FeltInvConverter>(typeConverter, context, this);
  patterns.add<ConstrainConverter>(typeConverter, context, this);
  patterns.add<MemberWriteConverter>(typeConverter, context, signalSymbols, this);
  patterns.add<MemberReadConverter>(typeConverter, context, signalSymbols);
}

class SMTOptimizedNonNativeLoweringPass
    : public smt::impl::SMTLoweringPassBase<SMTOptimizedNonNativeLoweringPass> {

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<smt::SMTDialect, mlir::func::FuncDialect>();
  }

  // Convert the body and signature of a @product function to SMT
  Operation *convertBodies(
      Operation *op, const SignalSymbols &signalSymbols, const OptimizedNonNativeStrategy &strategy
  ) {
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

    auto &mia = getAnalysis<ModuleIntervalAnalysis>();
    mia.setField(*selectedField);
    mia.setTrackUnreducedIntervals(true);
    auto am = getAnalysisManager();
    mia.ensureAnalysisRun(am);

    auto result =
        mod.walk([this, &mia, prime, &selectedField, &mod](component::StructDefOp structDef) {
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
      Operation *symbolTableOp = structDef->getParentOp();

      // Start by adding declare-funcs for each felt signal member.
      IRRewriter rewriter {&getContext()};
      rewriter.setInsertionPointToStart(&productFunc.getFunctionBody().front());

      auto preamble = productFunc->getLoc();
      const StructIntervals *intervals =
          mia.hasResult(structDef) ? &mia.getResult(structDef) : nullptr;
      OptimizedNonNativeStrategy strategy {
          &getContext(), selectedField->get(), mia.getSolver(), intervals
      };
      SmallVector<std::optional<UnreducedInterval>> productArgRanges;
      productArgRanges.reserve(productFunc.getNumArguments());
      for (auto [arg, type] :
           llvm::zip(productFunc.getArguments(), productFunc.getArgumentTypes())) {
        if (isa<felt::FeltType>(type)) {
          productArgRanges.emplace_back(strategy.getScalarValueRange(arg));
        } else {
          productArgRanges.emplace_back(std::nullopt);
        }
      }

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
        strategy.emitRangeConstraint(
            rewriter, memberDef.getLoc(), constraintSym.getResult(),
            strategy.getConstraintMemberRange(memberDef.getSymName())
        );
        strategy.emitRangeConstraint(
            rewriter, memberDef.getLoc(), witnessSym.getResult(),
            strategy.getWitnessMemberRange(memberDef.getSymName())
        );
        signalSymbols[memberDef.getSymName()] = {constraintSym.getResult(), witnessSym.getResult()};
      }

      std::string smtFuncName = "smt_" + structDef.getSymName().str();
      auto *result = convertStructProductToFunc(
          convertBodies(structDef, signalSymbols, strategy), &getContext()
      );
      if (result == nullptr) {
        signalPassFailure();
        return WalkResult::interrupt();
      }

      auto smtFunc =
          dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupSymbolIn(symbolTableOp, smtFuncName));
      if (!smtFunc) {
        mod.emitError() << "failed to locate lowered SMT function \"" << smtFuncName << "\"";
        signalPassFailure();
        return WalkResult::interrupt();
      }

      IRRewriter argRewriter {&getContext()};
      argRewriter.setInsertionPointToStart(&smtFunc.getBody().front());
      for (auto [idx, maybeRange] : llvm::enumerate(productArgRanges)) {
        if (!maybeRange.has_value()) {
          continue;
        }
        strategy.emitRangeConstraint(
            argRewriter, smtFunc.getLoc(), smtFunc.getArgument(idx), *maybeRange
        );
      }

      return WalkResult::advance();
    });
    if (!result.wasInterrupted()) {
      // Remove `llzk.main` attribute because `convertFunction()` above deleted structs.
      mod->removeAttr(MAIN_ATTR_NAME);
    }
  }
};

namespace smt {
std::unique_ptr<mlir::Pass> createSMTLoweringPass() {
  return std::make_unique<SMTOptimizedNonNativeLoweringPass>();
}
} // namespace smt

} // namespace llzk
