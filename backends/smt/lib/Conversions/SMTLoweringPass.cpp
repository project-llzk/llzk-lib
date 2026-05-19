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

using SignalSymbols = DenseMap<StringRef, std::pair<Value, Value>>;

namespace {

class RangeOptimizedSMTEncoding {
public:
  static constexpr uint64_t explicitReductionQuotientThreshold = 1000;

  RangeOptimizedSMTEncoding(
      MLIRContext *context, const Field &selectedField, llvm::APSInt fieldPrime,
      DataFlowSolver &dataflowSolver, const StructIntervals *intervals = nullptr
  )
      : ctx(context), field(selectedField), prime(std::move(fieldPrime)),
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

  void emitRangeConstraint(OpBuilder &builder, Location loc, Value value,
                           const UnreducedInterval &range) const {
    auto lower = createIntConstant(builder, loc, range.getLHS());
    auto upper = createIntConstant(builder, loc, range.getRHS());
    auto lowerBound = builder.create<smt::IntCmpOp>(
        loc, smt::IntPredicate::ge, value, lower.getResult()
    );
    auto upperBound = builder.create<smt::IntCmpOp>(
        loc, smt::IntPredicate::le, value, upper.getResult()
    );
    builder.create<smt::AssertOp>(loc, lowerBound.getResult());
    builder.create<smt::AssertOp>(loc, upperBound.getResult());
  }

  Value canonicalizeValue(OpBuilder &builder, Location loc, Value value,
                          const UnreducedInterval &range, StringRef prefix) const {
    if (isCanonical(range)) {
      return value;
    }

    if (!shouldUseExplicitReduction(range)) {
      return buildModPrimeExpr(builder, loc, value);
    }

    auto quotientRange = getQuotientRange(range);
    auto quotient = builder.create<smt::DeclareFunOp>(
        loc, smt::IntType::get(ctx), StringAttr::get(ctx, makeName(prefix, "_q"))
    );
    auto reduced = builder.create<smt::DeclareFunOp>(
        loc, smt::IntType::get(ctx), StringAttr::get(ctx, makeName(prefix, "_n"))
    );
    emitRangeConstraint(builder, loc, quotient.getResult(), quotientRange);
    emitRangeConstraint(builder, loc, reduced.getResult(), getDefaultFeltRange());

    auto primeConst = createPrimeConstant(builder, loc);
    auto quotientTimesPrime =
        builder.create<smt::IntMulOp>(loc, ValueRange {quotient.getResult(), primeConst.getResult()});
    auto reconstructed =
        builder.create<smt::IntAddOp>(loc, ValueRange {quotientTimesPrime.getResult(),
                                                       reduced.getResult()});
    auto eq = builder.create<smt::EqOp>(loc, value, reconstructed.getResult());
    builder.create<smt::AssertOp>(loc, eq.getResult());
    return reduced.getResult();
  }

  Value buildCanonicalEqualityPredicate(OpBuilder &builder, Location loc, Value lhs,
                                        const UnreducedInterval &lhsRange, Value rhs,
                                        const UnreducedInterval &rhsRange,
                                        StringRef prefix) const {
    if (unionWidthLessThanPrime(lhsRange, rhsRange)) {
      return builder.create<smt::EqOp>(loc, lhs, rhs).getResult();
    }

    Value lhsCanonical =
        canonicalizeValue(builder, loc, lhs, lhsRange, makeName(prefix, "_lhs"));
    Value rhsCanonical =
        canonicalizeValue(builder, loc, rhs, rhsRange, makeName(prefix, "_rhs"));
    return builder.create<smt::EqOp>(loc, lhsCanonical, rhsCanonical).getResult();
  }

  Value buildCongruenceEqualityPredicate(OpBuilder &builder, Location loc, Value lhs,
                                         const UnreducedInterval &lhsRange, Value rhs,
                                         const UnreducedInterval &rhsRange,
                                         StringRef prefix) const {
    if (explicitReductionQuotientThreshold > 0 &&
        unionWidthLessThanPrime(lhsRange, rhsRange)) {
      return builder.create<smt::EqOp>(loc, lhs, rhs).getResult();
    }

    auto diffRange = lhsRange - rhsRange;
    auto diff = builder.create<smt::IntSubOp>(loc, lhs, rhs);
    if (!shouldUseExplicitReduction(diffRange)) {
      auto reducedDiff = buildModPrimeExpr(builder, loc, diff.getResult());
      auto zero = createIntConstant(builder, loc, llvm::DynamicAPInt(0));
      return builder.create<smt::EqOp>(loc, reducedDiff, zero.getResult()).getResult();
    }

    auto quotientRange = getQuotientRange(diffRange);
    auto quotient = builder.create<smt::DeclareFunOp>(
        loc, smt::IntType::get(ctx), StringAttr::get(ctx, makeName(prefix, "_q"))
    );
    emitRangeConstraint(builder, loc, quotient.getResult(), quotientRange);

    auto primeConst = createPrimeConstant(builder, loc);
    auto quotientTimesPrime =
        builder.create<smt::IntMulOp>(loc, ValueRange {quotient.getResult(), primeConst.getResult()});
    return builder.create<smt::EqOp>(loc, diff.getResult(), quotientTimesPrime.getResult())
        .getResult();
  }

  void emitCongruenceEqualityAssertion(OpBuilder &builder, Location loc, Value lhs,
                                       const UnreducedInterval &lhsRange, Value rhs,
                                       const UnreducedInterval &rhsRange,
                                       StringRef prefix) const {
    Value predicate =
        buildCongruenceEqualityPredicate(builder, loc, lhs, lhsRange, rhs, rhsRange, prefix);
    builder.create<smt::AssertOp>(loc, predicate);
  }

private:
  MLIRContext *ctx;
  std::reference_wrapper<const Field> field;
  llvm::APSInt prime;
  llvm::DynamicAPInt primeDynamic;
  DataFlowSolver &solver;
  llvm::StringMap<UnreducedInterval> witnessRanges;
  llvm::StringMap<UnreducedInterval> constraintRanges;

  static bool isDirectMemberRef(const SourceRef &ref) {
    return ref.isRooted() && ref.getPath().size() == 1 && ref.getPath().front().isMember();
  }

  void captureMemberRanges(const llvm::MapVector<SourceRef, Interval> &reducedRanges,
                           llvm::StringMap<UnreducedInterval> &out) {
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

  UnreducedInterval lookupMemberRange(const llvm::StringMap<UnreducedInterval> &ranges,
                                      StringRef memberName) const {
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

  smt::IntConstantOp createPrimeConstant(OpBuilder &builder, Location loc) const {
    return builder.create<smt::IntConstantOp>(loc, IntegerAttr::get(ctx, prime));
  }

  Value buildModPrimeExpr(OpBuilder &builder, Location loc, Value value) const {
    auto primeConst = createPrimeConstant(builder, loc);
    return builder.create<smt::IntModOp>(loc, ValueRange {value, primeConst.getResult()})
        .getResult();
  }

  smt::IntConstantOp createIntConstant(OpBuilder &builder, Location loc,
                                       const llvm::DynamicAPInt &value) const {
    return builder.create<smt::IntConstantOp>(loc, IntegerAttr::get(ctx, toAPSInt(value)));
  }
};

} // namespace

class LLZKToSMTTypeConverter : public TypeConverter {
public:
  LLZKToSMTTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](mlir::IntegerType type) -> Type {
      if (type.isSignless() && type.getWidth() == 1) {
        return smt::BoolType::get(ctx);
      }
      return type;
    });
    addConversion([ctx](felt::FeltType) { return smt::IntType::get(ctx); });
  }
};

static inline bool containsFeltOrStruct(Type type) {
  return isa<component::StructType>(type) ||
         TypeSwitch<Type, bool>(type)
             .Case<felt::FeltType>([](auto) { return true; })
             .Case<array::ArrayType>([](array::ArrayType arrayType) {
    return containsFeltOrStruct(arrayType.getElementType());
  }).Default([](auto) { return false; });
}

static LogicalResult validateSupportedSMTMemberAccesses(component::StructDefOp structDef) {
  auto productFunc = structDef.getProductFuncOp();
  if (!productFunc) {
    return success();
  }

  WalkResult walkResult = productFunc.walk([&](Operation *op) -> WalkResult {
    if (auto memberWrite = dyn_cast<component::MemberWriteOp>(op)) {
      if (!isa<felt::FeltType>(memberWrite.getVal().getType())) {
        memberWrite.emitError("SMT lowering currently only supports felt-valued struct.writem");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }

    if (auto memberRead = dyn_cast<component::MemberReadOp>(op)) {
      if (!isa<felt::FeltType>(memberRead.getResult().getType())) {
        memberRead.emitError("SMT lowering currently only supports felt-valued struct.readm");
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });

  return walkResult.wasInterrupted() ? failure() : success();
}

// Define OpConversions
template <class From, class To> class BasicConverter : public OpConversionPattern<From> {
  using OpConversionPattern<From>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      From fromOp, typename From::Adaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    rewriter.replaceOpWithNewOp<To>(fromOp, adaptor.getOperands());

    return success();
  }
};

class FunctionDefConverter : public OpConversionPattern<function::FuncDefOp> {
  using OpConversionPattern<function::FuncDefOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      function::FuncDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter
  ) const override {
    // Convert the signature
    SmallVector<Type> convertedArgTypes =
        llvm::map_to_vector(op.getArgumentTypes(), [this](Type t) {
      return getTypeConverter()->convertType(t);
    });
    SmallVector<Type> convertedResultTypes = llvm::map_to_vector(
        llvm::filter_to_vector(
            op.getResultTypes(), [](Type t) { return !isa<component::StructType>(t); }
        ),
        [this](Type t) { return getTypeConverter()->convertType(t); }
    );

    auto newType = op.getFunctionType().clone(convertedArgTypes, convertedResultTypes);
    op.setFunctionType(newType);

    auto &block = op.getBlocks().front();
    auto signatureConversion = getTypeConverter()->convertBlockSignature(&block);
    if (signatureConversion.has_value()) {
      rewriter.applySignatureConversion(&block, *signatureConversion);
    } else {
      return failure();
    }

    return success();
  }
};

class FeltDivConverter : public OpConversionPattern<felt::DivFeltOp> {
public:
  FeltDivConverter(
      TypeConverter &typeConverter, MLIRContext *context,
      const RangeOptimizedSMTEncoding *rangeEncoding
  )
      : OpConversionPattern<felt::DivFeltOp>(typeConverter, context, /*benefit=*/2),
        encoding(rangeEncoding) {}

  LogicalResult matchAndRewrite(
      felt::DivFeltOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    auto div = rewriter.create<smt::DeclareFunOp>(
        op->getLoc(), smt::IntType::get(getContext()), StringAttr::get(getContext(), "div")
    );
    auto divRange = encoding->getScalarValueRange(op.getResult());
    auto lhsRange = encoding->getScalarValueRange(op.getLhs());
    auto rhsRange = encoding->getScalarValueRange(op.getRhs());
    auto zeroRange = UnreducedInterval(0, 0);
    encoding->emitRangeConstraint(rewriter, op.getLoc(), div.getResult(), divRange);

    auto zero = rewriter.create<smt::IntConstantOp>(
        op->getLoc(), IntegerAttr::get(getContext(), APSInt {APInt {1, 0}})
    );
    auto product = rewriter.create<smt::IntMulOp>(
        op->getLoc(), ValueRange {adaptor.getRhs(), div.getResult()}
    );
    auto productRange = rhsRange * divRange;
    auto productEqualsNumerator = encoding->buildCongruenceEqualityPredicate(
        rewriter, op.getLoc(), product.getResult(), productRange, adaptor.getLhs(), lhsRange,
        "felt_div"
    );

    if (!encoding->maybeContainsZeroResidue(rhsRange)) {
      rewriter.create<smt::AssertOp>(op->getLoc(), productEqualsNumerator);
      rewriter.replaceOp(op, div.getResult());
      return success();
    }

    auto denominatorIsZero = encoding->buildCanonicalEqualityPredicate(
        rewriter, op.getLoc(), adaptor.getRhs(), rhsRange, zero.getResult(), zeroRange,
        "felt_div_denom_zero"
    );
    auto divIsZero = encoding->buildCanonicalEqualityPredicate(
        rewriter, op.getLoc(), div.getResult(), divRange, zero.getResult(), zeroRange,
        "felt_div_result_zero"
    );
    auto divConstraint = rewriter.create<smt::IteOp>(
        op->getLoc(), denominatorIsZero, divIsZero, productEqualsNumerator
    );
    rewriter.create<smt::AssertOp>(op->getLoc(), divConstraint.getResult());
    rewriter.replaceOp(op, div.getResult());

    return success();
  }

private:
  const RangeOptimizedSMTEncoding *encoding;
};

class FeltInvConverter : public OpConversionPattern<felt::InvFeltOp> {
public:
  FeltInvConverter(
      TypeConverter &typeConverter, MLIRContext *context,
      const RangeOptimizedSMTEncoding *rangeEncoding
  )
      : OpConversionPattern<felt::InvFeltOp>(typeConverter, context, /*benefit=*/2),
        encoding(rangeEncoding) {}

  LogicalResult matchAndRewrite(
      felt::InvFeltOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    auto inv = rewriter.create<smt::DeclareFunOp>(
        op->getLoc(), smt::IntType::get(getContext()), StringAttr::get(getContext(), "inv")
    );
    auto invRange = encoding->getDefaultFeltRange();
    auto operandRange = encoding->getScalarValueRange(op.getOperand());
    auto zeroRange = UnreducedInterval(0, 0);
    auto oneRange = UnreducedInterval(1, 1);
    encoding->emitRangeConstraint(rewriter, op->getLoc(), inv.getResult(), invRange);

    auto zero = rewriter.create<smt::IntConstantOp>(
        op->getLoc(), IntegerAttr::get(getContext(), APSInt {APInt {1, 0}})
    );
    auto one = rewriter.create<smt::IntConstantOp>(
        op->getLoc(),
        IntegerAttr::get(getContext(), APSInt(llvm::APInt(64, 1), /*isUnsigned=*/false))
    );
    auto product = rewriter.create<smt::IntMulOp>(
        op->getLoc(), ValueRange {adaptor.getOperand(), inv.getResult()}
    );
    auto productRange = operandRange * invRange;
    auto productEqualsOne = encoding->buildCongruenceEqualityPredicate(
        rewriter, op->getLoc(), product.getResult(), productRange, one.getResult(), oneRange,
        "felt_inv"
    );

    if (!encoding->maybeContainsZeroResidue(operandRange)) {
      rewriter.create<smt::AssertOp>(op->getLoc(), productEqualsOne);
      rewriter.replaceOp(op, inv.getResult());
      return success();
    }

    auto operandIsZero = encoding->buildCanonicalEqualityPredicate(
        rewriter, op->getLoc(), adaptor.getOperand(), operandRange, zero.getResult(), zeroRange,
        "felt_inv_operand_zero"
    );
    auto invIsZero = encoding->buildCanonicalEqualityPredicate(
        rewriter, op->getLoc(), inv.getResult(), invRange, zero.getResult(), zeroRange,
        "felt_inv_result_zero"
    );
    auto invConstraint = rewriter.create<smt::IteOp>(
        op->getLoc(), operandIsZero, invIsZero, productEqualsOne
    );
    rewriter.create<smt::AssertOp>(op->getLoc(), invConstraint.getResult());
    rewriter.replaceOp(op, inv.getResult());

    return success();
  }

private:
  const RangeOptimizedSMTEncoding *encoding;
};

class MemberWriteConverter : public OpConversionPattern<component::MemberWriteOp> {
public:
  MemberWriteConverter(
      TypeConverter &typeConverter, MLIRContext *context, const SignalSymbols &signalMap,
      const RangeOptimizedSMTEncoding *rangeEncoding
  )
      : OpConversionPattern<component::MemberWriteOp>(typeConverter, context, /*benefit=*/2),
        symbols(signalMap), encoding(rangeEncoding) {}

  LogicalResult matchAndRewrite(
      component::MemberWriteOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (!encoding->isScalarFeltType(op.getVal().getType())) {
      op.emitError("SMT lowering currently only supports felt-valued struct.writem");
      return failure();
    }

    auto it = symbols.find(adaptor.getMemberName());
    if (it == symbols.end()) {
      return failure();
    }

    auto [_, witness] = it->second;
    auto witnessRange = encoding->getWitnessMemberRange(adaptor.getMemberName());
    auto valueRange = encoding->getScalarValueRange(op.getVal());
    encoding->emitCongruenceEqualityAssertion(
        rewriter, op.getLoc(), witness, witnessRange, adaptor.getVal(), valueRange,
        "member_write"
    );
    rewriter.eraseOp(op);

    return success();
  }

private:
  SignalSymbols symbols;
  const RangeOptimizedSMTEncoding *encoding;
};

class MemberReadConverter : public OpConversionPattern<component::MemberReadOp> {
public:
  MemberReadConverter(
      TypeConverter &typeConverter, MLIRContext *context, const SignalSymbols &signalMap
  )
      : OpConversionPattern<component::MemberReadOp>(typeConverter, context, /*benefit=*/2),
        symbols(signalMap) {}

  LogicalResult matchAndRewrite(
      component::MemberReadOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (!isa<felt::FeltType>(op.getResult().getType())) {
      op.emitError("SMT lowering currently only supports felt-valued struct.readm");
      return failure();
    }

    auto it = symbols.find(adaptor.getMemberName());
    if (it == symbols.end()) {
      return failure();
    }

    auto [constrain, _] = it->second;
    rewriter.replaceOp(op, ValueRange {constrain});

    return success();
  }

private:
  SignalSymbols symbols;
};

class StructDefConverter : public OpConversionPattern<component::StructDefOp> {
  using OpConversionPattern<component::StructDefOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      component::StructDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter
  ) const override {
    // Replace the struct.def with a single mlir func with the signature and body of
    // @struct::@product
    std::string smtFuncName = ("smt_" + op.getSymName()).str();
    auto productFunc = op.getProductFuncOp();
    auto smtFunc =
        rewriter.create<func::FuncOp>(op->getLoc(), smtFuncName, productFunc.getFunctionType());
    IRMapping mapping;
    productFunc.getFunctionBody().cloneInto(&smtFunc.getFunctionBody(), mapping);

    // Replace llzk::function.return with mlir::func.return
    smtFunc.walk([&](function::ReturnOp returnOp) {
      rewriter.setInsertionPoint(returnOp);
      rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp, returnOp.getOperands());
    });

    rewriter.eraseOp(op);
    return success();
  }
};

class ReturnConverter : public OpConversionPattern<function::ReturnOp> {
  using OpConversionPattern<function::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      function::ReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    // Don't return any !struct.type's from the SMT function
    SmallVector<Value> returnedValues;
    for (auto [val, type] : llvm::zip(adaptor.getOperands(), op.getOperandTypes())) {
      if (isa<component::StructType>(type)) {
        continue;
      }
      returnedValues.push_back(val);
    }

    rewriter.modifyOpInPlace(op, [&]() { op.getOperandsMutable().assign(returnedValues); });
    return success();
  }
};

class ConstrainConverter : public OpConversionPattern<constrain::EmitEqualityOp> {
public:
  ConstrainConverter(
      TypeConverter &typeConverter, MLIRContext *context,
      const RangeOptimizedSMTEncoding *rangeEncoding
  )
      : OpConversionPattern<constrain::EmitEqualityOp>(typeConverter, context, /*benefit=*/2),
        encoding(rangeEncoding) {}

  LogicalResult matchAndRewrite(
      constrain::EmitEqualityOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    auto lhsRange = encoding->getScalarValueRange(op.getLhs());
    auto rhsRange = encoding->getScalarValueRange(op.getRhs());
    encoding->emitCongruenceEqualityAssertion(
        rewriter, op.getLoc(), adaptor.getLhs(), lhsRange, adaptor.getRhs(), rhsRange,
        "constrain_eq"
    );
    rewriter.eraseOp(op);
    return success();
  }

private:
  const RangeOptimizedSMTEncoding *encoding;
};

class BoolCmpConverter : public OpConversionPattern<boolean::CmpOp> {
public:
  BoolCmpConverter(
      TypeConverter &typeConverter, MLIRContext *context,
      const RangeOptimizedSMTEncoding *rangeEncoding
  )
      : OpConversionPattern<boolean::CmpOp>(typeConverter, context), encoding(rangeEncoding) {}

  LogicalResult matchAndRewrite(
      boolean::CmpOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    auto lhsRange = encoding->getScalarValueRange(op.getLhs());
    auto rhsRange = encoding->getScalarValueRange(op.getRhs());

    switch (adaptor.getPredicate()) {
    case boolean::FeltCmpPredicate::EQ: {
      auto eq = encoding->buildCanonicalEqualityPredicate(
          rewriter, op.getLoc(), adaptor.getLhs(), lhsRange, adaptor.getRhs(), rhsRange,
          "bool_cmp_eq"
      );
      rewriter.replaceOp(op, eq);
      return success();
    }
    case boolean::FeltCmpPredicate::NE: {
      auto eq = encoding->buildCanonicalEqualityPredicate(
          rewriter, op.getLoc(), adaptor.getLhs(), lhsRange, adaptor.getRhs(), rhsRange,
          "bool_cmp_ne"
      );
      rewriter.replaceOp(op, rewriter.create<smt::NotOp>(op.getLoc(), eq).getResult());
      return success();
    }
    default: {
      static DenseMap<boolean::FeltCmpPredicate, smt::IntPredicate> predicateComparator = {
          {boolean::FeltCmpPredicate::GE, smt::IntPredicate::ge},
          {boolean::FeltCmpPredicate::GT, smt::IntPredicate::gt},
          {boolean::FeltCmpPredicate::LE, smt::IntPredicate::le},
          {boolean::FeltCmpPredicate::LT, smt::IntPredicate::lt}
      };

      Value lhs = adaptor.getLhs();
      Value rhs = adaptor.getRhs();
      if (encoding->spansModulusBoundary(lhsRange) || encoding->spansModulusBoundary(rhsRange) ||
          !encoding->sameResidueWindow(lhsRange, rhsRange)) {
        lhs = encoding->canonicalizeValue(
            rewriter, op.getLoc(), adaptor.getLhs(), lhsRange, "bool_cmp_ordered_lhs"
        );
        rhs = encoding->canonicalizeValue(
            rewriter, op.getLoc(), adaptor.getRhs(), rhsRange, "bool_cmp_ordered_rhs"
        );
      }

      rewriter.replaceOpWithNewOp<smt::IntCmpOp>(
          op, predicateComparator[adaptor.getPredicate()], lhs, rhs
      );
      return success();
    }
    }
  }

private:
  const RangeOptimizedSMTEncoding *encoding;
};

class SCFIfConverter : public OpConversionPattern<scf::IfOp> {
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      scf::IfOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    // Not doing anything interesting here, just convert the result types. A later pass will handle
    // the rest
    SmallVector<Type> convertedResultTypes =
        llvm::map_to_vector(op.getResultTypes(), [this](Type t) {
      return getTypeConverter()->convertType(t);
    });

    Value cond = adaptor.getCondition();
    if (!isa<IntegerType>(cond.getType())) {
      // We have to manually convert the condition type because it might be a block arg instead of
      // coming from a converted op
      cond = rewriter
                 .create<UnrealizedConversionCastOp>(
                     op.getLoc(), TypeRange {rewriter.getI1Type()}, cond
                 )
                 .getResult(0);
    }

    auto convertedIf = rewriter.create<scf::IfOp>(
        op.getLoc(), convertedResultTypes, cond,
        /*addThenBlock=*/false, /*addElseBlock=*/false
    );

    rewriter.inlineRegionBefore(
        op.getThenRegion(), convertedIf.getThenRegion(), convertedIf.getThenRegion().end()
    );
    rewriter.inlineRegionBefore(
        op.getElseRegion(), convertedIf.getElseRegion(), convertedIf.getElseRegion().end()
    );
    rewriter.replaceOp(op, convertedIf);
    return success();
  }
};

class YieldConverter : public OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      scf::YieldOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    // Make sure we're yielding the type-converted results so the scf.if's can have the right type
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getResults());
    return success();
  }
};

class FeltConstConverter : public OpConversionPattern<felt::FeltConstantOp> {
  using OpConversionPattern<felt::FeltConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      felt::FeltConstantOp op, OpAdaptor, ConversionPatternRewriter &rewriter
  ) const override {
    rewriter.replaceOpWithNewOp<smt::IntConstantOp>(
        op, IntegerAttr::get(getContext(), APSInt {op.getValue().getValue()})
    );
    return success();
  }
};

class SMTLoweringPass : public smt::impl::SMTLoweringPassBase<SMTLoweringPass> {

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<smt::SMTDialect, mlir::func::FuncDialect>();
  }

  // Hoist a @product function.def inside a struct.def to a free MLIR func.func
  Operation *convertFunction(Operation *op) {
    if (op == nullptr) {
      return op;
    }
    MLIRContext *context = &getContext();

    LLZKToSMTTypeConverter typeConverter {context};
    RewritePatternSet patterns {context};
    ConversionTarget target {*context};

    target.addIllegalOp<component::StructDefOp>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalOp<func::FuncOp>();

    patterns.add<StructDefConverter>(typeConverter, context);

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      return nullptr;
    }

    return op;
  }

  // Convert the body and signature of a @product function to SMT
  Operation *convertBodies(Operation *op, const SignalSymbols &signalSymbols,
                           const RangeOptimizedSMTEncoding *encoding) {
    if (op == nullptr) {
      return op;
    }

    MLIRContext *context = &getContext();

    LLZKToSMTTypeConverter typeConverter {context};
    RewritePatternSet patterns {context};
    ConversionTarget target {*context};

    target.addIllegalDialect<felt::FeltDialect>();
    target.addIllegalDialect<constrain::ConstrainDialect>();
    target.addLegalDialect<smt::SMTDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalOp<component::MemberWriteOp, component::MemberReadOp>();
    target.addLegalOp<component::CreateStructOp>();
    target.addDynamicallyLegalOp<function::ReturnOp>([](function::ReturnOp returnOp) {
      return llvm::none_of(returnOp.getOperandTypes(), [](Type type) {
        return isa<component::StructType>(type);
      });
    });

    target.addDynamicallyLegalOp<function::FuncDefOp>([](function::FuncDefOp funcOp) {
      bool signatureLegal = llvm::none_of(funcOp.getArgumentTypes(), containsFeltOrStruct) &&
                            llvm::none_of(funcOp.getResultTypes(), containsFeltOrStruct);

      return signatureLegal;
    });
    target.addDynamicallyLegalOp<scf::YieldOp>([](scf::YieldOp yieldOp) {
      return llvm::none_of(yieldOp.getOperandTypes(), containsFeltOrStruct);
    });
    target.addDynamicallyLegalOp<scf::IfOp>([](scf::IfOp ifOp) {
      return llvm::none_of(ifOp.getResultTypes(), containsFeltOrStruct);
    });

    patterns.add<
        BasicConverter<felt::AddFeltOp, smt::IntAddOp>,
        BasicConverter<felt::SubFeltOp, smt::IntSubOp>,
        BasicConverter<felt::MulFeltOp, smt::IntMulOp>,
        BasicConverter<felt::NegFeltOp, smt::IntNegOp>,
        BasicConverter<felt::SignedIntDivFeltOp, smt::IntDivOp>,
        BasicConverter<felt::UnsignedModFeltOp, smt::IntModOp>,
        BasicConverter<felt::SignedModFeltOp, smt::IntModOp>, FeltConstConverter,
        ReturnConverter, SCFIfConverter, YieldConverter>(typeConverter, context);
    patterns.add<FunctionDefConverter>(typeConverter, context);
    patterns.add<BoolCmpConverter>(typeConverter, context, encoding);
    patterns.add<FeltDivConverter>(typeConverter, context, encoding);
    patterns.add<FeltInvConverter>(typeConverter, context, encoding);
    patterns.add<ConstrainConverter>(typeConverter, context, encoding);
    patterns.add<MemberWriteConverter>(typeConverter, context, signalSymbols, encoding);
    patterns.add<MemberReadConverter>(typeConverter, context, signalSymbols);

    ConversionConfig config;
    config.buildMaterializations = false;
    if (failed(applyPartialConversion(op, target, std::move(patterns), config))) {
      return nullptr;
    }

    SmallVector<component::CreateStructOp> deadStructs;
    op->walk([&](component::CreateStructOp createStructOp) {
      if (createStructOp->use_empty()) {
        deadStructs.push_back(createStructOp);
      }
    });
    for (component::CreateStructOp createStructOp : deadStructs) {
      createStructOp->erase();
    }

    return op;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    FieldSet fields;
    if (!fieldName.empty()) {
      auto fieldLookupResult = Field::tryGetField(fieldName);
      if (failed(fieldLookupResult)) {
        mod.emitError() << "unknown field \"" << fieldName << "\"";
        return signalPassFailure();
      }
      fields.insert(fieldLookupResult.value());
    }

    // Ignore failure; if we found no fields that will be handled later
    (void)collectFields(mod, fields);

    if (fields.empty()) {
      mod.emitError() << "no prime field specified; could not deduce";
      return signalPassFailure();
    }

    if (fields.size() > 1) {
      mod.emitError() << "multiple fields unsupported";
      return signalPassFailure();
    }

    auto selectedField = *(fields.begin());
    auto prime = toAPSInt(selectedField.get().prime());

    auto &mia = getAnalysis<ModuleIntervalAnalysis>();
    mia.setField(selectedField);
    mia.setTrackUnreducedIntervals(true);
    auto am = getAnalysisManager();
    mia.ensureAnalysisRun(am);

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
      const StructIntervals *intervals = mia.hasResult(structDef) ? &mia.getResult(structDef)
                                                                  : nullptr;
      RangeOptimizedSMTEncoding encoding {
          &getContext(), selectedField.get(), prime, mia.getSolver(), intervals};
      SmallVector<std::optional<UnreducedInterval>> productArgRanges;
      productArgRanges.reserve(productFunc.getNumArguments());
      for (auto [arg, type] : llvm::zip(productFunc.getArguments(), productFunc.getArgumentTypes())) {
        if (isa<felt::FeltType>(type)) {
          productArgRanges.emplace_back(encoding.getScalarValueRange(arg));
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
            preamble, smt::IntType::get(&getContext()),
            StringAttr::get(&getContext(), witnessName)
        );
        encoding.emitRangeConstraint(
            rewriter, memberDef.getLoc(), constraintSym.getResult(),
            encoding.getConstraintMemberRange(memberDef.getSymName())
        );
        encoding.emitRangeConstraint(
            rewriter, memberDef.getLoc(), witnessSym.getResult(),
            encoding.getWitnessMemberRange(memberDef.getSymName())
        );
        signalSymbols[memberDef.getSymName()] = {constraintSym.getResult(), witnessSym.getResult()};
      }

      std::string smtFuncName = "smt_" + structDef.getSymName().str();
      auto *result = convertFunction(convertBodies(structDef, signalSymbols, &encoding));
      if (result == nullptr) {
        signalPassFailure();
        return WalkResult::interrupt();
      }

      auto smtFunc = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupSymbolIn(symbolTableOp, smtFuncName)
      );
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
        encoding.emitRangeConstraint(
            argRewriter, smtFunc.getLoc(), smtFunc.getArgument(idx), *maybeRange
        );
      }

      return WalkResult::advance();
    });
  }
};

namespace smt {
std::unique_ptr<mlir::Pass> createSMTLoweringPass() { return std::make_unique<SMTLoweringPass>(); }
} // namespace smt

} // namespace llzk
