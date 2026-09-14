//===-- SMTLoweringCommon.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "SMTLoweringCommon.h"

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/Include/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/SMT/IR/SMTOps.h"
#include "llzk/Dialect/SMT/IR/SMTTypes.h"
#include "llzk/Dialect/String/IR/Ops.h"
#include "llzk/Util/TypeHelper.h"
#include "llzk/Util/Walk.h"

#include <mlir/IR/SymbolTable.h>

#include <llvm/ADT/TypeSwitch.h>

#include <utility>

using namespace mlir;

namespace llzk::smt::detail {

std::pair<mlir::Value, mlir::Value> SMTIntTheoryEmitter::getRangeBoundAssertions(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value, const UnreducedInterval &range
) const {
  auto lower = createIntConstant(builder, loc, range.getLHS());
  auto upper = createIntConstant(builder, loc, range.getRHS());
  auto lowerBound =
      builder.create<smt::IntCmpOp>(loc, smt::IntPredicate::ge, value, lower.getResult());
  auto upperBound =
      builder.create<smt::IntCmpOp>(loc, smt::IntPredicate::le, value, upper.getResult());

  return {lowerBound.getResult(), upperBound.getResult()};
}

void SMTIntTheoryEmitter::emitRangeConstraint(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value, const UnreducedInterval &range
) const {
  auto [lowerBound, upperBound] = getRangeBoundAssertions(builder, loc, value, range);
  // Assert the lower bound of the canonical/unreduced interval for this symbol.
  builder.create<smt::AssertOp>(loc, lowerBound);
  // Assert the upper bound of the canonical/unreduced interval for this symbol.
  builder.create<smt::AssertOp>(loc, upperBound);
}

mlir::Value SMTIntTheoryEmitter::emitFreshSymbol(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::StringRef name
) const {
  std::string freshName = getFreshName(name);
  return builder
      .create<smt::DeclareFunOp>(loc, smt::IntType::get(ctx), StringAttr::get(ctx, freshName))
      .getResult();
}

mlir::Value SMTIntTheoryEmitter::emitConstant(
    mlir::OpBuilder &builder, mlir::Location loc, const llvm::DynamicAPInt &value
) const {
  return createIntConstant(builder, loc, value).getResult();
}

mlir::Value SMTIntTheoryEmitter::emitSub(
    mlir::OpBuilder &builder, mlir::Location loc, Value lhs, Value rhs
) const {
  return builder.create<smt::IntSubOp>(loc, lhs, rhs).getResult();
}

Value SMTIntTheoryEmitter::emitAdd(OpBuilder &builder, Location loc, Value lhs, Value rhs) const {
  return builder.create<smt::IntAddOp>(loc, ValueRange {lhs, rhs}).getResult();
}

Value SMTIntTheoryEmitter::emitMul(OpBuilder &builder, Location loc, Value lhs, Value rhs) const {
  return builder.create<smt::IntMulOp>(loc, ValueRange {lhs, rhs}).getResult();
}

Value SMTIntTheoryEmitter::emitDiv(OpBuilder &builder, Location loc, Value lhs, Value rhs) const {
  return builder.create<smt::IntDivOp>(loc, lhs, rhs).getResult();
}

Value SMTIntTheoryEmitter::emitSignedDiv(
    OpBuilder &builder, Location loc, Value lhs, Value rhs
) const {
  return emitTruncatingSignedDivision(builder, loc, lhs, rhs);
}

Value SMTIntTheoryEmitter::emitSignedRem(
    OpBuilder &builder, Location loc, Value lhs, Value rhs
) const {
  Value quotient = emitTruncatingSignedDivision(builder, loc, lhs, rhs);
  Value product = emitMul(builder, loc, quotient, rhs);
  return emitSub(builder, loc, lhs, product);
}

Value SMTIntTheoryEmitter::emitModPrime(OpBuilder &builder, Location loc, Value value) const {
  auto primeConst = createPrimeConstant(builder, loc);
  return builder.create<smt::IntModOp>(loc, ValueRange {value, primeConst.getResult()}).getResult();
}

Value SMTIntTheoryEmitter::emitPrimeMultiple(OpBuilder &builder, Location loc, Value factor) const {
  auto primeConst = createPrimeConstant(builder, loc);
  return emitMul(builder, loc, factor, primeConst.getResult());
}

Value SMTIntTheoryEmitter::emitOrderedComparison(
    OpBuilder &builder, Location loc, boolean::FeltCmpPredicate predicate, Value lhs, Value rhs
) const {
  static DenseMap<boolean::FeltCmpPredicate, smt::IntPredicate> predicateComparator = {
      {boolean::FeltCmpPredicate::GE, smt::IntPredicate::ge},
      {boolean::FeltCmpPredicate::GT, smt::IntPredicate::gt},
      {boolean::FeltCmpPredicate::LE, smt::IntPredicate::le},
      {boolean::FeltCmpPredicate::LT, smt::IntPredicate::lt}
  };
  return builder.create<smt::IntCmpOp>(loc, predicateComparator[predicate], lhs, rhs).getResult();
}

/// |value| = if value < 0 then -value else value
Value SMTIntTheoryEmitter::emitAbsValue(OpBuilder &builder, Location loc, Value value) const {
  Value zero = emitConstant(builder, loc, llvm::DynamicAPInt(0));
  Value isNegative =
      emitOrderedComparison(builder, loc, boolean::FeltCmpPredicate::LT, value, zero);
  Value negated = emitSub(builder, loc, zero, value);
  return builder.create<smt::IteOp>(loc, isNegative, negated, value).getResult();
}

/// absQuotient = |lhs| / |rhs|
/// quotient = if sign(lhs) != sign(rhs) then -absQuotient else absQuotient
Value SMTIntTheoryEmitter::emitTruncatingSignedDivision(
    OpBuilder &builder, Location loc, Value lhs, Value rhs
) const {
  Value zero = emitConstant(builder, loc, llvm::DynamicAPInt(0));
  Value lhsNeg = emitOrderedComparison(builder, loc, boolean::FeltCmpPredicate::LT, lhs, zero);
  Value rhsNeg = emitOrderedComparison(builder, loc, boolean::FeltCmpPredicate::LT, rhs, zero);
  Value lhsAbs = emitAbsValue(builder, loc, lhs);
  Value rhsAbs = emitAbsValue(builder, loc, rhs);
  Value absQuotient = emitDiv(builder, loc, lhsAbs, rhsAbs);
  // we can use xor here because we are checking if the signs are different
  Value signsDiffer = builder.create<smt::XOrOp>(loc, ValueRange {lhsNeg, rhsNeg}).getResult();
  Value negatedQuotient = emitSub(builder, loc, zero, absQuotient);
  return builder.create<smt::IteOp>(loc, signsDiffer, negatedQuotient, absQuotient).getResult();
}

std::string SMTIntTheoryEmitter::getFreshName(StringRef baseName) const {
  unsigned count = freshSymbolCounts[baseName]++;
  if (count == 0) {
    return baseName.str();
  }

  std::string uniqueName(baseName);
  uniqueName += "_";
  uniqueName += std::to_string(count);
  return uniqueName;
}

smt::IntConstantOp
SMTIntTheoryEmitter::createPrimeConstant(OpBuilder &builder, Location loc) const {
  return builder.create<smt::IntConstantOp>(loc, IntegerAttr::get(ctx, prime));
}

smt::IntConstantOp SMTIntTheoryEmitter::createIntConstant(
    OpBuilder &builder, Location loc, const llvm::DynamicAPInt &value
) const {
  return builder.create<smt::IntConstantOp>(loc, IntegerAttr::get(ctx, toAPSInt(value)));
}

FailureOr<FieldRef> resolveSelectedField(ModuleOp mod, StringRef fieldName) {
  FieldSet fields;
  if (!fieldName.empty()) {
    auto fieldLookupResult = Field::tryGetField(fieldName);
    if (failed(fieldLookupResult)) {
      mod.emitError() << "unknown field \"" << fieldName << "\"";
      return failure();
    }
    fields.insert(fieldLookupResult.value());
  }

  (void)collectFields(mod, fields);

  if (fields.empty()) {
    mod.emitError() << "no prime field specified; could not deduce";
    return failure();
  }

  if (fields.size() > 1) {
    mod.emitError() << "multiple fields unsupported";
    return failure();
  }

  return *(fields.begin());
}

Value SMTIntTheoryEmitter::emitArraySelect(
    Location loc, Value array, ValueRange indices, OpBuilder &builder
) {
  for (auto index : indices) {
    array = builder.create<smt::ArraySelectOp>(loc, array, index).getResult();
  }
  return array;
}

bool isFeltOrArrayOfFelt(mlir::Type type) {
  if (isa<felt::FeltType>(type)) {
    return true;
  }
  if (auto arrType = dyn_cast<array::ArrayType>(type)) {
    return isa<felt::FeltType>(arrType.getElementType());
  }
  return false;
}

mlir::Value SMTIntTheoryEmitter::emitQuantifiedAssertion(
    Location loc, Value array, ArrayRef<size_t> extents,
    function_ref<mlir::Value(mlir::Value)> body, OpBuilder &builder
) {
  SmallVector<Type> forallTypes(extents.size(), smt::IntType::get(builder.getContext()));
  return builder
      .create<smt::ForallOp>(
          loc, forallTypes,
          [this, &extents, &array,
           &body](OpBuilder &builder, Location loc, ValueRange indices) -> Value {
    SmallVector<Value> antecedents;
    antecedents.reserve(2 * extents.size());
    for (auto [index, extent] : llvm::zip(indices, extents)) {
      auto [lo, hi] = getRangeBoundAssertions(
          builder, loc, index, UnreducedInterval {0, static_cast<int64_t>(extent - 1)}
      );
      antecedents.push_back(lo);
      antecedents.push_back(hi);
    }

    Value antecedent = builder.create<smt::AndOp>(loc, antecedents).getResult();
    Value currentElement = emitArraySelect(loc, array, indices, builder);
    auto consequent = body(currentElement);
    return builder.create<smt::ImpliesOp>(loc, antecedent, consequent);
  }
      )
      .getResult();
}

LLZKToSMTTypeConverter::LLZKToSMTTypeConverter(MLIRContext *ctx) {
  addConversion([](Type type) { return type; });
  addConversion([this, ctx](array::ArrayType arrType) {
    return smt::ArrayType::get(ctx, smt::IntType::get(ctx), convertType(arrType.getElementType()));
  });
  addConversion([ctx](IndexType) { return smt::IntType::get(ctx); });
  addConversion([ctx](IntegerType type) -> Type {
    if (type.isSignless() && type.getWidth() == 1) {
      return smt::BoolType::get(ctx);
    }
    return smt::IntType::get(ctx);
  });
  addConversion([ctx](felt::FeltType) { return smt::IntType::get(ctx); });
  addConversion([this, ctx](array::ArrayType arrType) {
    return smt::ArrayType::get(ctx, smt::IntType::get(ctx), convertType(arrType.getElementType()));
  });
}

bool containsFeltOrStruct(Type type) {
  return isa<component::StructType>(type) ||
         TypeSwitch<Type, bool>(type)
             .Case<felt::FeltType>([](auto) { return true; })
             .Case<array::ArrayType>([](array::ArrayType arrayType) {
    return containsFeltOrStruct(arrayType.getElementType());
  }).Default([](auto) { return false; });
}

Operation *convertStructProductToFunc(Operation *op, MLIRContext *context) {
  if (op == nullptr) {
    return op;
  }

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

void configureSMTNoCFBodyConversionTarget(ConversionTarget &target) {
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
}

Operation *
applySMTNoCFBodyConversion(Operation *op, ConversionTarget &target, RewritePatternSet &&patterns) {
  if (op == nullptr) {
    return op;
  }

  ConversionConfig config;
  config.buildMaterializations = false;
  if (failed(applyPartialConversion(op, target, std::move(patterns), config))) {
    return nullptr;
  }

  auto deadStructs = walkCollect<component::CreateStructOp>(*op, [](auto createStructOp) {
    return createStructOp->use_empty();
  });
  for (component::CreateStructOp createStructOp : deadStructs) {
    createStructOp->erase();
  }

  return op;
}

LogicalResult FunctionDefConverter::matchAndRewrite(
    function::FuncDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter
) const {
  SmallVector<Type> convertedArgTypes = llvm::map_to_vector(op.getArgumentTypes(), [this](Type t) {
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
  if (!signatureConversion.has_value()) {
    return failure();
  }
  rewriter.applySignatureConversion(&block, *signatureConversion);
  return success();
}

MemberReadConverter::MemberReadConverter(
    TypeConverter &converter, MLIRContext *context, const SignalSymbols &signalMap
)
    : OpConversionPattern<component::MemberReadOp>(converter, context, /*benefit=*/2),
      symbols(signalMap) {}

LogicalResult MemberReadConverter::matchAndRewrite(
    component::MemberReadOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  if (!isFeltOrArrayOfFelt(op.getResult().getType())) {
    op.emitError("SMT lowering currently only supports felt- or array-of-felt-valued struct.readm");
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

LogicalResult StructDefConverter::matchAndRewrite(
    component::StructDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter
) const {
  std::string smtFuncName = ("smt_" + op.getSymName()).str();
  auto productFunc = op.getProductFuncOp();
  auto smtFunc =
      rewriter.create<func::FuncOp>(op->getLoc(), smtFuncName, productFunc.getFunctionType());
  IRMapping mapping;
  productFunc.getFunctionBody().cloneInto(&smtFunc.getFunctionBody(), mapping);

  smtFunc.walk([&rewriter](function::ReturnOp returnOp) {
    rewriter.setInsertionPoint(returnOp);
    rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp, returnOp.getOperands());
  });

  rewriter.eraseOp(op);
  return success();
}

LogicalResult ReturnConverter::matchAndRewrite(
    function::ReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  SmallVector<Value> returnedValues;
  for (auto [val, type] : llvm::zip(adaptor.getOperands(), op.getOperandTypes())) {
    if (!isa<component::StructType>(type)) {
      returnedValues.push_back(val);
    }
  }

  rewriter.modifyOpInPlace(op, [&returnedValues, &op]() {
    op.getOperandsMutable().assign(returnedValues);
  });
  return success();
}

LogicalResult SCFIfConverter::matchAndRewrite(
    scf::IfOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  SmallVector<Type> convertedResultTypes = llvm::map_to_vector(op.getResultTypes(), [this](Type t) {
    return getTypeConverter()->convertType(t);
  });

  Value cond = adaptor.getCondition();
  if (!isa<IntegerType>(cond.getType())) {
    cond =
        rewriter
            .create<UnrealizedConversionCastOp>(op.getLoc(), TypeRange {rewriter.getI1Type()}, cond)
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

LogicalResult YieldConverter::matchAndRewrite(
    scf::YieldOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getResults());
  return success();
}

LogicalResult FeltConstConverter::matchAndRewrite(
    felt::FeltConstantOp op, OpAdaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<smt::IntConstantOp>(
      op, IntegerAttr::get(getContext(), APSInt {op.getValue().getValue()})
  );
  return success();
}

LogicalResult IndexConstConverter::matchAndRewrite(
    arith::ConstantIndexOp op, OpAdaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<smt::IntConstantOp>(op, dyn_cast<IntegerAttr>(op.getValue()));
  return success();
}

WriteArrayConverter::WriteArrayConverter(
    mlir::TypeConverter &converter, mlir::MLIRContext *context, ArrayWritePolicy _policy
)
    : OpConversionPattern<array::WriteArrayOp>(converter, context, /*benefit=*/2),
      policy {std::move(_policy)} {}

LogicalResult WriteArrayConverter::matchAndRewrite(
    array::WriteArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  // Turn `arr[i] = val` to `assert arr[i] == val`
  if (policy(op.getArrRef()) == ArrayWriteMode::WriteOnce) {
    Value selected = selectMultidimensionalArray(
        op->getLoc(), adaptor.getArrRef(), adaptor.getIndices(), rewriter
    );
    rewriter.replaceOpWithNewOp<smt::AssertOp>(
        op, rewriter.create<smt::EqOp>(op->getLoc(), selected, adaptor.getRvalue()).getResult()
    );
    return success();

  } else {
    // TODO: Track a fresh SMT value for the most recently stored copy of the array, store to that,
    // and update the most recent. This requires doing it in order, though, and handling control
    // flow carefully
    op.emitError("SMT lowering currently only supports write-once arrays");
    return failure();
  }
}

LogicalResult ReadArrayConverter::matchAndRewrite(
    array::ReadArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  auto readResult = selectMultidimensionalArray(
      op->getLoc(), adaptor.getArrRef(), adaptor.getIndices(), rewriter
  );
  rewriter.replaceOp(op, readResult.getDefiningOp());
  return success();
}

} // namespace llzk::smt::detail
