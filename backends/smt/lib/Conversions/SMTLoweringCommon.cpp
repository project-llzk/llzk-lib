//===-- SMTLoweringCommon.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "SMTLoweringCommon.h"

#include "llzk/Analysis/Intervals.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/Include/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/String/IR/Ops.h"
#include "llzk/Util/TypeHelper.h"
#include "llzk/Util/Walk.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SMT/IR/SMTOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/TypeSwitch.h>

#include <utility>

using namespace mlir;

namespace llzk::smt::detail {

std::pair<Value, Value> SMTIntTheoryEmitter::getRangeBoundAssertions(
    OpBuilder &builder, Location loc, Value value, const UnreducedInterval &range
) const {
  auto lower = createIntConstant(builder, loc, range.getLHS());
  auto upper = createIntConstant(builder, loc, range.getRHS());
  auto lowerBound =
      smt::IntCmpOp::create(builder, loc, smt::IntPredicate::ge, value, lower.getResult());
  auto upperBound =
      smt::IntCmpOp::create(builder, loc, smt::IntPredicate::le, value, upper.getResult());

  return {lowerBound.getResult(), upperBound.getResult()};
}

void SMTIntTheoryEmitter::emitRangeConstraint(
    OpBuilder &builder, Location loc, Value value, const UnreducedInterval &range
) const {
  auto [lowerBound, upperBound] = getRangeBoundAssertions(builder, loc, value, range);
  // Assert the lower bound of the canonical/unreduced interval for this symbol.
  smt::AssertOp::create(builder, loc, lowerBound);
  // Assert the upper bound of the canonical/unreduced interval for this symbol.
  smt::AssertOp::create(builder, loc, upperBound);
}

Value SMTIntTheoryEmitter::emitFreshSymbol(OpBuilder &builder, Location loc, StringRef name) const {
  std::string freshName = getFreshName(name);
  return smt::DeclareFunOp::create(
             builder, loc, smt::IntType::get(ctx), StringAttr::get(ctx, freshName)
  )
      .getResult();
}

Value SMTIntTheoryEmitter::emitFreshArray(
    OpBuilder &builder, Location loc, StringRef name, smt::ArrayType type
) const {
  return smt::DeclareFunOp::create(builder, loc, type, StringAttr::get(ctx, getFreshName(name)))
      .getResult();
}

Value SMTIntTheoryEmitter::emitConstant(
    OpBuilder &builder, Location loc, const DynamicAPInt &value
) const {
  return createIntConstant(builder, loc, value).getResult();
}

Value SMTIntTheoryEmitter::emitSub(OpBuilder &builder, Location loc, Value lhs, Value rhs) const {
  return smt::IntSubOp::create(builder, loc, lhs, rhs).getResult();
}

Value SMTIntTheoryEmitter::emitAdd(OpBuilder &builder, Location loc, Value lhs, Value rhs) const {
  return smt::IntAddOp::create(builder, loc, ValueRange {lhs, rhs}).getResult();
}

Value SMTIntTheoryEmitter::emitMul(OpBuilder &builder, Location loc, Value lhs, Value rhs) const {
  return smt::IntMulOp::create(builder, loc, ValueRange {lhs, rhs}).getResult();
}

Value SMTIntTheoryEmitter::emitDiv(OpBuilder &builder, Location loc, Value lhs, Value rhs) const {
  return smt::IntDivOp::create(builder, loc, lhs, rhs).getResult();
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
  return smt::IntModOp::create(builder, loc, ValueRange {value, primeConst.getResult()})
      .getResult();
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
  return smt::IntCmpOp::create(builder, loc, predicateComparator[predicate], lhs, rhs).getResult();
}

/// |value| = if value < 0 then -value else value
Value SMTIntTheoryEmitter::emitAbsValue(OpBuilder &builder, Location loc, Value value) const {
  Value zero = emitConstant(builder, loc, DynamicAPInt(0));
  Value isNegative =
      emitOrderedComparison(builder, loc, boolean::FeltCmpPredicate::LT, value, zero);
  Value negated = emitSub(builder, loc, zero, value);
  return smt::IteOp::create(builder, loc, isNegative, negated, value).getResult();
}

/// absQuotient = |lhs| / |rhs|
/// quotient = if sign(lhs) != sign(rhs) then -absQuotient else absQuotient
Value SMTIntTheoryEmitter::emitTruncatingSignedDivision(
    OpBuilder &builder, Location loc, Value lhs, Value rhs
) const {
  Value zero = emitConstant(builder, loc, DynamicAPInt(0));
  Value lhsNeg = emitOrderedComparison(builder, loc, boolean::FeltCmpPredicate::LT, lhs, zero);
  Value rhsNeg = emitOrderedComparison(builder, loc, boolean::FeltCmpPredicate::LT, rhs, zero);
  Value lhsAbs = emitAbsValue(builder, loc, lhs);
  Value rhsAbs = emitAbsValue(builder, loc, rhs);
  Value absQuotient = emitDiv(builder, loc, lhsAbs, rhsAbs);
  // we can use xor here because we are checking if the signs are different
  Value signsDiffer = smt::XOrOp::create(builder, loc, ValueRange {lhsNeg, rhsNeg}).getResult();
  Value negatedQuotient = emitSub(builder, loc, zero, absQuotient);
  return smt::IteOp::create(builder, loc, signsDiffer, negatedQuotient, absQuotient).getResult();
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
  return smt::IntConstantOp::create(builder, loc, IntegerAttr::get(ctx, prime));
}

smt::IntConstantOp SMTIntTheoryEmitter::createIntConstant(
    OpBuilder &builder, Location loc, const DynamicAPInt &value
) const {
  return smt::IntConstantOp::create(builder, loc, IntegerAttr::get(ctx, toAPSInt(value)));
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
    array = smt::ArraySelectOp::create(builder, loc, array, index).getResult();
  }
  return array;
}

Value SMTIntTheoryEmitter::emitArrayStore(
    Location loc, Value array, ValueRange indices, Value value, OpBuilder &builder
) {
  assert(!indices.empty() && "an array store requires at least one index");

  SmallVector<Value> arrays {array};
  arrays.reserve(indices.size());
  for (Value index : indices.drop_back()) {
    arrays.push_back(smt::ArraySelectOp::create(builder, loc, arrays.back(), index).getResult());
  }

  Value updated =
      smt::ArrayStoreOp::create(builder, loc, arrays.back(), indices.back(), value).getResult();
  for (size_t depth = indices.size() - 1; depth > 0; --depth) {
    updated =
        smt::ArrayStoreOp::create(builder, loc, arrays[depth - 1], indices[depth - 1], updated)
            .getResult();
  }
  return updated;
}

FailureOr<ResolvedArraySemantics> ArraySemanticsResolver::resolve(Value array) const {
  ArrayWriteMode mode = policy(array);
  if (mode == ArrayWriteMode::WriteOnce) {
    return ResolvedArraySemantics {mode, std::nullopt};
  }

  auto allocation = array.getDefiningOp<array::CreateArrayOp>();
  if (!allocation) {
    emitError(
        array.getLoc()
    ) << "overwrite array policy did not provide a supported canonical state root";
    return failure();
  }

  Value allocationValue = allocation.getResult();
  return ResolvedArraySemantics {
      mode,
      ArrayStateRoot {allocationValue, allocationValue, allocation.getLoc(), allocation->getBlock()}
  };
}

FailureOr<std::unique_ptr<ArrayLoweringState>>
ArrayLoweringState::create(Operation *scope, const ArraySemanticsResolver &resolver) {
  auto state = std::make_unique<ArrayLoweringState>();
  if (failed(state->initialize(scope, resolver))) {
    return failure();
  }
  return state;
}

LogicalResult
ArrayLoweringState::initialize(Operation *scope, const ArraySemanticsResolver &resolver) {
  WalkResult registration = scope->walk([&](array::CreateArrayOp allocation) {
    auto arrayType = allocation.getType();
    if (!isa<felt::FeltType>(arrayType.getElementType())) {
      return WalkResult::advance();
    }

    FailureOr<ResolvedArraySemantics> semantics = resolver.resolve(allocation.getResult());
    if (failed(semantics)) {
      return WalkResult::interrupt();
    }
    if (semantics->mode == ArrayWriteMode::WriteOnce) {
      return WalkResult::advance();
    }
    if (!semantics->overwriteRoot.has_value()) {
      allocation.emitError("overwrite array semantics require a state root");
      return WalkResult::interrupt();
    }

    const ArrayStateRoot &root = *semantics->overwriteRoot;
    unsigned recordIndex = records.size();
    if (!recordByKey.try_emplace(root.key, recordIndex).second) {
      allocation.emitError("duplicate overwrite array state root");
      return WalkResult::interrupt();
    }
    records.push_back(Record {root, {}, false});
    return WalkResult::advance();
  });
  if (registration.wasInterrupted()) {
    return failure();
  }

  for (Record &record : records) {
    for (OpOperand &use : record.root.initialValue.getUses()) {
      Operation *user = use.getOwner();
      if (isa<array::ArrayLengthOp>(user)) {
        continue;
      }
      bool isDirectAccess = isa<array::ReadArrayOp, array::WriteArrayOp>(user) &&
                            user->getBlock() == record.root.stateBlock;
      if (isDirectAccess) {
        continue;
      }

      InFlightDiagnostic diagnostic = user->emitError(
          "SMT overwrite lowering requires direct array reads and writes in the "
          "allocation's block; aliases, escapes, and control-flow state threading are "
          "not supported"
      );
      diagnostic.attachNote(record.root.diagnosticLoc).append("array allocated here");
      return failure();
    }
  }

  WalkResult collection = scope->walk([&](Operation *operation) {
    Value arrayValue;
    if (auto read = dyn_cast<array::ReadArrayOp>(operation)) {
      arrayValue = read.getArrRef();
    } else if (auto write = dyn_cast<array::WriteArrayOp>(operation)) {
      arrayValue = write.getArrRef();
    } else {
      return WalkResult::advance();
    }

    FailureOr<ResolvedArraySemantics> semantics = resolver.resolve(arrayValue);
    if (failed(semantics)) {
      return WalkResult::interrupt();
    }
    if (semantics->mode == ArrayWriteMode::WriteOnce) {
      return WalkResult::advance();
    }
    if (!semantics->overwriteRoot.has_value()) {
      operation->emitError("overwrite array semantics require a state root");
      return WalkResult::interrupt();
    }

    auto recordIt = recordByKey.find(semantics->overwriteRoot->key);
    if (recordIt == recordByKey.end()) {
      operation->emitError("failed to find prevalidated overwrite array state");
      return WalkResult::interrupt();
    }
    unsigned recordIndex = recordIt->second;
    records[recordIndex].accesses.push_back(operation);
    overwriteAccesses[operation] = recordIndex;
    return WalkResult::advance();
  });
  return success(!collection.wasInterrupted());
}

bool ArrayLoweringState::isOverwriteAccess(Operation *access) const {
  return overwriteAccesses.contains(access);
}

LogicalResult ArrayLoweringState::lowerOverwriteChain(
    Operation *trigger, ConversionPatternRewriter &rewriter, SMTIntTheoryEmitter &emitter
) {
  auto accessIt = overwriteAccesses.find(trigger);
  if (accessIt == overwriteAccesses.end()) {
    return failure();
  }

  Record &record = records[accessIt->second];
  if (record.processed) {
    return success();
  }

  Value current = rewriter.getRemappedValue(record.root.initialValue);
  if (!current || !isa<smt::ArrayType>(current.getType())) {
    return failure();
  }

  // Lower the complete source-ordered chain in one transaction. This avoids
  // assigning semantics to the conversion driver's pattern visitation order.
  for (Operation *access : record.accesses) {
    rewriter.setInsertionPoint(access);
    if (auto write = dyn_cast<array::WriteArrayOp>(access)) {
      SmallVector<Value> indices;
      if (failed(rewriter.getRemappedValues(write.getIndices(), indices))) {
        return failure();
      }
      Value rvalue = rewriter.getRemappedValue(write.getRvalue());
      if (!rvalue) {
        return failure();
      }
      current = emitter.emitArrayStore(write.getLoc(), current, indices, rvalue, rewriter);
      rewriter.eraseOp(write);
      continue;
    }

    auto read = cast<array::ReadArrayOp>(access);
    SmallVector<Value> indices;
    if (failed(rewriter.getRemappedValues(read.getIndices(), indices))) {
      return failure();
    }
    Value selected = emitter.emitArraySelect(read.getLoc(), current, indices, rewriter);
    rewriter.replaceOp(read, selected);
  }

  record.processed = true;
  return success();
}

bool isFeltOrArrayOfFelt(Type type) {
  if (isa<felt::FeltType>(type)) {
    return true;
  }
  if (auto arrType = dyn_cast<array::ArrayType>(type)) {
    return isa<felt::FeltType>(arrType.getElementType());
  }
  return false;
}

FailureOr<SmallVector<size_t>> getExtents(array::ArrayType type) {
  SmallVector<size_t> extents;
  for (auto dim : type.getShape()) {
    if (dim < 0) {
      return failure();
    }
    extents.push_back(dim);
  }
  return extents;
}

Value SMTIntTheoryEmitter::emitQuantifiedAssertion(
    Location loc, ArrayRef<size_t> extents, function_ref<Value(ValueRange)> body, OpBuilder &builder
) {
  SmallVector<Type> forallTypes(extents.size(), smt::IntType::get(builder.getContext()));
  auto elementInRange = [this, &extents,
                         &body](OpBuilder &b, Location l, ValueRange indices) -> Value {
    SmallVector<Value> antecedents;
    antecedents.reserve(2 * extents.size());
    for (auto [index, extent] : zip(indices, extents)) {
      auto [lo, hi] = getRangeBoundAssertions(
          b, l, index, UnreducedInterval {0, static_cast<int64_t>(extent - 1)}
      );
      antecedents.push_back(lo);
      antecedents.push_back(hi);
    }

    Value antecedent = smt::AndOp::create(b, l, antecedents).getResult();
    auto consequent = body(indices);
    return smt::ImpliesOp::create(b, l, antecedent, consequent);
  };
  return smt::ForallOp::create(builder, loc, forallTypes, elementInRange).getResult();
}

static inline Type smtArrayOfRank(MLIRContext *ctx, int64_t rank, Type elementType) {
  if (rank == 0) {
    return elementType;
  }
  return smt::ArrayType::get(
      ctx, smt::IntType::get(ctx), smtArrayOfRank(ctx, rank - 1, elementType)
  );
}

LLZKToSMTTypeConverter::LLZKToSMTTypeConverter(MLIRContext *ctx) {

  addConversion([](Type type) { return type; });
  addConversion([this, ctx](array::ArrayType arrType) {
    return smtArrayOfRank(ctx, arrType.getRank(), convertType(arrType.getElementType()));
  });
  addConversion([ctx](IndexType) { return smt::IntType::get(ctx); });
  addConversion([ctx](IntegerType type) -> Type {
    if (type.isSignless() && type.getWidth() == 1) {
      return mlir::smt::BoolType::get(ctx);
    }
    return smt::IntType::get(ctx);
  });
  addConversion([ctx](felt::FeltType) { return smt::IntType::get(ctx); });
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
  target.addLegalDialect<mlir::smt::SMTDialect>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addIllegalOp<component::MemberWriteOp, component::MemberReadOp>();
  target.addLegalOp<component::CreateStructOp>();
  target.addDynamicallyLegalOp<function::ReturnOp>([](function::ReturnOp returnOp) {
    return none_of(returnOp.getOperandTypes(), [](Type type) {
      return isa<component::StructType>(type);
    });
  });

  target.addDynamicallyLegalOp<function::FuncDefOp>([](function::FuncDefOp funcOp) {
    bool signatureLegal = none_of(funcOp.getArgumentTypes(), containsFeltOrStruct) &&
                          none_of(funcOp.getResultTypes(), containsFeltOrStruct);
    return signatureLegal;
  });
  target.addDynamicallyLegalOp<scf::YieldOp>([](scf::YieldOp yieldOp) {
    return none_of(yieldOp.getOperandTypes(), containsFeltOrStruct);
  });
  target.addDynamicallyLegalOp<scf::IfOp>([](scf::IfOp ifOp) {
    return none_of(ifOp.getResultTypes(), containsFeltOrStruct);
  });
}

void configureSMTOverwriteArrayConversionTarget(ConversionTarget &target) {
  target.addDynamicallyLegalOp<array::CreateArrayOp>([](array::CreateArrayOp op) {
    return !isa<felt::FeltType>(op.getType().getElementType());
  });
  target.addDynamicallyLegalOp<array::ArrayLengthOp>([](array::ArrayLengthOp op) {
    return !isa<felt::FeltType>(op.getArrRefType().getElementType());
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
  SmallVector<Type> convertedArgTypes = map_to_vector(op.getArgumentTypes(), [this](Type t) {
    return getTypeConverter()->convertType(t);
  });
  SmallVector<Type> convertedResultTypes = map_to_vector(
      filter_to_vector(op.getResultTypes(), [](Type t) { return !isa<component::StructType>(t); }),
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
      func::FuncOp::create(rewriter, op->getLoc(), smtFuncName, productFunc.getFunctionType());
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
  for (auto [val, type] : zip(adaptor.getOperands(), op.getOperandTypes())) {
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
  SmallVector<Type> convertedResultTypes = map_to_vector(op.getResultTypes(), [this](Type t) {
    return getTypeConverter()->convertType(t);
  });

  Value cond = adaptor.getCondition();
  if (!isa<IntegerType>(cond.getType())) {
    cond = UnrealizedConversionCastOp::create(
               rewriter, op.getLoc(), TypeRange {rewriter.getI1Type()}, cond
    )
               .getResult(0);
  }

  auto convertedIf = scf::IfOp::create(
      rewriter, op.getLoc(), convertedResultTypes, cond,
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
  rewriter.replaceOpWithNewOp<mlir::smt::IntConstantOp>(
      op, IntegerAttr::get(getContext(), APSInt {op.getValue().getValue()})
  );
  return success();
}

IndexConstConverter::IndexConstConverter(
    TypeConverter &converter, MLIRContext *context, SMTIntTheoryEmitter *theoryEmitter
)
    : OpConversionPattern<arith::ConstantIndexOp>(converter, context, /*benefit=*/2),
      emitter {theoryEmitter} {}

LogicalResult IndexConstConverter::matchAndRewrite(
    arith::ConstantIndexOp op, OpAdaptor, ConversionPatternRewriter &rewriter
) const {
  auto smtIndexOp =
      smt::IntConstantOp::create(rewriter, op.getLoc(), dyn_cast<IntegerAttr>(op.getValue()));

  // Constrain the index to be between 0 and 2^64 - 1
  emitter->emitRangeConstraint(
      rewriter, op.getLoc(), smtIndexOp.getResult(),
      UnreducedInterval {
          // Have to do this because the other constructor for UnreducedInterval only accepts
          // int64_t
          llvm::DynamicAPInt {llvm::APSInt {64, 0}},
          llvm::DynamicAPInt {llvm::APInt::getSignedMaxValue(64)}
      }
  );
  rewriter.replaceOp(op, smtIndexOp);
  return success();
}

CreateArrayConverter::CreateArrayConverter(
    TypeConverter &converter, MLIRContext *context, SMTIntTheoryEmitter *theoryEmitter
)
    : OpConversionPattern<array::CreateArrayOp>(converter, context, /*benefit=*/2),
      emitter {theoryEmitter} {}

LogicalResult CreateArrayConverter::matchAndRewrite(
    array::CreateArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  auto convertedType =
      dyn_cast_or_null<smt::ArrayType>(getTypeConverter()->convertType(op.getType()));
  if (!convertedType) {
    return failure();
  }

  Value current = emitter->emitFreshArray(rewriter, op.getLoc(), "array_alloc", convertedType);
  if (!adaptor.getElements().empty()) {
    FailureOr<SmallVector<size_t>> extents = getExtents(op.getType());
    if (failed(extents)) {
      op.emitError("SMT lowering cannot initialize a dynamically-shaped local array");
      return failure();
    }

    for (auto [linearIndex, element] : llvm::enumerate(adaptor.getElements())) {
      SmallVector<size_t> coordinates(extents->size());
      size_t remainder = linearIndex;
      for (size_t dimension = extents->size(); dimension > 0; --dimension) {
        size_t extent = (*extents)[dimension - 1];
        coordinates[dimension - 1] = remainder % extent;
        remainder /= extent;
      }

      SmallVector<Value> indices;
      indices.reserve(coordinates.size());
      for (size_t coordinate : coordinates) {
        indices.push_back(emitter->emitConstant(
            rewriter, op.getLoc(), llvm::DynamicAPInt {static_cast<int64_t>(coordinate)}
        ));
      }
      current = emitter->emitArrayStore(op.getLoc(), current, indices, element, rewriter);
    }
  }

  rewriter.replaceOp(op, current);
  return success();
}

ArrayLengthConverter::ArrayLengthConverter(
    TypeConverter &converter, MLIRContext *context, SMTIntTheoryEmitter *theoryEmitter
)
    : OpConversionPattern<array::ArrayLengthOp>(converter, context, /*benefit=*/2),
      emitter {theoryEmitter} {}

LogicalResult ArrayLengthConverter::matchAndRewrite(
    array::ArrayLengthOp op, OpAdaptor, ConversionPatternRewriter &rewriter
) const {
  llvm::APInt dimensionValue;
  if (!matchPattern(op.getDim(), m_ConstantInt(&dimensionValue))) {
    return rewriter.notifyMatchFailure(op, "array dimension is not constant");
  }
  std::optional<int64_t> dimension = dimensionValue.trySExtValue();
  FailureOr<SmallVector<size_t>> extents = getExtents(op.getArrRefType());
  if (!dimension || *dimension < 0 || failed(extents) ||
      static_cast<size_t>(*dimension) >= extents->size()) {
    return rewriter.notifyMatchFailure(op, "array dimension has no static extent");
  }

  Value length = emitter->emitConstant(
      rewriter, op.getLoc(),
      llvm::DynamicAPInt {static_cast<int64_t>((*extents)[static_cast<size_t>(*dimension)])}
  );
  rewriter.replaceOp(op, length);
  return success();
}

WriteArrayConverter::WriteArrayConverter(
    TypeConverter &converter, MLIRContext *context, ArrayLoweringState *arrayState,
    SMTIntTheoryEmitter *theoryEmitter
)
    : OpConversionPattern<array::WriteArrayOp>(converter, context, /*benefit=*/2),
      state {arrayState}, emitter {theoryEmitter} {}

LogicalResult WriteArrayConverter::matchAndRewrite(
    array::WriteArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {

  if (!isa<felt::FeltType>(op.getArrRef().getType().getElementType())) {
    return failure();
  }

  // Turn `arr[i] = val` to `assert arr[i] == val`
  if (!state->isOverwriteAccess(op)) {
    Value selected =
        emitter->emitArraySelect(op->getLoc(), adaptor.getArrRef(), adaptor.getIndices(), rewriter);
    // I don't think interval analysis does much interesting with arrays so I don't think we can do
    // better than this?
    auto reducedSelected = emitter->emitModPrime(rewriter, op->getLoc(), selected);
    auto reducedRval = emitter->emitModPrime(rewriter, op->getLoc(), adaptor.getRvalue());
    rewriter.replaceOpWithNewOp<smt::AssertOp>(
        op, smt::EqOp::create(rewriter, op->getLoc(), reducedSelected, reducedRval).getResult()
    );
    return success();
  }

  return state->lowerOverwriteChain(op, rewriter, *emitter);
}

ReadArrayConverter::ReadArrayConverter(
    TypeConverter &converter, MLIRContext *context, ArrayLoweringState *arrayState,
    SMTIntTheoryEmitter *theoryEmitter
)
    : OpConversionPattern<array::ReadArrayOp>(converter, context, /*benefit=*/2),
      state {arrayState}, emitter {theoryEmitter} {}

LogicalResult ReadArrayConverter::matchAndRewrite(
    array::ReadArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  if (state->isOverwriteAccess(op)) {
    return state->lowerOverwriteChain(op, rewriter, *emitter);
  }
  auto readResult =
      emitter->emitArraySelect(op->getLoc(), adaptor.getArrRef(), adaptor.getIndices(), rewriter);
  rewriter.replaceOp(op, readResult.getDefiningOp());
  return success();
}

} // namespace llzk::smt::detail
