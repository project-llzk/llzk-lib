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
#include "llzk/Dialect/Constrain/IR/Ops.h"
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

Value selectMultidimensionalArray(
    Location loc, Value array, ValueRange indices, OpBuilder &builder
) {
  for (auto index : indices) {
    array = builder.create<smt::ArraySelectOp>(loc, array, index).getResult();
  }
  return array;
}

LLZKToSMTTypeConverter::LLZKToSMTTypeConverter(MLIRContext *ctx) {
  addConversion([](Type type) { return type; });
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

// arr[i, j, k] => arr[i][j][k]
static inline Value
smtReadArray(Location loc, Value array, ValueRange indices, PatternRewriter &rewriter) {
  for (auto index : indices) {
    array = rewriter.create<smt::ArraySelectOp>(loc, array, index).getResult();
  }
  return array;
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
    Value selected =
        smtReadArray(op->getLoc(), adaptor.getArrRef(), adaptor.getIndices(), rewriter);
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
  auto readResult = smtReadArray(op->getLoc(), adaptor.getArrRef(), adaptor.getIndices(), rewriter);
  rewriter.replaceOp(op, readResult.getDefiningOp());
  return success();
}

} // namespace llzk::smt::detail
