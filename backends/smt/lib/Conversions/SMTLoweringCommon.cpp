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
#include "llzk/Dialect/String/IR/Ops.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/SymbolTable.h>

#include <llvm/ADT/TypeSwitch.h>

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

LLZKToSMTTypeConverter::LLZKToSMTTypeConverter(MLIRContext *ctx) {
  addConversion([](Type type) { return type; });
  addConversion([ctx](IntegerType type) -> Type {
    if (type.isSignless() && type.getWidth() == 1) {
      return smt::BoolType::get(ctx);
    }
    return type;
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

void configureSMTNoCFBodyConversionTarget(MLIRContext &context, ConversionTarget &target) {
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

Operation *applySMTNoCFBodyConversion(Operation *op, ConversionTarget &target,
                                      RewritePatternSet &&patterns) {
  if (op == nullptr) {
    return op;
  }

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

LogicalResult validateSupportedSMTMemberAccesses(component::StructDefOp structDef) {
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

LogicalResult FunctionDefConverter::matchAndRewrite(
    function::FuncDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter
) const {
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
  if (!signatureConversion.has_value()) {
    return failure();
  }
  rewriter.applySignatureConversion(&block, *signatureConversion);
  return success();
}

MemberReadConverter::MemberReadConverter(
    TypeConverter &typeConverter, MLIRContext *context, const SignalSymbols &signalMap
)
    : OpConversionPattern<component::MemberReadOp>(typeConverter, context, /*benefit=*/2),
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

  smtFunc.walk([&](function::ReturnOp returnOp) {
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

  rewriter.modifyOpInPlace(op, [&]() { op.getOperandsMutable().assign(returnedValues); });
  return success();
}

LogicalResult SCFIfConverter::matchAndRewrite(
    scf::IfOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  SmallVector<Type> convertedResultTypes =
      llvm::map_to_vector(op.getResultTypes(), [this](Type t) {
    return getTypeConverter()->convertType(t);
  });

  Value cond = adaptor.getCondition();
  if (!isa<IntegerType>(cond.getType())) {
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

} // namespace llzk::smt::detail
