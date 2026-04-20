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
#include "llzk/Util/Field.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

#include <algorithm>
#include <utility>

namespace llzk {
namespace smt {
#define GEN_PASS_DECL_SMTLOWERINGPASS
#define GEN_PASS_DEF_SMTLOWERINGPASS
#include "smt/Conversions/ConversionPasses.h.inc"
} // namespace smt

using namespace mlir;

using SignalSymbols = DenseMap<StringRef, std::pair<Value, Value>>;
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
    addConversion([this](array::ArrayType type) {
      auto elemType = convertType(type.getElementType());
      return array::ArrayType::get(elemType, type.getShape());
    });
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
  using OpConversionPattern<felt::DivFeltOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      felt::DivFeltOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {

    // Rewrite `%z = felt.div %x, %y` into:
    // %z = smt.declare_fun "z" : !smt.int
    // %c0 = smt.constant 0 : !smt.int
    // %y0 = smt.eq %y, %c0 : !smt.bool
    // %z0 = smt.eq %z, %c0 : !smt.bool
    // %yz = smt.int.mul %y, %z : !smt.int
    // %yz_eq_x = smt.eq %yz, %x : !smt.bool
    // %div_constraint = smt.ite %y0, %z0, %yz_eq_x : !smt.bool
    // smt.assert %div_constraint

    auto div = rewriter.create<smt::DeclareFunOp>(op->getLoc(), smt::IntType::get(getContext()));
    auto zero = rewriter.create<smt::IntConstantOp>(
        op->getLoc(), IntegerAttr::get(getContext(), APSInt {APInt {1, 0}})
    );
    auto denominatorIsZero =
        rewriter.create<smt::EqOp>(op->getLoc(), adaptor.getRhs(), zero.getResult());
    auto divIsZero = rewriter.create<smt::EqOp>(op->getLoc(), div.getResult(), zero.getResult());
    auto product = rewriter.create<smt::IntMulOp>(
        op->getLoc(), ValueRange {adaptor.getRhs(), div.getResult()}
    );
    auto productEqualsNumerator =
        rewriter.create<smt::EqOp>(op->getLoc(), product.getResult(), adaptor.getLhs());
    auto divConstraint = rewriter.create<smt::IteOp>(
        op->getLoc(), denominatorIsZero.getResult(), divIsZero.getResult(),
        productEqualsNumerator.getResult()
    );
    rewriter.create<smt::AssertOp>(op->getLoc(), divConstraint.getResult());
    rewriter.replaceOp(op, div.getResult());

    return success();
  }
};

class MemberWriteConverter : public OpConversionPattern<component::MemberWriteOp> {

  SignalSymbols symbols;
  APSInt prime;

public:
  MemberWriteConverter(
      TypeConverter &_typeConverter, MLIRContext *context, const SignalSymbols &_symbols,
      APSInt _prime
  )
      : OpConversionPattern {_typeConverter, context, /*benefit=*/2}, symbols {_symbols},
        prime {std::move(_prime)} {}
  using OpConversionPattern<component::MemberWriteOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      component::MemberWriteOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {

    auto it = symbols.find(adaptor.getMemberName());
    if (it == symbols.end()) {
      return failure();
    }

    auto [_, witness] = it->second;
    auto mod =
        rewriter.create<smt::IntConstantOp>(op->getLoc(), IntegerAttr::get(getContext(), prime));

    auto witnessMod = rewriter.create<smt::IntModOp>(op->getLoc(), witness, mod);
    auto valueMod = rewriter.create<smt::IntModOp>(op->getLoc(), adaptor.getVal(), mod);
    auto equal =
        rewriter.create<smt::EqOp>(op->getLoc(), witnessMod.getResult(), valueMod.getResult());
    rewriter.replaceOpWithNewOp<smt::AssertOp>(op, equal);

    return success();
  }
};

class MemberReadConverter : public OpConversionPattern<component::MemberReadOp> {
  SignalSymbols symbols;

public:
  MemberReadConverter(
      TypeConverter &_typeConverter, MLIRContext *context, const SignalSymbols &_symbols
  )
      : OpConversionPattern {_typeConverter, context, /*benefit=*/2}, symbols {_symbols} {}
  using OpConversionPattern<component::MemberReadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      component::MemberReadOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {

    // Create a symbol for the signal
    auto it = symbols.find(adaptor.getMemberName());
    if (it == symbols.end()) {
      return failure();
    }

    auto [constrain, witness] = it->second;
    rewriter.replaceOp(op, constrain.getDefiningOp());

    return success();
  }
};

class StructDefConverter : public OpConversionPattern<component::StructDefOp> {
  using OpConversionPattern<component::StructDefOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      component::StructDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter
  ) const override {
    // Replace the struct.def with a single mlir func with the signature and body of
    // @struct::@product
    std::string smt_func_name = ("smt_" + op.getSymName()).str();
    auto productFunc = op.getProductFuncOp();
    auto smtFunc =
        rewriter.create<func::FuncOp>(op->getLoc(), smt_func_name, productFunc.getFunctionType());
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
  APSInt prime;

public:
  ConstrainConverter(TypeConverter &_typeConverter, MLIRContext *context, APSInt _prime)
      : OpConversionPattern {_typeConverter, context, /*benefit=*/2}, prime {std::move(_prime)} {}

  LogicalResult matchAndRewrite(
      constrain::EmitEqualityOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    auto mod =
        rewriter.create<smt::IntConstantOp>(op->getLoc(), IntegerAttr::get(getContext(), prime));

    auto lhsMod = rewriter.create<smt::IntModOp>(op->getLoc(), adaptor.getLhs(), mod);
    auto rhsMod = rewriter.create<smt::IntModOp>(op->getLoc(), adaptor.getRhs(), mod);
    auto eq = rewriter.create<smt::EqOp>(op->getLoc(), lhsMod.getResult(), rhsMod.getResult());
    rewriter.replaceOpWithNewOp<smt::AssertOp>(op, eq.getResult());
    return success();
  }
};

class BoolCmpConverter : public OpConversionPattern<boolean::CmpOp> {
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
      break;
    }
    }
    return success();
  }
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
  Operation *convertBodies(Operation *op, const SignalSymbols &signalSymbols, APSInt prime) {
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
        BasicConverter<felt::SignedIntDivFeltOp, smt::IntDivOp>, FeltConstConverter,
        BoolCmpConverter, FunctionDefConverter, ReturnConverter, SCFIfConverter, YieldConverter>(
        typeConverter, context
    );
    patterns.add<FeltDivConverter>(typeConverter, context);
    patterns.add<ConstrainConverter>(typeConverter, context, prime);
    patterns.add<MemberWriteConverter>(typeConverter, context, signalSymbols, prime);
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

    mod.walk([this, prime](component::StructDefOp structDef) {
      // Start by adding declare-funcs for each signal
      IRRewriter rewriter {&getContext()};
      rewriter.setInsertionPointToStart(&structDef.getProductFuncOp().getFunctionBody().front());

      auto preamble = structDef.getProductFuncOp()->getLoc();

      // Insert symbols for each signal
      SignalSymbols signalSymbols;
      for (auto memberDef : structDef.getMemberDefs()) {
        auto constraintSym = rewriter.create<smt::DeclareFunOp>(
            preamble, smt::IntType::get(&getContext()),
            StringAttr::get(&getContext(), memberDef.getSymName() + "_c")
        );
        auto witnessSym = rewriter.create<smt::DeclareFunOp>(
            preamble, smt::IntType::get(&getContext()),
            StringAttr::get(&getContext(), memberDef.getSymName() + "_w")
        );
        signalSymbols[memberDef.getSymName()] = {constraintSym.getResult(), witnessSym.getResult()};
      }

      auto *result = convertFunction(convertBodies(structDef, signalSymbols, prime));
      if (result == nullptr) {
        signalPassFailure();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }
};

namespace smt {
std::unique_ptr<mlir::Pass> createSMTLoweringPass() { return std::make_unique<SMTLoweringPass>(); }
} // namespace smt

} // namespace llzk
