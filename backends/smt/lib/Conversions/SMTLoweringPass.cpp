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

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
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
#include "llzk/Dialect/String/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/Field.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

#include <algorithm>
#include <utility>

#include "smt/Conversions/ConversionPasses.h"
#include "smt/Dialect/IR/SMTDialect.h"
#include "smt/Dialect/IR/SMTOps.h"
#include "smt/Dialect/IR/SMTTypes.h"

namespace llzk {
namespace smt {
#define GEN_PASS_DEF_SMTLOWERINGPASS
#include "smt/Conversions/ConversionPasses.h.inc"
} // namespace smt

using namespace mlir;

using SignalSymbols = DenseMap<StringRef, std::pair<Value, Value>>;
class LLZKToSMTTypeConverter : public TypeConverter {
public:
  LLZKToSMTTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](felt::FeltType) { return smt::IntType::get(ctx); });
    addConversion([this](array::ArrayType type) {
      auto elemType = convertType(type.getElementType());
      return array::ArrayType::get(elemType, type.getShape());
    });
  }
};

static inline bool containsFelt(Type type) {
  return TypeSwitch<Type, bool>(type)
      .Case<felt::FeltType>([](auto) { return true; })
      .Case<array::ArrayType>([](array::ArrayType type) {
    return containsFelt(type.getElementType());
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
            op.getResultTypes(),
            [](Type t) { return !static_cast<bool>(dyn_cast<component::StructType>(t)); }
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

class MemberWriteConverter : public OpConversionPattern<component::MemberWriteOp> {

  SignalSymbols symbols;
  APSInt prime;

public:
  MemberWriteConverter(
      TypeConverter &typeConverter, MLIRContext *context, const SignalSymbols &symbols, APSInt prime
  )
      : OpConversionPattern {typeConverter, context, /*benefit=*/2}, symbols {symbols},
        prime {std::move(prime)} {}
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

    // TODO: How do I get the field modulus
    auto equal = rewriter.create<smt::EqOp>(
        op->getLoc(), rewriter.create<smt::IntModOp>(op->getLoc(), witness, mod).getResult(),
        rewriter.create<smt::IntModOp>(op->getLoc(), adaptor.getVal(), mod).getResult()
    );
    rewriter.replaceOpWithNewOp<smt::AssertOp>(op, equal);

    return success();
  }
};

class MemberReadConverter : public OpConversionPattern<component::MemberReadOp> {
  SignalSymbols symbols;

public:
  MemberReadConverter(
      TypeConverter &typeConverter, MLIRContext *context, const SignalSymbols &symbols
  )
      : OpConversionPattern {typeConverter, context, /*benefit=*/2}, symbols {symbols} {}
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
  using OpConversionPattern<constrain::EmitEqualityOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      constrain::EmitEqualityOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    auto eq = rewriter.create<smt::EqOp>(op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOpWithNewOp<smt::AssertOp>(op, eq.getResult());
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
    target.addIllegalOp<component::MemberWriteOp, component::MemberReadOp>();
    target.addLegalOp<component::CreateStructOp>();
    target.addDynamicallyLegalOp<function::ReturnOp>([](function::ReturnOp op) {
      return llvm::none_of(op.getOperandTypes(), [](Type type) {
        return isa<component::StructType>(type);
      });
    });

    target.addDynamicallyLegalOp<function::FuncDefOp>([](function::FuncDefOp funcOp) {
      bool signatureLegal = !llvm::any_of(funcOp.getArgumentTypes(), containsFelt) &&
                            !llvm::any_of(funcOp.getResultTypes(), containsFelt);

      return signatureLegal;
    });

    patterns.add<
        BasicConverter<felt::AddFeltOp, smt::IntAddOp>,
        BasicConverter<felt::SubFeltOp, smt::IntSubOp>,
        BasicConverter<felt::MulFeltOp, smt::IntMulOp>, FunctionDefConverter, ReturnConverter,
        ConstrainConverter>(typeConverter, context);
    patterns.add<MemberWriteConverter>(typeConverter, context, signalSymbols, prime);
    patterns.add<MemberReadConverter>(typeConverter, context, signalSymbols);

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
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

    if (failed(collectFields(mod, fields))) {
      // TODO: take input as an argument
      return signalPassFailure();
    }

    if (fields.size() > 1) {
      mod.emitError() << "multiple fields unsupported";
      return signalPassFailure();
    }

    const auto &selectedField = *(fields.begin());
    auto prime = toAPSInt(selectedField.get().prime());

    mod.walk([this, prime](component::StructDefOp structDef) {
      // Start by adding declare-funcs for each signal
      IRRewriter rewriter {&getContext()};
      rewriter.setInsertionPointToStart(&structDef.getProductFuncOp().getFunctionBody().front());

      SignalSymbols signalSymbols;

      for (auto memberDef : structDef.getMemberDefs()) {
        auto constraintSym = rewriter.create<smt::DeclareFunOp>(
            structDef.getProductFuncOp()->getLoc(), smt::IntType::get(&getContext()),
            StringAttr::get(&getContext(), memberDef.getSymName() + "_c")
        );
        auto witnessSym = rewriter.create<smt::DeclareFunOp>(
            structDef.getProductFuncOp()->getLoc(), smt::IntType::get(&getContext()),
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
