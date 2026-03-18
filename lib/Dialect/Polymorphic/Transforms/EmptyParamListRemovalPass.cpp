//===-- EmptyParamListRemovalPass.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-drop-empty-params` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Struct/IR/Types.h"

#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/Transforms/DialectConversion.h>

// Include the generated base pass class definitions.
namespace llzk::polymorphic {
#define GEN_PASS_DEF_EMPTYPARAMLISTREMOVALPASS
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h.inc"
} // namespace llzk::polymorphic

#include "SharedImpl.h"

#define DEBUG_TYPE "llzk-drop-empty-params"

using namespace mlir;
using namespace llzk::array;
using namespace llzk::component;
using namespace llzk::polymorphic;
using namespace llzk::polymorphic::detail;

namespace {

inline bool hasEmptyParamList(StructType t) {
  if (ArrayAttr paramList = t.getParams()) {
    return paramList.empty();
  }
  return false;
}

/// Convert StructType with empty parameter list to one with no parameters.
class EmptyParamListStructTypeConverter : public TypeConverter {
public:
  EmptyParamListStructTypeConverter() : TypeConverter() {

    addConversion([](Type inputTy) { return inputTy; });

    addConversion([](StructType inputTy) -> StructType {
      return hasEmptyParamList(inputTy) ? StructType::get(inputTy.getNameRef()) : inputTy;
    });

    addConversion([this](ArrayType inputTy) {
      // Recursively convert element type
      return ArrayType::get(
          this->convertType(inputTy.getElementType()), inputTy.getDimensionSizes()
      );
    });
  }
};

class TemplateOpPattern : public OpConversionPattern<TemplateOp> {
public:
  using OpConversionPattern<TemplateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TemplateOp op, TemplateOpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (op.hasConstOps<ConstParamSymbolOpInterface>()) {
      return failure();
    }
    ModuleOp newOp = rewriter.create<ModuleOp>(op.getLoc(), adaptor.getSymName());
    // Clear body region of the new module
    Region &newOpBody = newOp.getBodyRegion();
    if (!newOpBody.empty()) {
      rewriter.eraseBlock(&newOpBody.front());
    }
    // Move the template body into the module
    Region &oldOpBody = adaptor.getBodyRegion();
    if (failed(rewriter.convertRegionTypes(&oldOpBody, *getTypeConverter()))) {
      return failure();
    }
    rewriter.inlineRegionBefore(oldOpBody, newOpBody, newOpBody.end());
    // Erase the now-empty TemplateOp
    rewriter.eraseOp(op);
    return success();
  }
};

/// Find all structs with empty parameter list and remove the parameter list altogether. Also,
/// replace StructType with an empty parameter list to have nullptr for the parameter list.
class EmptyParamRemovalPass
    : public llzk::polymorphic::impl::EmptyParamListRemovalPassBase<EmptyParamRemovalPass> {

  void runOnOperation() override {
    ModuleOp modOp = getOperation();
    MLIRContext *ctx = modOp.getContext();
    EmptyParamListStructTypeConverter tyConv;
    ConversionTarget target = newConverterDefinedTarget<>(tyConv, ctx);
    // Mark TemplateOp legal only if it has constant param or expr.
    target.addDynamicallyLegalOp<TemplateOp>([](TemplateOp op) {
      return op.hasConstOps<ConstParamSymbolOpInterface>();
    });
    RewritePatternSet patterns = newGeneralRewritePatternSet(tyConv, ctx, target);
    patterns.add<TemplateOpPattern>(tyConv, ctx);
    if (failed(applyFullConversion(modOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> llzk::polymorphic::createEmptyParamListRemoval() {
  return std::make_unique<EmptyParamRemovalPass>();
};
