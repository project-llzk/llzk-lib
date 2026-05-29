//===-- EmptyTemplateRemovalPass.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-drop-empty-templates` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Attrs.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "llzk/Util/Debug.h"

#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/Transforms/DialectConversion.h>

// Include the generated base pass class definitions.
namespace llzk::polymorphic {
#define GEN_PASS_DEF_EMPTYTEMPLATEREMOVALPASS
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h.inc"
} // namespace llzk::polymorphic

#include "SharedImpl.h"

#define DEBUG_TYPE "llzk-drop-empty-templates"

using namespace mlir;
using namespace llzk::array;
using namespace llzk::component;
using namespace llzk::function;
using namespace llzk::pod;
using namespace llzk::polymorphic;
using namespace llzk::polymorphic::detail;

namespace {

static inline bool hasEmptyParamList(StructType t) {
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

    addConversion([this](PodType inputTy) {
      // Recursively convert record types
      llvm::ArrayRef<RecordAttr> records = inputTy.getRecords();
      if (records.empty()) {
        return inputTy;
      }
      llvm::SmallVector<RecordAttr> newRecords;
      newRecords.reserve(records.size());
      MLIRContext *ctx = inputTy.getContext();
      for (RecordAttr attr : records) {
        newRecords.push_back(
            RecordAttr::get(ctx, attr.getName(), this->convertType(attr.getType()))
        );
      }
      return PodType::get(ctx, newRecords);
    });
  }
};

/// Delete templates with no struct or function definitions.
class DeleteNoDefTemplatePattern : public OpConversionPattern<TemplateOp> {
public:
  using OpConversionPattern<TemplateOp>::OpConversionPattern;

  static inline bool legal(TemplateOp op) {
    return llvm::any_of(op.getBodyRegion().getOps(), [](Operation &p) {
      return llvm::isa<StructDefOp, FuncDefOp>(p);
    });
  }

  LogicalResult match(TemplateOp op) const override { return failure(legal(op)); }

  void
  rewrite(TemplateOp op, TemplateOpAdaptor, ConversionPatternRewriter &rewriter) const override {
    LLVM_DEBUG({
      llvm::dbgs() << "found template with no struct or function definitions: " << op << '\n';
    });
    rewriter.eraseOp(op);
  }
};

/// Convert templates with no constant parameters or expressions into modules.
class ReplaceNoParamTemplatePattern : public OpConversionPattern<TemplateOp> {
public:
  using OpConversionPattern<TemplateOp>::OpConversionPattern;

  static inline bool legal(TemplateOp op) {
    return op.hasConstOps<TemplateSymbolBindingOpInterface>();
  }

  LogicalResult matchAndRewrite(
      TemplateOp op, TemplateOpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (legal(op)) {
      return failure();
    }
    LLVM_DEBUG({
      llvm::dbgs() << "found template with no constant parameters or expressions: " << op << '\n';
    });
    // Convert types within the current body.
    Region &currentBody = adaptor.getBodyRegion();
    if (failed(rewriter.convertRegionTypes(&currentBody, *getTypeConverter()))) {
      LLVM_DEBUG(llvm::dbgs() << "convertRegionTypes(currentBody) failed!\n");
      return failure();
    }
    // Insert new ModuleOp at location of the current template.
    ModuleOp newOp = rewriter.create<ModuleOp>(op.getLoc(), adaptor.getSymName());
    // Move the current body into the module and erase the now-empty template op.
    // First, clear body region of the new module to prepare for `inlineRegionBefore`.
    Region &newOpBody = newOp.getBodyRegion();
    if (!newOpBody.empty()) {
      rewriter.eraseBlock(&newOpBody.front());
    }
    rewriter.inlineRegionBefore(currentBody, newOpBody, newOpBody.end());
    rewriter.eraseOp(op);
    return success();
  }
};

class EmptyTemplateRemovalPass
    : public llzk::polymorphic::impl::EmptyTemplateRemovalPassBase<EmptyTemplateRemovalPass> {

  void runOnOperation() override {
    ModuleOp modOp = getOperation();
    MLIRContext *ctx = modOp.getContext();
    EmptyParamListStructTypeConverter tyConv;
    ConversionTarget target = newConverterDefinedTarget<>(tyConv, ctx);
    // Mark TemplateOp legal only if legal according to both patterns.
    target.addDynamicallyLegalOp<TemplateOp>([](TemplateOp op) {
      return DeleteNoDefTemplatePattern::legal(op) && ReplaceNoParamTemplatePattern::legal(op);
    });
    RewritePatternSet patterns = llzk::newGeneralRewritePatternSet(tyConv, ctx, target);
    // Try `DeleteNoDefTemplatePattern` first since full removal is better that replacement.
    patterns.add<DeleteNoDefTemplatePattern>(tyConv, ctx);
    patterns.add<ReplaceNoParamTemplatePattern>(tyConv, ctx);
    if (failed(applyFullConversion(modOp, target, std::move(patterns)))) {
      llvm::errs() << "HELLO FROM HERE\n";
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> llzk::polymorphic::createEmptyTemplateRemoval() {
  return std::make_unique<EmptyTemplateRemovalPass>();
};
