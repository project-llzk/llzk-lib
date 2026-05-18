//===-- Dialect.cpp - Dialect method implementations ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Struct/IR/Dialect.h"

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Versioning.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Shared/TypeConversionPatterns.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/TypeSwitch.h>

// TableGen'd implementation files
#include "llzk/Dialect/Struct/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/Struct/IR/Types.cpp.inc"

using namespace mlir;
using namespace llzk;
using namespace llzk::array;
using namespace llzk::component;
using namespace llzk::constrain;
using namespace llzk::function;
using namespace llzk::global;
using namespace llzk::polymorphic;

namespace {

/// Type converter that remaps StructType nameRefs per the migration FQN renaming table.
class V1StructNameTypeConverter : public TypeConverter {
  const DenseMap<SymbolRefAttr, SymbolRefAttr> &fqnMap;

public:
  explicit V1StructNameTypeConverter(const DenseMap<SymbolRefAttr, SymbolRefAttr> &renamingMap)
      : fqnMap(renamingMap) {

    addConversion([](Type t) { return t; });

    addConversion([this](StructType t) {
      auto it = fqnMap.find(t.getNameRef());
      SymbolRefAttr newRef = (it != fqnMap.end()) ? it->second : t.getNameRef();
      bool changed = (newRef != t.getNameRef());
      ArrayAttr params = t.getParams();
      if (params) {
        SmallVector<Attribute> updated;
        bool paramsChanged = false;
        for (Attribute a : params) {
          if (auto ta = dyn_cast<TypeAttr>(a)) {
            Type inner = convertType(ta.getValue());
            updated.push_back(TypeAttr::get(inner));
            paramsChanged |= (inner != ta.getValue());
          } else {
            updated.push_back(a);
          }
        }
        if (paramsChanged) {
          params = ArrayAttr::get(t.getContext(), updated);
          changed = true;
        }
      }
      return changed ? StructType::get(newRef, params) : t;
    });
  }
};

/// Pattern for CallOp that updates result types and remaps the callee symbol path to
/// account for the extra template nesting introduced by the migration struct wrapping.
class V1CallOpPattern : public OpConversionPattern<CallOp> {
  const DenseMap<SymbolRefAttr, SymbolRefAttr> &fqnMap;

public:
  V1CallOpPattern(
      TypeConverter &converter, MLIRContext *ctx,
      const DenseMap<SymbolRefAttr, SymbolRefAttr> &renamingMap
  )
      : OpConversionPattern<CallOp>(converter, ctx), fqnMap(renamingMap) {}

  LogicalResult matchAndRewrite(
      CallOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), newResultTypes))) {
      return op->emitError("Could not convert Op result types.");
    }
    // Remap callee if its path prefix matches a struct FQN that was wrapped.
    SymbolRefAttr calleeAttr = op.getCalleeAttr();
    SmallVector<FlatSymbolRefAttr> calleePieces = getPieces(calleeAttr);
    for (const auto &[oldFQN, newFQN] : fqnMap) {
      SmallVector<FlatSymbolRefAttr> oldPieces = getPieces(oldFQN);
      if (calleePieces.size() > oldPieces.size() &&
          std::equal(
              calleePieces.begin(), calleePieces.begin() + (ptrdiff_t)oldPieces.size(),
              oldPieces.begin()
          )) {
        SmallVector<FlatSymbolRefAttr> newPieces = getPieces(newFQN);
        newPieces.append(calleePieces.begin() + (ptrdiff_t)oldPieces.size(), calleePieces.end());
        calleeAttr = asSymbolRefAttr(newPieces);
        break;
      }
    }
    replaceOpWithNewOp<CallOp>(
        rewriter, op, newResultTypes, calleeAttr, adaptor.getMapOperands(),
        op.getNumDimsPerMapAttr(), adaptor.getArgOperands()
    );
    return success();
  }
};

// Prior to version 2, `StructDefOp` had a `const_params` attribute containing the list of struct
// parameters. In version 2, those parameters are represented explicitly as `poly.param` ops inside
// a `poly.template` that wraps the `StructDefOp`.
//
// If the `StructDefOp::readProperties()` function encounters the old `const_params` attribute, it
// stores them in a temporary `llzk::kV1ConstParamsAttr` attribute. This migration function creates
// a `poly.param` op for each such parameter and creates a `poly.template` to wrap these followed by
// the `StructDefOp`.
LogicalResult migrateToV2(Operation *rootOp) {
  // Ensure the Polymorphic dialect is loaded so we can create `poly.template` ops.
  rootOp->getContext()->loadDialect<polymorphic::PolymorphicDialect>();

  // Collect mappings from old to new FQN (fully-qualified name) for each updated struct.
  llvm::DenseMap<SymbolRefAttr, SymbolRefAttr> oldToNewFQN;

  // Visit all StructDefOp and perform the necessary transformation.
  rootOp->walk<WalkOrder::PreOrder>([&oldToNewFQN](StructDefOp structOp) -> WalkResult {
    Attribute constParamsAttr = structOp->getAttr(llzk::kV1ConstParamsAttr);
    if (!constParamsAttr) {
      return WalkResult::advance();
    }
    structOp->removeAttr(llzk::kV1ConstParamsAttr);

    // Create the TemplateOp at the position of the StructDefOp, using the
    // struct's own name (it becomes the outer template name).
    OpBuilder builder(structOp);
    auto templateOp =
        builder.create<polymorphic::TemplateOp>(structOp.getLoc(), structOp.getSymName());

    // Populate TemplateParamOps (in order) before the struct inside the template.
    Block &templateBody = templateOp.getBodyRegion().emplaceBlock();
    OpBuilder templateBuilder = OpBuilder::atBlockBegin(&templateBody);
    auto constParams = llvm::cast<ArrayAttr>(constParamsAttr);
    for (auto paramRef : constParams.getAsRange<FlatSymbolRefAttr>()) {
      templateBuilder.create<polymorphic::TemplateParamOp>(
          structOp.getLoc(), paramRef.getValue(),
          /*type_opt=*/TypeAttr {}
      );
    }

    // Compute the old FQN before the struct is moved into the template.
    SymbolRefAttr oldFQN = structOp.getFullyQualifiedName();

    // Move the StructDefOp into the template body (after the params).
    structOp->moveBefore(&templateBody, templateBody.end());

    // Record the FQN change: since the template name equals the old struct name, the
    // new FQN is the old FQN with the struct name appended as one more nesting level.
    oldToNewFQN[oldFQN] = appendLeaf(oldFQN, structOp.getSymNameAttr());

    // Skip descending into the now-moved StructDefOp.
    return WalkResult::skip();
  });

  // Done if no structs were updated.
  if (oldToNewFQN.empty()) {
    return success();
  }

  // Update all references to the old struct FQNs using the dialect conversion framework
  // so that every StructType and CallOp callee is updated if necessary.
  MLIRContext *ctx = rootOp->getContext();
  V1StructNameTypeConverter tyConv(oldToNewFQN);
  ConversionTarget target(*ctx);
  target.markUnknownOpDynamicallyLegal([&tyConv](Operation *op) {
    return defaultLegalityCheck(tyConv, op);
  });

  // Build pattern set for all LLZK op types. V1CallOpPattern (benefit 1) overrides
  // the default CallOpClassReplacePattern (benefit 0) for CallOp.
  RewritePatternSet patterns = newGeneralRewritePatternSet(tyConv, ctx, target);
  patterns.add<V1CallOpPattern>(tyConv, ctx, oldToNewFQN);
  return applyPartialConversion(rootOp, target, std::move(patterns));
}

} // namespace

//===------------------------------------------------------------------===//
// StructDialect
//===------------------------------------------------------------------===//

namespace llzk::component {

/// Implement version upgrade for StructDialect.
struct StructDialectBytecodeInterface : public LLZKDialectBytecodeInterface<StructDialect> {
  using LLZKDialectBytecodeInterface::LLZKDialectBytecodeInterface;

  LogicalResult upgradeFromVersion(
      mlir::Operation *rootOp, const LLZKDialectVersion &current,
      const LLZKDialectVersion &requested
  ) const override {
    assert(requested < current && "pre-condition");
    if (requested.majorVersion < 2) {
      if (failed(migrateToV2(rootOp))) {
        return failure();
      }
    }
    // Future migrations can be added here if necessary.
    return success();
  }
};

} // namespace llzk::component

auto llzk::component::StructDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/Struct/IR/Ops.cpp.inc"
  >();

  addTypes<
    #define GET_TYPEDEF_LIST
    #include "llzk/Dialect/Struct/IR/Types.cpp.inc"
  >();
  // clang-format on
  addInterfaces<StructDialectBytecodeInterface>();
}
