//===-- Ops.cpp - Operation implementations ---------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Types.h"
#include "llzk/Dialect/Shared/OpHelpers.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolTableLLZK.h"

// Include TableGen'd declarations
#include "llzk/Dialect/Polymorphic/IR/OpInterfaces.cpp.inc"

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Polymorphic/IR/Ops.cpp.inc"

using namespace mlir;
using namespace llzk::component;

namespace llzk::polymorphic {

bool isInTemplate(Operation *op) { return getParentOfType<TemplateOp>(op); }

FailureOr<TemplateOp> verifyInTemplate(Operation *op) {
  if (TemplateOp res = getParentOfType<TemplateOp>(op)) {
    return res;
  }
  return op->emitOpError() << "only valid within a '" << TemplateOp::getOperationName()
                           << "' ancestor";
}

//===------------------------------------------------------------------===//
// TemplateParamOp
//===------------------------------------------------------------------===//

namespace {

LogicalResult checkForNameConflict(SymbolTableCollection &tables, SymbolOpInterface op) {
  // Ensure parameter name does not conflict with an existing top-level symbol
  // because that would cause an ambiguity in symbol resolution within structs.
  auto res = lookupTopLevelSymbol(tables, FlatSymbolRefAttr::get(op.getNameAttr()), op, false);
  if (succeeded(res)) {
    return op.emitOpError()
        .append("name conflicts with an existing symbol")
        .attachNote(res->get()->getLoc())
        .append("symbol already defined here");
  }
  return success();
}

} // namespace

LogicalResult TemplateParamOp::verifySymbolUses(SymbolTableCollection &tables) {
  return checkForNameConflict(tables, *this);
}

//===------------------------------------------------------------------===//
// TemplateExprOp
//===------------------------------------------------------------------===//

LogicalResult TemplateExprOp::verifySymbolUses(SymbolTableCollection &tables) {
  if (failed(checkForNameConflict(tables, *this))) {
    return failure(); // checkForNameConflict() already emits a sufficient error message
  }
  // Ensure no symbol used within the initializer region is defined via a `TemplateExprOp`.
  // This prevents cyclic definitions of `TemplateExprOp`. Searches all symbol uses within
  // this op and also within any nested symbol tables.
  Operation *thisOp = this->getOperation();
  if (TemplateOp parentTemplate = getParentOfType<TemplateOp>(thisOp)) {
    LogicalResult errorState = success();
    auto checkUses = [this, &parentTemplate, &errorState](Operation *symTableOp, bool) {
      if (auto uses = llzk::getSymbolUses(symTableOp)) {
        for (SymbolTable::SymbolUse use : uses.value()) {
          // Only need to check flat refs since `TemplateExprOp` refs must be flat
          auto usedSym = llvm::dyn_cast<FlatSymbolRefAttr>(use.getSymbolRef());
          if (usedSym && parentTemplate.hasConstNamed<TemplateExprOp>(usedSym)) {
            InFlightDiagnostic diag = this->emitOpError().append(
                "initialization cannot use a symbol defined by another `",
                TemplateExprOp::getOperationName(), "` within this template"
            );
            diag.attachNote(use.getUser()->getLoc()).append("symbol ", usedSym, " used here");
            auto def = parentTemplate.getConstNamed<TemplateExprOp>(usedSym);
            diag.attachNote(def.getLoc()).append("defined here");
            errorState = diag; // transformation to LogicalResult reports the error
            return;
          }
        }
      }
    };
    checkUses(thisOp, true);
    if (failed(errorState)) {
      return errorState;
    }
    SymbolTable::walkSymbolTables(thisOp, /*allSymUsesVisible=*/true, checkUses);
    if (failed(errorState)) {
      return errorState;
    }
  }
  return success();
}

LogicalResult TemplateExprOp::verifyRegions() {
  Region &region = getInitializerRegion();
  if (!region.hasOneBlock()) {
    return emitOpError("expected initializer region with a single block");
  }
  Block &block = region.back();
  if (!llvm::isa<YieldOp>(block.getTerminator())) {
    return emitOpError("expected initializer region to end with a '")
           << YieldOp::getOperationName() << '\'';
  }
  // Check or ops with side-effects that are not allowed within `poly.expr`.
  Operation *illegalOp = nullptr;
  auto walkRes = block.walk([&illegalOp](Operation *p) {
    // Note: If side-effect traits are added to ops in the future, this check should
    // be updated to check for those traits instead of specific op types.
    if (llvm::isa<global::GlobalRefOpInterface, function::CallOp>(p)) {
      illegalOp = p;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkRes.wasInterrupted()) {
    assert(illegalOp); // was set in the walk above
    return illegalOp->emitOpError().append(
        "is not allowed within a `", TemplateExprOp::getOperationName(), "` initializer"
    );
  }
  return success();
}

Type TemplateExprOp::getType() {
  Region &region = getInitializerRegion();
  assert(region.hasOneBlock() && "per `verifyRegions()`");
  YieldOp yieldOp = llvm::dyn_cast<YieldOp>(region.back().getTerminator());
  assert(yieldOp && "per `verifyRegions()`");
  return yieldOp.getVal().getType();
}

std::optional<Type> TemplateExprOp::getTypeOpt() { return getType(); }

//===------------------------------------------------------------------===//
// ConstReadOp
//===------------------------------------------------------------------===//

LogicalResult ConstReadOp::verifySymbolUses(SymbolTableCollection &tables) {
  FailureOr<TemplateOp> getParentRes = verifyInTemplate(*this);
  if (failed(getParentRes)) {
    return failure(); // verifyInTemplate() already emits a sufficient error message
  }
  // Ensure the named constant is a parameter of the parent struct
  FlatSymbolRefAttr name = this->getConstNameAttr();
  auto constParam = getParentRes->getConstNamed<ConstParamSymbolOpInterface>(name);
  if (!constParam) {
    return this->emitOpError()
        .append("references unknown symbol \"", name, '"')
        .attachNote(getParentRes->getLoc())
        .append("must reference a param or expr of this template");
  }
  // Ensure the type of the constant read matches the type of the referenced parameter (if any).
  if (std::optional<Type> paramType = constParam.getTypeOpt()) {
    if (this->getType() != *paramType) {
      return this->emitOpError().append(
          "type ", this->getType(), " does not match constant param type ", *paramType
      );
    }
  }

  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(tables, *this, getType());
}

//===------------------------------------------------------------------===//
// ApplyMapOp
//===------------------------------------------------------------------===//

LogicalResult ApplyMapOp::verify() {
  // Check input and output dimensions match.
  AffineMap map = getMap();

  // Verify that the map only produces one result.
  if (map.getNumResults() != 1) {
    return emitOpError("must produce exactly one value");
  }

  // Verify that operand count matches affine map dimension and symbol count.
  unsigned mapDims = map.getNumDims();
  if (getNumOperands() != mapDims + map.getNumSymbols()) {
    return emitOpError("operand count must equal affine map dimension+symbol count");
  } else if (mapDims != getNumDimsAttr().getInt()) {
    return emitOpError("dimension operand count must equal affine map dimension count");
  }

  return success();
}

//===------------------------------------------------------------------===//
// UnifiableCastOp
//===------------------------------------------------------------------===//

LogicalResult UnifiableCastOp::verify() {
  if (!typesUnify(getInput().getType(), getResult().getType())) {
    return emitOpError() << "input type " << getInput().getType() << " and output type "
                         << getResult().getType() << " are not unifiable";
  }

  return success();
}

} // namespace llzk::polymorphic
