//===-- Ops.cpp - Operation implementations ---------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/SymbolHelper.h"

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Polymorphic/IR/Ops.cpp.inc"

using namespace mlir;
using namespace llzk::component;

namespace llzk::polymorphic {

//===------------------------------------------------------------------===//
// TemplateOp
//===------------------------------------------------------------------===//

SmallVector<Attribute> TemplateOp::getParamNames() {
  return llvm::to_vector(llvm::map_range(getParamOps(), [](TemplateParamOp p) -> mlir::Attribute {
    return FlatSymbolRefAttr::get(p.getSymNameAttr());
  }));
}

bool TemplateOp::hasParamNamed(StringRef find) {
  auto status = this->walk([&](TemplateParamOp paramOp) {
    return (paramOp.getSymName() == find) ? WalkResult::interrupt() : WalkResult::advance();
  });
  return status.wasInterrupted();
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
  return checkForNameConflict(tables, *this);
}

LogicalResult TemplateExprOp::verifyRegions() {
  mlir::Region &region = getInitializerRegion();
  if (!region.hasOneBlock()) {
    return emitOpError("expected initializer region with a single block");
  }
  YieldOp yieldOp = llvm::dyn_cast<YieldOp>(region.back().getTerminator());
  if (!yieldOp) {
    return emitOpError("expected initializer region to end with a '")
           << YieldOp::getOperationName() << '\'';
  }

  // TODO: VERIFY: An `poly.expr` symbol cannot be used within its own region.
  // TODO: VERIFY: Cannot have cyclic definitions between expr regions.
  // Both of these could be covered by simply ensuring that `poly.expr` symbols
  // cannot be used at all within the initializer region.

  return success();
}

Type TemplateExprOp::getType() {
  mlir::Region &region = getInitializerRegion();
  assert(region.hasOneBlock() && "per `verifyRegions()`");
  YieldOp yieldOp = llvm::dyn_cast<YieldOp>(region.back().getTerminator());
  assert(yieldOp && "per `verifyRegions()`");
  return yieldOp.getVal().getType();
}

//===------------------------------------------------------------------===//
// ConstReadOp
//===------------------------------------------------------------------===//

LogicalResult ConstReadOp::verifySymbolUses(SymbolTableCollection &tables) {
  FailureOr<StructDefOp> getParentRes = verifyInStruct(*this);
  if (failed(getParentRes)) {
    return failure(); // verifyInStruct() already emits a sufficient error message
  }
  // Ensure the named constant is a parameter of the parent struct
  if (!getParentRes->hasParamNamed(this->getConstNameAttr())) {
    return this->emitOpError()
        .append("references unknown symbol \"", this->getConstNameAttr(), '"')
        .attachNote(getParentRes->getLoc())
        .append("must reference a parameter of this struct");
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
