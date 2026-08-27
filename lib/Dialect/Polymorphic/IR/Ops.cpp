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
#include "llzk/Dialect/Verif/IR/Ops.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolTableLLZK.h"

#include <llvm/ADT/DenseSet.h>

// Include TableGen'd declarations
#include "llzk/Dialect/Polymorphic/IR/OpInterfaces.cpp.inc"

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Polymorphic/IR/Ops.cpp.inc"

using namespace mlir;
using namespace llzk::component;
using namespace llzk::verif;

namespace llzk::polymorphic {

bool isInTemplate(Operation *op) { return getParentOfType<TemplateOp>(op); }

FailureOr<TemplateOp> verifyInTemplate(Operation *op) {
  if (TemplateOp res = getParentOfType<TemplateOp>(op)) {
    return res;
  }
  return op->emitOpError() << "only valid within a '" << TemplateOp::getOperationName()
                           << "' ancestor";
}

/// Verify the optional transform-carried name pattern against current parameters.
///
/// The pattern is generic discardable metadata, so this owner boundary is the one place that
/// prevents malformed cardinality or element types from reaching refinement passes.
LogicalResult TemplateOp::verify() {
  Attribute rawPattern = (*this)->getDiscardableAttr(TEMPLATE_NAME_PATTERN_ATTR);
  if (!rawPattern) {
    return success();
  }

  auto pattern = llvm::dyn_cast<ArrayAttr>(rawPattern);
  if (!pattern) {
    return emitOpError() << "expected '" << TEMPLATE_NAME_PATTERN_ATTR << "' to be an ArrayAttr";
  }

  size_t parameterCount = numConstOps<TemplateParamOp>();
  size_t expectedChunkCount = parameterCount + 1;
  if (pattern.size() != expectedChunkCount) {
    return emitOpError() << "expected '" << TEMPLATE_NAME_PATTERN_ATTR << "' to contain "
                         << expectedChunkCount << " literal chunk(s) for " << parameterCount
                         << " template parameter(s), but found " << pattern.size();
  }
  for (size_t index = 0; index < pattern.size(); ++index) {
    if (!llvm::isa<StringAttr>(pattern[index])) {
      return emitOpError() << "expected '" << TEMPLATE_NAME_PATTERN_ATTR << "' element " << index
                           << " to be a StringAttr";
    }
  }
  return success();
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
  TemplateOp parentTemplate = getParentOfType<TemplateOp>(thisOp);
  assert(parentTemplate && "per ODS");
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
  if (succeeded(errorState)) {
    SymbolTable::walkSymbolTables(thisOp, /*allSymUsesVisible=*/true, checkUses);
  }
  return errorState;
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
  FailureOr<TemplateOp> getParentRes = getConstResolutionTemplate(tables, *this);
  if (failed(getParentRes)) {
    return failure(); // getConstResolutionTemplate() failure cases emit a sufficient error message
  }
  if (!*getParentRes) {
    return this->emitOpError() << "only valid within a '" << TemplateOp::getOperationName()
                               << "' ancestor or '" << ContractOp::getOperationName()
                               << "' that targets an operation with a '"
                               << TemplateOp::getOperationName() << "' ancestor";
  }
  // Ensure the named constant is a parameter of the parent struct
  FlatSymbolRefAttr name = this->getConstNameAttr();
  auto bindingOp = getParentRes->getConstNamed<TemplateSymbolBindingOpInterface>(name);
  if (!bindingOp) {
    return this->emitOpError()
        .append("references unknown symbol \"", name, '"')
        .attachNote(getParentRes->getLoc())
        .append("must reference a param or expr of this template");
  }
  // Ensure the type of the constant read matches the type of the referenced parameter (if any).
  if (std::optional<Type> paramType = bindingOp.getTypeOpt()) {
    if (llvm::isa<TypeVarType>(*paramType)) {
      return this->emitOpError().append(
          "cannot target \"", name, "\" because it is a type variable"
      );
    }
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

OpFoldResult ApplyMapOp::fold(FoldAdaptor adaptor) {
  SmallVector<Attribute> operands;
  operands.reserve(adaptor.getMapOperands().size());
  for (Attribute attr : adaptor.getMapOperands()) {
    if (!attr) {
      return {};
    }
    operands.push_back(attr);
  }

  SmallVector<Attribute> result;
  bool hasPoison = false;
  auto folded = getMap().constantFold(operands, result, &hasPoison);
  if (failed(folded) || hasPoison || result.size() != 1) {
    return {};
  }
  return result.front();
}

//===------------------------------------------------------------------===//
// UnifiableCastOp
//===------------------------------------------------------------------===//

/// Collect the result types of every unifiable-cast refinement reachable from `value`.
///
/// A unifiable cast preserves its runtime value, so cast chains are transparent for the purpose
/// of checking that separate uses do not make incompatible refinements.
static void collectReachableCastResultTypes(
    Value value, SmallVectorImpl<Type> &resultTypes, DenseSet<Value> &visited
) {
  if (!visited.insert(value).second) {
    return;
  }
  for (OpOperand &use : value.getUses()) {
    if (auto castOp = dyn_cast<UnifiableCastOp>(use.getOwner())) {
      resultTypes.push_back(castOp.getResult().getType());
      collectReachableCastResultTypes(castOp.getResult(), resultTypes, visited);
    }
  }
}

/// Return the first value in the unifiable-cast chain producing `value`.
static Value getUnifiableCastRoot(Value value) {
  while (auto castOp = value.getDefiningOp<UnifiableCastOp>()) {
    value = castOp.getInput();
  }
  return value;
}

LogicalResult UnifiableCastOp::verify() {
  auto inputType = getInput().getType();
  auto resultType = getResult().getType();
  if (!typesUnify(inputType, resultType)) {
    return emitOpError() << "input type " << inputType << " and output type " << resultType
                         << " are not unifiable";
  }
  // A unifiable cast is a reinterpretation, not a field conversion. In particular, it must not
  // erase a selected field and allow a subsequent cast to select a different field.
  if (!typesUnifyWithoutLosingFeltFields(inputType, resultType)) {
    return emitOpError() << "input type " << inputType << " and output type " << resultType
                         << " would discard a specified felt field";
  }

  // A type variable in an earlier cast may stand for a type containing a selected felt field.
  // Keep that field refinement from the cast-chain root when checking this result, so that
  // `felt<"bn128"> -> tvar -> felt<"babybear">` cannot reinterpret one runtime value under
  // two moduli.
  Type rootType = getUnifiableCastRoot(getInput()).getType();
  if (!typesUnifyWithoutLosingFeltFields(rootType, resultType)) {
    return emitOpError() << "input type " << rootType << " and output type " << resultType
                         << " would discard a specified felt field";
  }

  // All casts reachable through an input preserve the same runtime value. Their result types
  // must not select conflicting felt fields, otherwise an unspecified felt could be interpreted
  // under two distinct fields via sibling casts.
  SmallVector<Type> refinementTypes;
  DenseSet<Value> visited;
  collectReachableCastResultTypes(getUnifiableCastRoot(getInput()), refinementTypes, visited);
  for (Type refinementType : refinementTypes) {
    if (typesHaveConflictingFeltFields(resultType, refinementType)) {
      return emitOpError() << "result type " << resultType
                           << " conflicts with another unifiable cast refinement of the same value";
    }
  }

  return success();
}

} // namespace llzk::polymorphic
