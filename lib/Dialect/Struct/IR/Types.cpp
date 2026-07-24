//===-- Types.cpp - Struct type implementations -----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Struct/IR/Types.h"

#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"

using namespace mlir;
using namespace llzk::polymorphic;

namespace llzk::component {

LogicalResult StructType::verify(
    function_ref<InFlightDiagnostic()> emitError, SymbolRefAttr /*nameRef*/, ArrayAttr params
) {
  return verifyStructTypeParams(wrapNonNullableInFlightDiagnostic(emitError), params);
}

FailureOr<SymbolLookupResult<StructDefOp>> StructType::getDefinition(
    SymbolTableCollection &symbolTable, Operation *op, bool reportMissing
) const {
  // First ensure this StructType passes verification
  ArrayAttr typeParams = this->getParams();
  if (failed(StructType::verify([op] { return op->emitError(); }, getNameRef(), typeParams))) {
    return failure();
  }
  // Perform lookup and ensure the symbol references a StructDefOp
  auto res = lookupTopLevelSymbol<StructDefOp>(symbolTable, getNameRef(), op, reportMissing);
  if (failed(res) || !res.value()) {
    if (reportMissing) {
      return op->emitError() << "could not find '" << StructDefOp::getOperationName()
                             << "' named \"" << getNameRef() << '"';
    } else {
      return failure();
    }
  }
  // If this StructType contains parameters, make sure the StructDefOp is within a TemplateOp with
  // the same number of params.
  if (typeParams) {
    TemplateOp parent = getParentOfType<TemplateOp>(*res.value());
    size_t numExpected = parent ? parent.numConstOps<TemplateParamOp>() : 0;
    if (typeParams.size() != numExpected) {
      return op->emitError() << '\'' << StructType::name << "' type has " << typeParams.size()
                             << " parameters but \"" << res.value().get().getSymName()
                             << "\" expects " << numExpected;
    }
    if (parent) {
      for (auto [paramOp, value] :
           llvm::zip_equal(parent.getConstOps<TemplateParamOp>(), typeParams.getValue())) {
        auto feltValue = llvm::dyn_cast<llzk::felt::FeltConstAttr>(value);
        std::optional<Type> restriction = paramOp.getTypeOpt();
        if (!feltValue || !restriction) {
          continue;
        }
        auto feltType = llvm::dyn_cast<llzk::felt::FeltType>(*restriction);
        if (!feltType || failed(feltValue.getMaterializedType(feltType))) {
          return op->emitError() << "instantiation value '" << value
                                 << "' is not compatible with parameter \"@" << paramOp.getName()
                                 << "\" type restriction " << *restriction;
        }
      }
    }
  }
  return res;
}

LogicalResult StructType::verifySymbolRef(SymbolTableCollection &symbolTable, Operation *op) {
  return getDefinition(symbolTable, op);
}

LogicalResult StructType::hasColumns(SymbolTableCollection &symbolTable, Operation *op) const {
  auto lookup = getDefinition(symbolTable, op);
  if (failed(lookup)) {
    return lookup;
  }
  return lookup->get().hasColumns();
}

} // namespace llzk::component
