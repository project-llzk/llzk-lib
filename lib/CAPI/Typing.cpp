//===-- Typing.cpp - C API for llzk types -----------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Util/TypeHelper.h"

#include "llzk-c/Typing.h"

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/CAPI/Wrap.h>

using namespace llzk;
using namespace mlir;

void llzkAssertValidAttrForParamOfType(MlirAttribute attr) {
  assertValidAttrForParamOfType(unwrap(attr));
}

bool llzkIsValidType(MlirType type) { return isValidType(unwrap(type)); }

bool llzkIsValidColumnType(MlirType type, MlirOperation op) {
  SymbolTableCollection sc;
  return isValidColumnType(unwrap(type), sc, unwrap(op));
}

bool llzkIsValidGlobalType(MlirType type) { return isValidGlobalType(unwrap(type)); }

bool llzkIsValidEmitEqType(MlirType type) { return isValidEmitEqType(unwrap(type)); }

bool llzkIsValidConstReadType(MlirType type) { return isValidConstReadType(unwrap(type)); }

bool llzkIsValidArrayElemType(MlirType type) { return isValidArrayElemType(unwrap(type)); }

bool llzkIsValidArrayType(MlirType type) { return isValidArrayType(unwrap(type)); }

bool llzkIsConcreteType(MlirType type, bool allowStructParams) {
  return isConcreteType(unwrap(type), allowStructParams);
}

bool llzkHasAffineMapAttr(MlirType type) { return hasAffineMapAttr(unwrap(type)); }

bool llzkTypeParamsUnify(
    intptr_t numLhsParams, MlirAttribute const *lhsParams, intptr_t numRhsParams,
    MlirAttribute const *rhsParams
) {
  SmallVector<Attribute> lhsSto, rhsSto;
  return typeParamsUnify(
      unwrapList(numLhsParams, lhsParams, lhsSto), unwrapList(numRhsParams, rhsParams, rhsSto)
  );
}

bool llzkArrayAttrTypeParamsUnify(MlirAttribute lhsParams, MlirAttribute rhsParams) {
  return typeParamsUnify(
      llvm::cast<ArrayAttr>(unwrap(lhsParams)), llvm::cast<ArrayAttr>(unwrap(rhsParams))
  );
}

bool llzkTypesUnify(
    MlirType lhs, MlirType rhs, intptr_t nRhsReversePrefix, MlirStringRef const *rhsReversePrefix
) {
  SmallVector<StringRef> rhsReversePrefixSto;
  return llzk::typesUnify(
      unwrap(lhs), unwrap(rhs), unwrapList(nRhsReversePrefix, rhsReversePrefix, rhsReversePrefixSto)
  );
}

bool llzkIsMoreConcreteUnification(
    MlirType oldTy, MlirType newTy, bool (*knownOldToNew)(MlirType, MlirType, void *),
    void *userData
) {
  return isMoreConcreteUnification(
      unwrap(oldTy), unwrap(newTy), [knownOldToNew, userData](auto lhs, auto rhs) {
    return knownOldToNew(wrap(lhs), wrap(rhs), userData);
  }
  );
}

MlirAttribute llzkForceIntAttrType(MlirAttribute attr, MlirLocation loc) {
  auto emitErrorFn = [&loc]() { return InFlightDiagnosticWrapper(mlir::emitError(unwrap(loc))); };
  FailureOr<Attribute> forced = forceIntAttrType(unwrap(attr), emitErrorFn);
  return wrap(succeeded(forced) ? *forced : nullptr);
}
