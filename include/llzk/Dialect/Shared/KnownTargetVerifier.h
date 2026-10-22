//===-- KnownTargetVerifier.h - Shared call verification --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Shared/CallLikeOpInterfaces.h"
#include "llzk/Util/Compare.h"
#include "llzk/Util/ErrorHelper.h"
#include "llzk/Util/SymbolLookup.h"
#include "llzk/Util/SymbolTableLLZK.h"
#include "llzk/Util/TypeHelper.h"

#include <llvm/ADT/DenseSet.h>

namespace llzk {

/// Shared signature and template-instantiation checks for a resolved call-like target.
/// OriginOp must provide the LLZKCallLikeOpInterface methods, template parameters,
/// and a callee attribute. TargetOp must provide a function type and symbol name.
/// Operation-specific eligibility checks and error aggregation remain with the caller.
template <typename OriginOp, typename TargetOp> class KnownTargetVerifier {
public:
  KnownTargetVerifier(OriginOp *op, const SymbolLookupResult<TargetOp> &target)
      : origin(op), tgt(*target), tgtType(tgt.getFunctionType()),
        targetNamespace(target.getNamespace()), targetViaInclude(target.viaInclude()) {}

  /// Return the resolved target for operation-specific checks.
  TargetOp getTarget() const { return tgt; }

  /// Return the target signature used for input and output verification.
  mlir::FunctionType getTargetType() const { return tgtType; }

  /// Check arity and namespace-aware type compatibility, emitting call-site diagnostics.
  template <typename T>
  mlir::LogicalResult verifyTypesMatch(
      mlir::ValueTypeRange<T> originTypes, mlir::ArrayRef<mlir::Type> tgtTypes, const char *aspect
  ) {
    if (tgtTypes.size() != originTypes.size()) {
      return origin->emitOpError()
          .append("incorrect number of ", aspect, "s for callee, expected ", tgtTypes.size())
          .attachNote(tgt.getLoc())
          .append("callee defined here");
    }
    for (unsigned i = 0, e = tgtTypes.size(); i != e; ++i) {
      if (!typesUnify(originTypes[i], tgtTypes[i], targetNamespace)) {
        auto diag =
            origin->emitOpError().append(aspect, " type mismatch: expected type ", tgtTypes[i]);
        if (targetViaInclude) {
          diag.append(" from included target \"", origin->getCalleeAttr(), '"');
        }
        return diag.append(", but found ", originTypes[i], " for ", aspect, " number ", i);
      }
    }
    return mlir::success();
  }

  /// Verify instantiations against the enclosing template. Explicit values are optional
  /// only when every template parameter appears in the target signature; provided
  /// values must agree with both declared restrictions and signature inference.
  mlir::LogicalResult verifyTemplateInstantiation(polymorphic::TemplateOp tgtOpParent) {
    auto realParams = tgtOpParent.getConstOps<polymorphic::TemplateParamOp>();
    mlir::ArrayAttr callParams = origin->getTemplateParamsAttr();

    // When there is no instantiation list, just ensure that it's not required.
    if (isNullOrEmpty(callParams)) {
      llvm::SmallDenseSet<mlir::SymbolRefAttr> referencedInSignature;
      llzk::getSymbolsUsedIn(tgtType.getInputs(), referencedInSignature);
      llzk::getSymbolsUsedIn(tgtType.getResults(), referencedInSignature);

      bool allParamsReferenced = llvm::all_of(realParams, [&](polymorphic::TemplateParamOp p) {
        return referencedInSignature.contains(mlir::FlatSymbolRefAttr::get(p.getNameAttr()));
      });
      if (allParamsReferenced) {
        return mlir::success();
      }
      return origin->emitOpError().append(
          "must provide template instantiation parameters when calling \"@", tgt.getSymName(),
          "\" because not all template parameters of \"@", tgtOpParent.getSymName(),
          "\" appear in the function type signature"
      );
    }

    // Ensure `forceIntAttrTypes()` was successful on the call site's template parameters.
    if (mlir::failed(llzk::forceIntAttrTypes(callParams.getValue(), [this] {
      return llzk::InFlightDiagnosticWrapper(this->origin->emitOpError());
    }))) {
      return mlir::failure();
    }

    // The instantiation list is present. Check it has exactly one entry per template param.
    size_t numTemplateParams = llvm::range_size(realParams);
    if (callParams.size() != numTemplateParams) {
      return origin->emitOpError().append(
          "template instantiation has ", callParams.size(), " parameter(s) but \"@",
          tgtOpParent.getSymName(), "\" expects ", numTemplateParams, " template parameter(s)"
      );
    }

    // Check type compatibility of each provided value with the declared parameter type (if any).
    if (mlir::failed(origin->verifyTemplateParamValuesCompatibility(realParams))) {
      return mlir::failure();
    }

    // Check that the provided instantiation values are consistent with what type unification
    // of the target function types against the call's operand and result types would determine.
    mlir::FailureOr<UnificationMap> unifyResult =
        origin->unifyTypeSignatureWithNamespace(tgtType, targetNamespace);
    // Verification continues after input/output errors to aggregate diagnostics.
    if (mlir::failed(unifyResult)) {
      return mlir::failure();
    }
    return origin->verifyTemplateParamsMatchInferred(realParams, unifyResult.value());
  }

private:
  OriginOp *origin;
  TargetOp tgt;
  mlir::FunctionType tgtType;
  std::vector<llvm::StringRef> targetNamespace;
  bool targetViaInclude;
};

} // namespace llzk
