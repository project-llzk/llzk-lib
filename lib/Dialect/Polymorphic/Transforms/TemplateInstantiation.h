//===-- TemplateInstantiation.h ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Private parameter substitution helpers shared by polymorphic transformations.
/// Callers own cloning, symbol insertion, specialization identity and scheduling.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/Diagnostics.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>

#include <optional>

namespace llzk::polymorphic::detail {

/// Substitute known template bindings in function types and inference candidates.
/// Unbound parameters are preserved for partial instantiation. This converter owns
/// its bindings and flattens array element types that become arrays themselves.
class TemplateTypeConverter : public mlir::TypeConverter {
  llvm::DenseMap<mlir::Attribute, mlir::Attribute> paramNameToValue;
  mlir::Attribute convertIfPossible(mlir::Attribute attr) const;

public:
  explicit TemplateTypeConverter(llvm::DenseMap<mlir::Attribute, mlir::Attribute> bindings);

  /// Substitute a bound attribute or the type contained in a TypeAttr.
  mlir::Attribute convertAttr(mlir::Attribute attr) const;

  /// Return whether the parameter has a supplied binding.
  bool containsParam(mlir::Attribute name) const { return paramNameToValue.contains(name); }

  /// Return the bindings used by body substitution patterns.
  const llvm::DenseMap<mlir::Attribute, mlir::Attribute> &getParamMap() const {
    return paramNameToValue;
  }
};

/// Re-spell struct types nested in \p templateParams for \p destinationRoot, resolving
/// their current names from \p lookupFrom. Numeric and other value parameters are
/// preserved. A null list becomes an empty list. Reports inaccessible type
/// parameters and unsupported includes at \p requestSite, where instantiation was
/// requested, so moving bindings does not move their diagnostics into a clone.
mlir::FailureOr<mlir::ArrayAttr> rebaseTemplateParams(
    mlir::SymbolTableCollection &tables, mlir::ArrayAttr templateParams,
    mlir::Operation *lookupFrom, mlir::ModuleOp destinationRoot, mlir::Operation *requestSite
);

/// Materialize a template value as a constant of the requested type, or return null
/// when no supported constant operation can represent it. Callers diagnose lossy
/// conversions, including nonzero integers interpreted as booleans.
mlir::Value materializeTemplateConstant(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Type resultType, mlir::Attribute value
);

/// Report diagnostics emitted while substituting a clone, preserving notes and
/// moving notes with unknown locations to the instantiation site.
void reportDelayedDiagnostics(
    mlir::Operation *site, llvm::SmallVector<mlir::Diagnostic> &&diagnostics
);

/// Rewrite callees rooted at a type parameter using that parameter's struct binding.
/// Other callees and explicit call arguments are left unchanged.
void convertCalleesInPlace(
    mlir::Operation *op, const llvm::DenseMap<mlir::Attribute, mlir::Attribute> &paramNameToValue
);

/// Add foldable template-expression results to bindings in declaration order.
/// Expressions with unresolved operands are skipped to allow partial instantiation.
void evaluateTemplateExprs(
    TemplateOp templateOp, llvm::DenseMap<mlir::Attribute, mlir::Attribute> &paramNameToConcrete
);

/// Return the callee-side unification-derived value for a template parameter, if any.
std::optional<mlir::Attribute>
inferUnifiedParam(const UnificationMap &unifyResult, mlir::SymbolRefAttr paramName);

/// Substitute bindings and self-type references in an already inserted struct clone.
/// Append materialization warnings for the caller to report at each instantiation site.
/// Does not clone, insert symbols, infer parameters, inline calls or unroll loops.
mlir::LogicalResult substituteStructBody(
    component::StructDefOp newStruct, component::StructType typeAtDef,
    const llvm::DenseMap<mlir::Attribute, mlir::Attribute> &paramNameToConcrete,
    llvm::SmallVector<mlir::Diagnostic> &diagnostics
);

/// Substitute bindings in a function clone, including constants and array accesses.
/// Append materialization warnings without reporting them. The caller must report
/// them and verify nested call targets after applying its own symbol rewrites.
/// Does not clone, insert symbols, infer parameters, inline calls or unroll loops.
mlir::LogicalResult substituteFunctionBody(
    function::FuncDefOp newFunc,
    const llvm::DenseMap<mlir::Attribute, mlir::Attribute> &paramNameToConcrete,
    llvm::SmallVector<mlir::Diagnostic> &delayedDiagnostics
);

/// Substitute bindings in a copied template expression, including constant reads and dependent
/// operation types and attributes. Append materialization warnings for the caller to report at
/// the instantiation site. The caller owns insertion and erasing the clone on failure.
mlir::LogicalResult substituteTemplateExprBody(
    TemplateExprOp expr,
    const llvm::DenseMap<mlir::Attribute, mlir::Attribute> &paramNameToConcrete,
    llvm::SmallVector<mlir::Diagnostic> &delayedDiagnostics
);

} // namespace llzk::polymorphic::detail
