//===-- FlatteningContext.h -------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Private state shared by flattening phases.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Types.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Interfaces/LoopLikeInterface.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

#include <cstdint>
#include <optional>
#include <tuple>

namespace llzk::polymorphic::detail::flattening {

/// Cache function instantiations by the source operation and exact concrete parameter bindings.
///
/// The instantiated symbol path is recorded only after its clone has been inserted and converted,
/// so callers never reuse a partially-created function.
class FunctionInstantiationCache {
  struct Entry {
    mlir::ArrayAttr concreteParamKey;
    mlir::SymbolRefAttr functionPath;
  };

  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<Entry>> entries;

public:
  /// Return the instantiated function path for this exact source/key pair, if one was published.
  std::optional<mlir::SymbolRefAttr>
  lookup(function::FuncDefOp sourceFunc, mlir::ArrayAttr concreteParamKey) const;

  /// Publish a completed instantiation for this source/key pair.
  void record(
      function::FuncDefOp sourceFunc, mlir::ArrayAttr concreteParamKey,
      mlir::SymbolRefAttr functionPath
  );

  /// Drop cached operation handles before the operations that own them are erased.
  void clear() { entries.clear(); }
};

/// Per-invocation instantiation history, legal refinements, and fixpoint/unroll state.
class ConversionTracker {
  /// Tracks if some step performed a modification of the code such that another pass should be run.
  bool modified = false;
  /// Maps original remote (i.e., use site) type to new remote type.
  /// Note: The keys are always parameterized StructType and the values are no-parameter StructType.
  llvm::DenseMap<component::StructType, component::StructType> structInstantiations;
  /// Contains the reverse of mappings in `structInstantiations` for use in legal conversion check.
  llvm::DenseMap<component::StructType, component::StructType> reverseInstantiations;
  /// Tracks original free function definitions for which instantiated clones were created.
  llvm::DenseSet<mlir::SymbolRefAttr> funcInstantiations;
  /// Successful partial functions keyed by their source operation and exact concrete bindings.
  /// The rendered symbol names are only values; they are never used as cache identity.
  FunctionInstantiationCache partialFuncInstantiations;
  /// Maps new remote type (i.e., the values in 'structInstantiations') to a list of Diagnostic
  /// to report at the location(s) of the compute() that causes the instantiation to the StructType.
  llvm::DenseMap<component::StructType, llvm::SmallVector<mlir::Diagnostic>> delayedDiagnostics;
  /// Successful full functions keyed by their source operation and exact concrete bindings.
  /// Rendered symbol names are presentation-only and can collide for valid parameter values.
  /// This cache must outlive individual rewrite pattern instances because flattening re-runs
  /// Step 2 across fixpoint iterations.
  FunctionInstantiationCache fullFuncInstantiations;

  /// The per-loop expansion limit shared by unrolling and folding guards.
  uint64_t unrollIterationLimit;
  /// Avoid repeating a warning at the same source location across clones and fixpoint rounds.
  llvm::DenseSet<std::pair<mlir::Location, mlir::StringAttr>> unrollWarnings;

public:
  /// Initialize the expansion limit for this invocation of the flattening pass.
  explicit ConversionTracker(uint64_t limit) : modified(false), unrollIterationLimit(limit) {}

  /// Warn once per source location and reason when a loop cannot safely be expanded.
  bool shouldSkipUnroll(mlir::Operation *op);

  /// Preserve skipped loops, their bodies, and enclosing control flow that could erase them.
  /// Return false for a null operation.
  bool mustPreserveLoop(mlir::Operation *op);

  /// Protect skipped loops and their nested/enclosing control flow from greedy rewriting.
  /// Regionless body operations remain eligible for folding and type instantiation/propagation.
  llvm::DenseSet<mlir::Operation *> getPreservedLoopOps(mlir::ModuleOp modOp);

  /// Return whether the current flattening iteration has changed the IR.
  bool isModified() const { return modified; }

  /// Clear the per-iteration modification flag before starting the next iteration.
  void resetModifiedFlag() { modified = false; }

  /// Merge the modification status from one rewrite step into the iteration state.
  void updateModifiedFlag(bool currStepModified) { modified |= currStepModified; }

  /// Record a struct instantiation from the original use-site type to its cloned replacement type.
  void recordInstantiation(component::StructType oldType, component::StructType newType);

  /// Return the instantiated type of the given StructType, if any.
  std::optional<component::StructType> getInstantiation(component::StructType oldType) const;

  /// Return the original parameterized type that produced the given instantiated type, if any.
  std::optional<component::StructType> getPreimage(component::StructType newType) const;

  /// Record that the given free function was instantiated.
  void recordInstantiation(mlir::SymbolRefAttr funcName);

  /// Return the successfully converted partial function for this exact source/key pair, if any.
  std::optional<mlir::SymbolRefAttr> lookupPartialFuncInstantiation(
      function::FuncDefOp sourceFunc, mlir::ArrayAttr concreteParamKey
  ) const;

  /// Publish a successful partial conversion after insertion and body conversion have completed.
  void recordPartialFuncInstantiation(
      function::FuncDefOp sourceFunc, mlir::ArrayAttr concreteParamKey, TemplateOp templateOp,
      function::FuncDefOp functionOp
  );

  /// No partial-function cache entry is read after cleanup starts; release its operation names at
  /// that boundary so the tracker does not retain stale handles while cleanup erases templates.
  void clearPartialFuncInstantiations() { partialFuncInstantiations.clear(); }

  /// Collect the fully-qualified names of all structs and free functions that were instantiated.
  llvm::DenseSet<mlir::SymbolRefAttr> getInstantiatedDefinitionNames() const;

  /// Emit diagnostics delayed until a compute call has been rewritten to the instantiated type.
  void reportDelayedDiagnostics(component::StructType newType, function::CallOp caller);

  /// Return the mutable diagnostic queue associated with `newType`.
  llvm::SmallVector<mlir::Diagnostic> &delayedDiagnosticSet(component::StructType newType) {
    return delayedDiagnostics[newType];
  }

  /// Return the successfully converted full function for this exact source/key pair, if any.
  std::optional<mlir::SymbolRefAttr>
  getFullFuncInstantiation(function::FuncDefOp sourceFunc, mlir::ArrayAttr concreteParamKey) const;

  /// Publish a successful full conversion after insertion and body conversion have completed.
  void recordFullFuncInstantiation(
      function::FuncDefOp sourceFunc, mlir::ArrayAttr concreteParamKey,
      mlir::SymbolRefAttr instantiatedPath
  );

  /// Check if the type conversion is legal, i.e., the new type unifies with and is more concrete
  /// than the old type with additional allowance for the results of struct flattening conversions.
  bool isLegalConversion(mlir::Type oldType, mlir::Type newType, const char *patName) const;

  /// Check whether every corresponding pair in `oldTypes` and `newTypes` is a legal conversion.
  template <typename T, typename U>
  inline bool areLegalConversions(T oldTypes, U newTypes, const char *patName) const {
    return llvm::all_of(
        llvm::zip_equal(oldTypes, newTypes),
        [this, &patName](std::tuple<mlir::Type, mlir::Type> oldThenNew) {
      return this->isLegalConversion(std::get<0>(oldThenNew), std::get<1>(oldThenNew), patName);
    }
    );
  }
};

/// Compute a loop trip count independently of trusted annotations.
std::optional<llvm::APInt> getConstantTripCount(mlir::LoopLikeOpInterface loopOp);

} // namespace llzk::polymorphic::detail::flattening
