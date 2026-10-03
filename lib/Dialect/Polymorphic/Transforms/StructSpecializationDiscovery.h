//===-- StructSpecializationDiscovery.h -------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Struct/IR/Ops.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>

#include <cstdint>

namespace llzk::polymorphic::detail {

/// A struct type that the caller must instantiate, paired with the source use
/// that needs it. A call inside a loop can require Child<0>, Child<1>, etc. at
/// the same site, so the source operation alone does not identify a request.
/// The kind distinguishes a rolled call result from nested type arguments at
/// that call. For an array element, familyType is the rolled struct type and
/// arrayIndices identifies the element requiring this specialization.
struct StructSpecializationRequest {
  enum class Kind { Plain, ArrayElement, RolledCallResult };

  mlir::Operation *site;
  component::StructType type;
  llvm::SmallVector<int64_t> arrayIndices;
  component::StructType familyType;
  Kind kind;
};

/// A rolled method call and the concrete specializations it may target. An
/// empty list is meaningful when a known array has an empty dimension.
struct RolledCallTargets {
  mlir::Operation *site;
  llvm::SmallVector<component::StructType> candidates;
};

/// Determine which concrete struct definitions are needed when instantiating
/// source with the supplied template arguments. For example, Parent<N> may
/// contain an array of Child<i> instances for 0 <= i < N. Evaluating that range
/// with N = 3 tells the caller to instantiate Child<0>, Child<1>, and Child<2>
/// while leaving the original loop intact.
///
/// Template parameters and expressions supply constants shared by the struct's
/// methods. Method arguments without known values remain unknown; intermediate
/// SSA values are computed separately for each method and loop iteration because
/// their values depend on that execution's inputs. Template-expression results
/// are stored in a private copy of bindings so the caller's inputs stay unchanged.
///
/// Supported input structure (assuming verified LLZK IR):
/// - Every method must have a defined, single-block body. Discovery visits every
///   method, even if no call to that method appears in the source.
/// - Template expressions are evaluated in declaration order. A known yielded
///   value becomes a binding available to subsequent expressions and methods.
/// - Visited operations may contain regions only for scf.if, scf.for and
///   scf.while. Other region operations and unstructured branches on explored
///   paths are unsupported.
/// - scf.if selects the branch when its condition is known. Otherwise both
///   branches must be supported, and a result stays known only if both branches
///   yield the same constant. Discovery does not retain alternative values or
///   infer facts from the branch condition.
/// - Every visited scf.for requires known signed bounds and a positive step,
///   representable as int64_t, with no induction overflow or unsignedCmp.
///   Iterations propagate known scalar iter_args; zero iterations return the
///   initial values. These requirements apply even when the loop contains no
///   struct uses.
/// - Every visited scf.while requires a known i1 condition at each check.
///   The condition region runs before the first iteration and after every body
///   yield. Its forwarded values become the body arguments or, when false, the
///   loop results. The shared step budget bounds nonterminating loops.
///
/// Values available for specialization:
/// - poly.read_const reads the supplied bindings and known template-expression
///   results. Method arguments start unknown.
/// - A region-free, memory-effect-free operation produces known values when all
///   operands are known and its fold hook returns attributes or known SSA values.
///   This includes foldable constants and arithmetic; purity alone is insufficient.
/// - Calls are inspected for result types but their bodies are not evaluated.
///   Call results, global reads (including constant globals), and POD, array and
///   struct-member reads remain unknown. Aggregate contents and writes are not
///   tracked. Unknown values may flow through ordinary computations, but cannot
///   supply a required specialization argument or loop bound.
///
/// Supported type dependencies:
/// - Member types, method signatures and visited operation result types are
///   inspected for struct uses, including uses nested in other types. Template
///   bindings substitute type variables, struct arguments and array dimensions.
///   Every requested struct argument must then be concrete.
/// - Affine arguments directly on a call's struct result type are evaluated from
///   the corresponding map-operand groups, whose values must be known integers.
/// - An affine struct nested in an array requires concrete nonnegative
///   dimensions. A bare affine operation result may remain rolled, as on
///   array.read or pod.read; a bare affine member is unsupported. POD records between
///   the array and struct preserve that array's indices. Each affine argument
///   must accept one input per dimension and fold to one non-poison result.
///   An empty dimension produces no requests but establishes a known empty set.
/// - Affine calls also contribute concrete specializations. A call with a rolled
///   struct argument receives the union from all arrays and affine calls with
///   the same rolled type in this source instance. A call with no known source
///   is diagnosed. Affine maps nested in struct type arguments are unsupported.
///
/// An unresolved required argument, unsupported explored structure or exhausted
/// step budget is diagnosed. Success describes dependencies of this source
/// instance; the caller must repeat discovery for newly instantiated definitions
/// to obtain the transitive set of specializations.
///
/// Results retain source parameterized types, including nested concrete type
/// arguments. Identity, caching, cloning, call retargeting and metadata belong to
/// the caller.
class StructSpecializationDiscovery {
public:
  using Bindings = llvm::DenseMap<mlir::Attribute, mlir::Attribute>;
  using Requests = llvm::SmallVector<StructSpecializationRequest>;
  using CallTargets = llvm::SmallVector<RolledCallTargets>;

  /// Each evaluated body operation and unique array index tuple consumes one
  /// step. The limit is shared by template expressions and all methods in one run.
  explicit StructSpecializationDiscovery(uint64_t maxSteps = 1000000) : limit(maxSteps) {}

  /// Evaluate enclosing template expressions in declaration order and return an
  /// extended copy of bindings for substitution into a concrete struct clone.
  /// If requested, return the steps left for discovering that clone's methods.
  mlir::FailureOr<Bindings> evaluateBindings(
      component::StructDefOp source, const Bindings &bindings, uint64_t *remainingSteps = nullptr
  ) const;

  /// Return ordered, deduplicated requests, or diagnose an unresolved dependency.
  /// Self references are omitted; the caller already owns the source instance.
  /// When supplied, visitedOperations receives explored operations so callers can
  /// retarget uses without inspecting branches excluded by constant conditions.
  /// callTargets receives the candidate sets for rolled method arguments.
  mlir::FailureOr<Requests> discover(
      component::StructDefOp source, const Bindings &bindings,
      llvm::SmallVectorImpl<mlir::Operation *> *visitedOperations = nullptr,
      CallTargets *callTargets = nullptr
  ) const;

private:
  uint64_t limit;
};

} // namespace llzk::polymorphic::detail
