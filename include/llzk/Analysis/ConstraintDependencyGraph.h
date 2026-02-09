//===-- ConstraintDependencyGraph.h -----------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Analysis/AnalysisWrappers.h"
#include "llzk/Analysis/SourceRef.h"
#include "llzk/Analysis/SourceRefLattice.h"
#include "llzk/Dialect/Array/IR/Ops.h"

#include <mlir/Pass/AnalysisManager.h>

#include <llvm/ADT/EquivalenceClasses.h>

namespace mlir {

class DataFlowSolver;

} // namespace mlir

namespace llzk {

using SourceRefRemappings = std::vector<std::pair<SourceRef, SourceRefLatticeValue>>;

/// @brief The dataflow analysis that computes the set of references that
/// LLZK operations use and produce. The analysis is simple: any operation will
/// simply output a union of its input references, regardless of what type of
/// operation it performs, as the analysis is operator-insensitive.
class SourceRefAnalysis : public dataflow::DenseForwardDataFlowAnalysis<SourceRefLattice> {
public:
  using dataflow::DenseForwardDataFlowAnalysis<SourceRefLattice>::DenseForwardDataFlowAnalysis;

  void visitCallControlFlowTransfer(
      mlir::CallOpInterface call, dataflow::CallControlFlowAction action,
      const SourceRefLattice &before, SourceRefLattice *after
  ) override;

  /// @brief Propagate `SourceRef` lattice values from operands to results.
  /// @param op
  /// @param before
  /// @param after
  mlir::LogicalResult visitOperation(
      mlir::Operation *op, const SourceRefLattice &before, SourceRefLattice *after
  ) override;

protected:
  void setToEntryState(SourceRefLattice *lattice) override {
    // the entry state is empty, so do nothing.
  }

  // Perform a standard union of operands into the results value.
  mlir::ChangeResult fallbackOpUpdate(
      mlir::Operation *op, const SourceRefLattice::ValueMap &operandVals,
      const SourceRefLattice &before, SourceRefLattice *after
  );

  // Perform the update for either a array.read op or an array.extract op, which
  // operate very similarly: index into the first operand using a variable number
  // of provided indices.
  void arraySubdivisionOpUpdate(
      array::ArrayAccessOpInterface op, const SourceRefLattice::ValueMap &operandVals,
      const SourceRefLattice &before, SourceRefLattice *after
  );

private:
  mlir::SymbolTableCollection tables;
};

/// @brief Parameters and shared objects to pass to child analyses.
struct CDGAnalysisContext {
  bool runIntraprocedural = false;

  bool runIntraproceduralAnalysis() const { return runIntraprocedural; }

  friend bool operator==(const CDGAnalysisContext &a, const CDGAnalysisContext &b) = default;
};

} // namespace llzk

template <> struct std::hash<llzk::CDGAnalysisContext> {
  size_t operator()(const llzk::CDGAnalysisContext &c) const {
    return std::hash<bool> {}(c.runIntraprocedural);
  }
};

namespace llzk {

/// @brief A dependency graph of constraints enforced by an LLZK struct.
///
/// Mathematically speaking, a constraint dependency graph (CDG) is a transitive closure
/// of edges where there is an edge between signals `a` and `b`
/// iff `a` and `b` appear in the same constraint.
///
/// Less formally, a CDG is a set of signals that constrain one another through
/// one or more emit operations (`constrain.in` or `constrain.eq`). The CDG only
/// indicate that signals are connected by constraints, but do not include information
/// about the type of computation that binds them together.
///
/// For example, a CDG of the form: {
///     {%arg1, %arg2, %arg3[@foo]}
/// }
/// Means that %arg1, %arg2, and member @foo of %arg3, are connected
/// via some constraints. These constraints could take the form of (in Circom notation):
///     %arg1 + %arg3[@foo] === %arg2
/// Or
///     %arg2 === %arg2 / %arg3[@foo]
/// Or any other form of constraint including those values.
///
/// The CDG also records information about constant values (e.g., felt.const) that
/// are included in constraints, but does not compute a transitive closure over
/// constant values, as constant value usage in constraints does not imply any
/// dependency between signal values (e.g., constraints a + b === 0 and c + d === 0 both use
/// constant 0, but does not enforce a dependency between a, b, c, and d).
class ConstraintDependencyGraph {
public:
  /// @brief Compute a ConstraintDependencyGraph (CDG)
  /// @param mod The LLZK-complaint module that is the parent of struct `s`.
  /// @param s The struct to compute the CDG for.
  /// @param solver A pre-configured DataFlowSolver. The liveness of the struct must
  /// already be computed in this solver in order for the constraint analysis to run.
  /// @param am A module-level analysis manager. This analysis manager needs to originate
  /// from a module-level analysis (i.e., for the `mod` module) so that analyses
  /// for other constraints can be queried via the getChildAnalysis method.
  /// @param ctx The analysis context.
  /// @return
  static mlir::FailureOr<ConstraintDependencyGraph> compute(
      mlir::ModuleOp mod, component::StructDefOp s, mlir::DataFlowSolver &solver,
      mlir::AnalysisManager &am, const CDGAnalysisContext &ctx
  );

  /// @brief Dumps the CDG to stderr.
  void dump() const;
  /// @brief Print the CDG to the specified output stream.
  /// @param os The LLVM/MLIR output stream.
  void print(mlir::raw_ostream &os) const;

  /// @brief Translate the SourceRefs in this CDG to that of a different
  /// context. Used to translate a CDG of a struct to a CDG for a called subcomponent.
  /// @param translation A vector of mappings of current reference prefix -> translated reference
  /// prefix.
  /// @return A CDG that contains only translated references. Non-constant references with
  /// no translation are omitted. This omissions allows calling components to ignore internal
  /// references within subcomponents that are inaccessible to the caller.
  ConstraintDependencyGraph translate(SourceRefRemappings translation) const;

  /// @brief Get the values that are connected to the given ref via emitted constraints.
  /// This method looks for constraints to the value in the ref and constraints to any
  /// prefix of this value.
  /// For example, if ref is an array element (foo[2]), this looks for constraints on
  /// foo[2] as well as foo, as arrays may be constrained in their entirety via constrain.in
  /// operations.
  /// @param ref
  /// @return The set of references that are connected to ref via constraints.
  SourceRefSet getConstrainingValues(const SourceRef &ref) const;

  const SourceRefLattice::Ref2Val &getRef2Val() const { return ref2Val; }

  /*
  Rule of three, needed for the mlir::SymbolTableCollection, which has no copy constructor.
  Since the mlir::SymbolTableCollection is a caching mechanism, we simply allow default, empty
  construction for copies.
  */

  ConstraintDependencyGraph(const ConstraintDependencyGraph &other)
      : mod(other.mod), structDef(other.structDef), ctx(other.ctx), signalSets(other.signalSets),
        constantSets(other.constantSets), ref2Val(other.ref2Val), tables() {}

  ConstraintDependencyGraph &operator=(const ConstraintDependencyGraph &other) {
    mod = other.mod;
    structDef = other.structDef;
    ctx = other.ctx;
    signalSets = other.signalSets;
    constantSets = other.constantSets;
    ref2Val = other.ref2Val;
    return *this;
  }
  virtual ~ConstraintDependencyGraph() = default;

private:
  mlir::ModuleOp mod;
  // Using mutable because many operations are not const by default, even for "const"-like
  // operations, like "getName()", and this reduces const_casts.
  mutable component::StructDefOp structDef;
  CDGAnalysisContext ctx;

  // Transitive closure only over signals.
  llvm::EquivalenceClasses<SourceRef> signalSets;
  // A simple set mapping of constants, as we do not want to compute a transitive closure over
  // constants.
  std::unordered_map<SourceRef, SourceRefSet, SourceRef::Hash> constantSets;

  // Maps references to the values where they are found.
  SourceRefLattice::Ref2Val ref2Val;

  // Also mutable for caching within otherwise const lookup operations.
  mutable mlir::SymbolTableCollection tables;

  /// @brief Constructs an empty CDG. The CDG is populated using computeConstraints.
  /// @param m The parent LLZK-compliant module.
  /// @param s The struct to analyze.
  /// @param intraprocedural Whether the CDG represents and intra-procedural or inter-procedural
  /// analysis
  ConstraintDependencyGraph(mlir::ModuleOp m, component::StructDefOp s, const CDGAnalysisContext &c)
      : mod(m), structDef(s), ctx(c) {}

  /// @brief Runs the constraint analysis to compute a transitive closure over SourceRefs
  /// as operated over by emit operations.
  /// @param solver The pre-configured solver.
  /// @param am The module-level AnalysisManager.
  /// @return mlir::success() if no issues were encountered, mlir::failure() otherwise
  mlir::LogicalResult computeConstraints(mlir::DataFlowSolver &solver, mlir::AnalysisManager &am);

  /// @brief Update the signalSets EquivalenceClasses based on the given
  /// emit operation. Relies on the caller to verify that `emitOp` is either
  /// an EmitEqualityOp or an EmitContainmentOp, as the logic for both is currently
  /// the same.
  /// @param solver The pre-configured solver.
  /// @param emitOp The emit operation that is creating a constraint.
  void walkConstrainOp(mlir::DataFlowSolver &solver, mlir::Operation *emitOp);
};

/// @brief An analysis wrapper around the ConstraintDependencyGraph for a given struct.
/// This analysis is a StructDefOp-level analysis that should not be directly
/// interacted with---rather, it is a utility used by the ConstraintDependencyGraphModuleAnalysis
/// that helps use MLIR's AnalysisManager to cache dependencies for sub-components.
class ConstraintDependencyGraphStructAnalysis
    : public StructAnalysis<ConstraintDependencyGraph, CDGAnalysisContext> {
public:
  using StructAnalysis::StructAnalysis;
  virtual ~ConstraintDependencyGraphStructAnalysis() = default;

  /// @brief Construct a CDG, using the module's analysis manager to query
  /// ConstraintDependencyGraph objects for nested components.
  mlir::LogicalResult runAnalysis(
      mlir::DataFlowSolver &solver, mlir::AnalysisManager &moduleAnalysisManager,
      const CDGAnalysisContext &ctx
  ) override;
};

/// @brief A module-level analysis for constructing ConstraintDependencyGraph objects for
/// all structs in the given LLZK module.
class ConstraintDependencyGraphModuleAnalysis
    : public ModuleAnalysis<
          ConstraintDependencyGraph, CDGAnalysisContext, ConstraintDependencyGraphStructAnalysis> {

public:
  using ModuleAnalysis::ModuleAnalysis;
  virtual ~ConstraintDependencyGraphModuleAnalysis() = default;

  void setIntraprocedural(bool runIntraprocedural) {
    ctx = {.runIntraprocedural = runIntraprocedural};
  }

protected:
  void initializeSolver() override { (void)solver.load<SourceRefAnalysis>(); }

  const CDGAnalysisContext &getContext() const override { return ctx; }

private:
  CDGAnalysisContext ctx = {.runIntraprocedural = false};
};

} // namespace llzk
