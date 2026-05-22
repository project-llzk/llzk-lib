//===-- IntervalAnalysis.h --------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Analysis/AbstractLatticeValue.h"
#include "llzk/Analysis/AnalysisWrappers.h"
#include "llzk/Analysis/ConstraintDependencyGraph.h"
#include "llzk/Analysis/Intervals.h"
#include "llzk/Analysis/SparseAnalysis.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Util/Compare.h"
#include "llzk/Util/Field.h"

#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/AnalysisManager.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/ScopeExit.h>
#include <llvm/Support/SMTAPI.h>

#include <array>
#include <mutex>
#include <optional>
#include <unordered_set>

namespace llzk {

/* ExpressionValue */

/// @brief Tracks a solver expression and an interval range for that expression.
/// Used as a scalar lattice value.
class ExpressionValue {
public:
  /* Must be default initializable to be a ScalarLatticeValue. */
  ExpressionValue() : i(), expr(nullptr), unreduced(std::nullopt) {}

  explicit ExpressionValue(const Field &f)
      : i(Interval::Entire(f)), expr(nullptr), unreduced(std::nullopt) {}

  ExpressionValue(const Field &f, llvm::SMTExprRef exprRef)
      : i(Interval::Entire(f)), expr(exprRef), unreduced(std::nullopt) {}

  ExpressionValue(const Field &f, llvm::SMTExprRef exprRef, const llvm::DynamicAPInt &singleVal)
      : i(Interval::Degenerate(f, singleVal)), expr(exprRef), unreduced(std::nullopt) {}

  ExpressionValue(
      llvm::SMTExprRef exprRef, const Interval &interval,
      std::optional<UnreducedInterval> unreducedInterval = std::nullopt
  )
      : i(interval), expr(exprRef), unreduced(std::move(unreducedInterval)) {}

  llvm::SMTExprRef getExpr() const { return expr; }

  const Interval &getInterval() const { return i; }

  bool hasUnreducedInterval() const { return unreduced.has_value(); }

  const std::optional<UnreducedInterval> &getOptionalUnreducedInterval() const { return unreduced; }

  const UnreducedInterval &getUnreducedInterval() const {
    ensure(unreduced.has_value(), "unreduced interval not set");
    return *unreduced;
  }

  const Field &getField() const { return i.getField(); }

  /// @brief Return the current expression with a new interval.
  /// @param newInterval
  /// @return
  ExpressionValue withInterval(const Interval &newInterval) const {
    return ExpressionValue(expr, newInterval, unreduced);
  }

  /// @brief Return the current expression with a new SMT expression.
  ExpressionValue withExpression(const llvm::SMTExprRef &newExpr) const {
    return ExpressionValue(newExpr, i, unreduced);
  }

  ExpressionValue withUnreducedInterval(const UnreducedInterval &newUnreducedInterval) const {
    return ExpressionValue(expr, i, newUnreducedInterval);
  }

  ExpressionValue
  withOptionalUnreducedInterval(std::optional<UnreducedInterval> newUnreducedInterval) const {
    return ExpressionValue(expr, i, std::move(newUnreducedInterval));
  }

  ExpressionValue dropUnreducedInterval() const { return ExpressionValue(expr, i, std::nullopt); }

  /* Required to be a ScalarLatticeValue. */
  /// @brief Fold two expressions together when overapproximating array elements.
  ExpressionValue &join(const ExpressionValue & /*rhs*/) {
    i = Interval::Entire(getField());
    unreduced = std::nullopt;
    return *this;
  }

  bool operator==(const ExpressionValue &rhs) const;

  bool isBoolSort(const llvm::SMTSolverRef &solver) const {
    return solver->getBoolSort() == solver->getSort(expr);
  }

  /// @brief Compute the intersection of the lhs and rhs intervals, and create a solver
  /// expression that constrains both sides to be equal.
  /// @param solver
  /// @param lhs
  /// @param rhs
  /// @return
  friend ExpressionValue intersection(
      const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs
  );

  /// @brief Compute the union of the lhs and rhs intervals, and create a solver
  /// expression that constrains both sides to be equal.
  /// @param solver
  /// @param lhs
  /// @param rhs
  /// @return
  friend ExpressionValue
  join(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs);

  // arithmetic ops

  friend ExpressionValue
  add(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs);

  friend ExpressionValue
  sub(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs);

  friend ExpressionValue
  mul(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs);

  friend ExpressionValue
  div(const llvm::SMTSolverRef &solver, mlir::Operation *op, const ExpressionValue &lhs,
      const ExpressionValue &rhs);

  friend ExpressionValue uintDiv(
      const llvm::SMTSolverRef &solver, mlir::Operation *op, const ExpressionValue &lhs,
      const ExpressionValue &rhs
  );

  friend ExpressionValue sintDiv(
      const llvm::SMTSolverRef &solver, mlir::Operation *op, const ExpressionValue &lhs,
      const ExpressionValue &rhs
  );

  friend ExpressionValue
  mod(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs);

  friend ExpressionValue
  bitAnd(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs);

  friend ExpressionValue
  bitOr(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs);

  friend ExpressionValue
  bitXor(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs);

  friend ExpressionValue shiftLeft(
      const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs
  );

  friend ExpressionValue shiftRight(
      const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs
  );

  friend ExpressionValue
  cmp(const llvm::SMTSolverRef &solver, boolean::CmpOp op, const ExpressionValue &lhs,
      const ExpressionValue &rhs);

  friend ExpressionValue
  boolAnd(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs);

  friend ExpressionValue
  boolOr(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs);

  friend ExpressionValue
  boolXor(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs);

  friend ExpressionValue neg(const llvm::SMTSolverRef &solver, const ExpressionValue &val);

  friend ExpressionValue notOp(const llvm::SMTSolverRef &solver, const ExpressionValue &val);

  friend ExpressionValue boolNot(const llvm::SMTSolverRef &solver, const ExpressionValue &val);

  friend ExpressionValue fallbackUnaryOp(
      const llvm::SMTSolverRef &solver, mlir::Operation *op, const ExpressionValue &val
  );

  /* Utility */

  void print(mlir::raw_ostream &os) const;

  friend mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const ExpressionValue &e) {
    e.print(os);
    return os;
  }

  struct Hash {
    unsigned operator()(const ExpressionValue &e) const {
      return Interval::Hash {}(e.i) ^ llvm::hash_value(e.expr) ^
             std::hash<bool> {}(e.unreduced.has_value()) ^
             (e.unreduced.has_value() ? UnreducedInterval::Hash {}(*e.unreduced) : 0U);
    }
  };

private:
  Interval i;
  llvm::SMTExprRef expr;
  std::optional<UnreducedInterval> unreduced;
};

/* IntervalAnalysisLatticeValue */

// NOLINTNEXTLINE(bugprone-exception-escape)
class IntervalAnalysisLatticeValue
    : public dataflow::AbstractLatticeValue<IntervalAnalysisLatticeValue, ExpressionValue> {
  using Base = dataflow::AbstractLatticeValue<IntervalAnalysisLatticeValue, ExpressionValue>;

public:
  explicit IntervalAnalysisLatticeValue(ExpressionValue e) : Base(std::move(e)) {}
  IntervalAnalysisLatticeValue() : Base() {}
  explicit IntervalAnalysisLatticeValue(mlir::ArrayRef<int64_t> shape) : Base(shape) {}
  IntervalAnalysisLatticeValue(const IntervalAnalysisLatticeValue &) = default;
  IntervalAnalysisLatticeValue(IntervalAnalysisLatticeValue &&) = default;
  IntervalAnalysisLatticeValue &operator=(const IntervalAnalysisLatticeValue &) = default;
  IntervalAnalysisLatticeValue &operator=(IntervalAnalysisLatticeValue &&) = default;
};

/* IntervalAnalysisLattice */

class IntervalDataFlowAnalysis;

class IntervalAnalysisLattice : public dataflow::AbstractSparseLattice {
public:
  using LatticeValue = IntervalAnalysisLatticeValue;
  // Map mlir::Values to LatticeValues
  using ValueMap = mlir::DenseMap<mlir::Value, LatticeValue>;
  // Map member references to LatticeValues. Used for member reads and writes.
  // Structure is component value -> member attribute -> latticeValue
  using MemberMap = mlir::DenseMap<mlir::Value, mlir::DenseMap<mlir::StringAttr, LatticeValue>>;
  // Expression to interval map for convenience.
  using ExpressionIntervals = mlir::DenseMap<llvm::SMTExprRef, Interval>;
  // Tracks all constraints and assignments in insertion order
  using ConstraintSet = llvm::SetVector<ExpressionValue>;

  using AbstractSparseLattice::AbstractSparseLattice;

  mlir::ChangeResult join(const AbstractSparseLattice &other) override;

  mlir::ChangeResult meet(const AbstractSparseLattice &other) override;

  void print(mlir::raw_ostream &os) const override;

  const LatticeValue &getValue() const { return val; }

  mlir::ChangeResult setValue(const LatticeValue &val);
  mlir::ChangeResult setValue(const ExpressionValue &e);

  mlir::ChangeResult addSolverConstraint(const ExpressionValue &e);

  friend mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const IntervalAnalysisLattice &l) {
    l.print(os);
    return os;
  }

  const ConstraintSet &getConstraints() const { return constraints; }

  mlir::FailureOr<Interval> findInterval(llvm::SMTExprRef expr) const;
  mlir::ChangeResult setInterval(llvm::SMTExprRef expr, const Interval &i);

private:
  LatticeValue val;
  ConstraintSet constraints;
};

/* IntervalDataFlowAnalysis */

class IntervalDataFlowAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<IntervalAnalysisLattice> {
  using Base = dataflow::SparseForwardDataFlowAnalysis<IntervalAnalysisLattice>;
  using Lattice = IntervalAnalysisLattice;
  using LatticeValue = IntervalAnalysisLattice::LatticeValue;

  // Map SourceRefs to their symbols.
  using SymbolMap = mlir::DenseMap<SourceRef, llvm::SMTExprRef>;

public:
  explicit IntervalDataFlowAnalysis(
      mlir::DataFlowSolver &dataflowSolver, llvm::SMTSolverRef smt, const Field &f,
      bool propInputConstraints, bool shouldTrackUnreducedIntervals
  )
      : Base::SparseForwardDataFlowAnalysis(dataflowSolver), _dataflowSolver(dataflowSolver),
        smtSolver(std::move(smt)), field(f), propagateInputConstraints(propInputConstraints),
        trackUnreducedIntervals(shouldTrackUnreducedIntervals) {}

  mlir::LogicalResult visitOperation(
      mlir::Operation *op, mlir::ArrayRef<const Lattice *> operands,
      mlir::ArrayRef<Lattice *> results
  ) override;

  /// @brief Either return the existing SMT expression that corresponds to the SourceRef,
  /// or create one.
  /// @param r
  /// @return
  llvm::SMTExprRef getOrCreateSymbol(const SourceRef &r);

  const llvm::DenseMap<SourceRef, llvm::DenseSet<Lattice *>> &getReadResults() const {
    return readResults;
  }

  const llvm::DenseMap<SourceRef, ExpressionValue> &getWriteResults() const { return writeResults; }

private:
  mlir::DataFlowSolver &_dataflowSolver;
  llvm::SMTSolverRef smtSolver;
  SymbolMap refSymbols;
  std::reference_wrapper<const Field> field;
  bool propagateInputConstraints;
  bool trackUnreducedIntervals;
  mlir::SymbolTableCollection tables;

  // Track SourceRef-indexed reads so writes to rooted storage can update existing readers.
  llvm::DenseMap<SourceRef, llvm::DenseSet<Lattice *>> readResults;
  // Track SourceRef-indexed writes. For now, we'll overapproximate repeated writes.
  llvm::DenseMap<SourceRef, ExpressionValue> writeResults;

  void setToEntryState(Lattice *lattice) override {
    // Initialize the value with an interval in our specified field.
    (void)lattice->setValue(ExpressionValue(field.get()));
  }

  static bool isBooleanType(mlir::Type ty) {
    if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(ty)) {
      return intTy.getWidth() == 1;
    }
    return false;
  }

  Interval getDefaultIntervalForType(mlir::Type ty) const {
    return isBooleanType(ty) ? Interval::Boolean(field.get()) : Interval::Entire(field.get());
  }

  std::optional<UnreducedInterval> getDefaultUnreducedIntervalForType(mlir::Type ty) const;

  std::optional<UnreducedInterval> getRefUnreducedInterval(const SourceRef &ref);

  llvm::SMTExprRef createSymbol(mlir::Type ty, const char *name) const;

  llvm::SMTExprRef createSymbol(const SourceRef &r) const;

  llvm::SMTExprRef createSymbol(mlir::Value val) const;

  ExpressionValue createUnknownValue(mlir::Value val) const {
    return ExpressionValue(
        createSymbol(val), getDefaultIntervalForType(val.getType()),
        getDefaultUnreducedIntervalForType(val.getType())
    );
  }

  inline bool isConstOp(mlir::Operation *op) const {
    return llvm::isa<
        felt::FeltConstantOp, mlir::arith::ConstantIndexOp, mlir::arith::ConstantIntOp>(op);
  }

  inline bool isBoolConstOp(mlir::Operation *op) const {
    if (auto constIntOp = llvm::dyn_cast<mlir::arith::ConstantIntOp>(op)) {
      auto valAttr = dyn_cast<mlir::IntegerAttr>(constIntOp.getValue());
      ensure(valAttr != nullptr, "arith::ConstantIntOp must have an IntegerAttr as its value");
      return valAttr.getValue().getBitWidth() == 1;
    }
    return false;
  }

  llvm::DynamicAPInt getConst(mlir::Operation *op) const;

  inline llvm::SMTExprRef createConstBitvectorExpr(const llvm::DynamicAPInt &v) const {
    return createConstBitvectorExpr(toAPSInt(v));
  }

  inline llvm::SMTExprRef createConstBitvectorExpr(const llvm::APSInt &v) const {
    return smtSolver->mkBitvector(v, field.get().bitWidth());
  }

  llvm::SMTExprRef createConstBoolExpr(bool v) const { return smtSolver->mkBoolean(v); }

  bool isArithmeticOp(mlir::Operation *op) const {
    return llvm::isa<
        felt::AddFeltOp, felt::SubFeltOp, felt::MulFeltOp, felt::DivFeltOp, felt::UnsignedModFeltOp,
        felt::SignedModFeltOp, felt::SignedIntDivFeltOp, felt::UnsignedIntDivFeltOp,
        mlir::arith::XOrIOp, felt::NegFeltOp, felt::InvFeltOp, felt::AndFeltOp, felt::OrFeltOp,
        felt::XorFeltOp, felt::NotFeltOp, felt::ShlFeltOp, felt::ShrFeltOp, boolean::CmpOp,
        boolean::AndBoolOp, boolean::OrBoolOp, boolean::XorBoolOp, boolean::NotBoolOp>(op);
  }

  ExpressionValue
  performBinaryArithmetic(mlir::Operation *op, const LatticeValue &a, const LatticeValue &b);

  ExpressionValue performUnaryArithmetic(mlir::Operation *op, const LatticeValue &a);

  /// @brief Recursively applies the new interval to the val's lattice value and to that value's
  /// operands, if possible. For example, if we know that X*Y is non-zero, then we know X and Y are
  /// non-zero, and can update X and Y's intervals accordingly.
  /// @param after The current lattice state. Assumes that this has already been joined with the
  /// `before` lattice in `visitOperation`, so lookups and updates can be performed on the `after`
  /// lattice alone.
  void applyInterval(mlir::Operation *originalOp, mlir::Value val, Interval newInterval);

  /// @brief Special handling for generalized (s - c0) * (s - c1) * ... * (s - cN) = 0 patterns.
  mlir::FailureOr<std::pair<llvm::DenseSet<mlir::Value>, Interval>>
  getGeneralizedDecompInterval(mlir::Operation *baseOp, mlir::Value lhs, mlir::Value rhs);

  bool isReadOp(mlir::Operation *op) const {
    return llvm::isa<component::MemberReadOp, polymorphic::ConstReadOp, array::ReadArrayOp>(op);
  }

  bool isDefinitionOp(mlir::Operation *op) const {
    return llvm::isa<
        component::StructDefOp, function::FuncDefOp, component::MemberDefOp, global::GlobalDefOp,
        mlir::ModuleOp>(op);
  }

  bool isReturnOp(mlir::Operation *op) const { return llvm::isa<function::ReturnOp>(op); }

  /// @brief Convert an array access op's indices into SourceRef path components.
  /// Constant indices are tracked precisely, while dynamic indices are widened to
  /// the full valid range for that array dimension.
  std::vector<SourceRefIndex>
  getArrayAccessIndices(mlir::Operation *baseOp, array::ArrayAccessOpInterface arrayAccessOp);

  /// @brief Build the SourceRef addressed by an array access op when its base is a
  /// block argument or rooted SSA value.
  mlir::FailureOr<SourceRef>
  getArrayAccessRef(mlir::Operation *baseOp, array::ArrayAccessOpInterface arrayAccessOp);

  /// @brief Compute the best known interval for a SourceRef from writes, constants,
  /// or an already-initialized root lattice value.
  Interval getRefInterval(const SourceRef &ref);

  /// @brief Return the best known ExpressionValue for a SourceRef, reusing an exact
  /// written value when available and otherwise pairing a fresh SSA symbol with the
  /// SourceRef's current interval.
  ExpressionValue getRefValue(const SourceRef &ref, mlir::Value val);

  /// @brief Record a write to the given SourceRef and eagerly refine any reads that
  /// are currently tracking the same storage location.
  void recordRefWrite(const SourceRef &writtenRef, const ExpressionValue &writeVal);

  /// @brief Get the SourceRef state that defines `val`.
  SourceRefLatticeValue getSourceRefState(mlir::Value val);
};

/* StructIntervals */

/// @brief Parameters and shared objects to pass to child analyses.
struct IntervalAnalysisContext {
  IntervalDataFlowAnalysis *intervalDFA;
  llvm::SMTSolverRef smtSolver;
  std::optional<std::reference_wrapper<const Field>> field;
  bool propagateInputConstraints = false;
  bool trackUnreducedIntervals = false;

  llvm::SMTExprRef getSymbol(const SourceRef &r) const { return intervalDFA->getOrCreateSymbol(r); }
  bool hasField() const { return field.has_value(); }
  const Field &getField() const {
    ensure(field.has_value(), "field not set within context");
    return field->get();
  }
  bool doInputConstraintPropagation() const { return propagateInputConstraints; }
  bool doTrackUnreducedIntervals() const { return trackUnreducedIntervals; }

  friend bool
  operator==(const IntervalAnalysisContext &a, const IntervalAnalysisContext &b) = default;
};

} // namespace llzk

template <> struct std::hash<llzk::IntervalAnalysisContext> {
  size_t operator()(const llzk::IntervalAnalysisContext &c) const {
    return llvm::hash_combine(
        std::hash<const llzk::IntervalDataFlowAnalysis *> {}(c.intervalDFA),
        std::hash<const llvm::SMTSolver *> {}(c.smtSolver.get()),
        std::hash<const llzk::Field *> {}(&c.getField()),
        std::hash<bool> {}(c.propagateInputConstraints),
        std::hash<bool> {}(c.trackUnreducedIntervals)
    );
  }
};

namespace llzk {

// Suppress false positive from `clang-tidy`
// NOLINTNEXTLINE(bugprone-exception-escape)
class StructIntervals {
public:
  /// @brief Compute the struct intervals.
  /// @param mod The LLZK-complaint module that is the parent of struct `s`.
  /// @param s The struct to compute value intervals for.
  /// @param solver A pre-configured DataFlowSolver. The liveness of the struct must
  /// already be computed in this solver in order for the analysis to run.
  /// @param am A module-level analysis manager. This analysis manager needs to originate
  /// from a module-level analysis (i.e., for the `mod` module) so that analyses
  /// for other constraints can be queried via the getChildAnalysis method.
  /// @return
  static mlir::FailureOr<StructIntervals> compute(
      mlir::ModuleOp mod, component::StructDefOp s, mlir::DataFlowSolver &solver,
      mlir::AnalysisManager &am, const IntervalAnalysisContext &ctx
  ) {
    StructIntervals si(mod, s);
    if (si.computeIntervals(solver, am, ctx).failed()) {
      return mlir::failure();
    }
    return si;
  }

  mlir::LogicalResult computeIntervals(
      mlir::DataFlowSolver &solver, mlir::AnalysisManager &am, const IntervalAnalysisContext &ctx
  );

  void print(
      mlir::raw_ostream &os, bool withConstraints = false, bool printCompute = false,
      bool printUnreduced = false
  ) const;

  const llvm::MapVector<SourceRef, Interval> &getConstrainIntervals() const {
    return constrainMemberRanges;
  }

  const llvm::MapVector<SourceRef, UnreducedInterval> &getConstrainUnreducedIntervals() const {
    return constrainMemberUnreducedRanges;
  }

  const llvm::SetVector<ExpressionValue> getConstrainSolverConstraints() const {
    return constrainSolverConstraints;
  }

  const llvm::MapVector<SourceRef, Interval> &getComputeIntervals() const {
    return computeMemberRanges;
  }

  const llvm::MapVector<SourceRef, UnreducedInterval> &getComputeUnreducedIntervals() const {
    return computeMemberUnreducedRanges;
  }

  const llvm::SetVector<ExpressionValue> getComputeSolverConstraints() const {
    return computeSolverConstraints;
  }

  friend mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const StructIntervals &si) {
    si.print(os);
    return os;
  }

private:
  mlir::ModuleOp mod;
  component::StructDefOp structDef;
  llvm::SMTSolverRef smtSolver;
  // llvm::MapVector keeps insertion order for consistent iteration
  llvm::MapVector<SourceRef, Interval> constrainMemberRanges, computeMemberRanges;
  llvm::MapVector<SourceRef, UnreducedInterval> constrainMemberUnreducedRanges,
      computeMemberUnreducedRanges;
  // llvm::SetVector for the same reasons as above
  llvm::SetVector<ExpressionValue> constrainSolverConstraints, computeSolverConstraints;

  StructIntervals(mlir::ModuleOp m, component::StructDefOp s) : mod(m), structDef(s) {}
};

/* StructIntervalAnalysis */

class ModuleIntervalAnalysis;

class StructIntervalAnalysis : public StructAnalysis<StructIntervals, IntervalAnalysisContext> {
public:
  using StructAnalysis::StructAnalysis;
  ~StructIntervalAnalysis() override = default;

  bool inProgress(const IntervalAnalysisContext &ctx) const {
    return inProgressContexts.contains(ctx);
  }

  mlir::LogicalResult runAnalysis(
      mlir::DataFlowSolver &solver, mlir::AnalysisManager &am, const IntervalAnalysisContext &ctx
  ) override {
    if (inProgress(ctx)) {
      return mlir::failure();
    }
    inProgressContexts.insert(ctx);
    auto cleanup = llvm::make_scope_exit([this, &ctx] { inProgressContexts.erase(ctx); });

    auto computeRes = StructIntervals::compute(getModule(), getStruct(), solver, am, ctx);
    if (mlir::failed(computeRes)) {
      return mlir::failure();
    }
    setResult(ctx, std::move(*computeRes));
    return mlir::success();
  }

private:
  std::unordered_set<IntervalAnalysisContext> inProgressContexts;
};

/* ModuleIntervalAnalysis */

class ModuleIntervalAnalysis
    : public ModuleAnalysis<StructIntervals, IntervalAnalysisContext, StructIntervalAnalysis> {

public:
  // We set intraprocedural to false for the sake of the SourceRefAnalysis
  ModuleIntervalAnalysis(mlir::Operation *op)
      : ModuleAnalysis(op, mlir::DataFlowConfig().setInterprocedural(false)), ctx {} {
    ctx.smtSolver = llvm::CreateZ3Solver();
  }
  ~ModuleIntervalAnalysis() override = default;

  void setField(const Field &f) { ctx.field = f; }
  void setPropagateInputConstraints(bool prop) { ctx.propagateInputConstraints = prop; }
  void setTrackUnreducedIntervals(bool track) { ctx.trackUnreducedIntervals = track; }

protected:
  void initializeSolver() override {
    ensure(ctx.hasField(), "field not set, could not generate analysis context");
    (void)solver.load<SourceRefAnalysis>();
    auto smtSolverRef = ctx.smtSolver;
    bool prop = ctx.propagateInputConstraints;
    bool track = ctx.trackUnreducedIntervals;
    ctx.intervalDFA =
        solver.load<IntervalDataFlowAnalysis, llvm::SMTSolverRef, const Field &, bool, bool>(
            std::move(smtSolverRef), ctx.getField(),
            std::move(prop), // NOLINT(performance-move-const-arg)
            std::move(track) // NOLINT(performance-move-const-arg)
        );
  }

  const IntervalAnalysisContext &getContext() const override {
    ensure(ctx.field.has_value(), "field not set, could not generate analysis context");
    return ctx;
  }

private:
  IntervalAnalysisContext ctx;
};

} // namespace llzk

namespace llvm {

template <> struct DenseMapInfo<llzk::ExpressionValue> {

  static SMTExprRef getEmptyExpr() {
    static const auto *emptyPtr = reinterpret_cast<SMTExprRef>(1);
    return emptyPtr;
  }
  static SMTExprRef getTombstoneExpr() {
    static const auto *tombstonePtr = reinterpret_cast<SMTExprRef>(2);
    return tombstonePtr;
  }

  static llzk::ExpressionValue getEmptyKey() {
    return llzk::ExpressionValue(llzk::Field::getField("bn128"), getEmptyExpr());
  }
  static inline llzk::ExpressionValue getTombstoneKey() {
    return llzk::ExpressionValue(llzk::Field::getField("bn128"), getTombstoneExpr());
  }
  static unsigned getHashValue(const llzk::ExpressionValue &e) {
    return llzk::ExpressionValue::Hash {}(e);
  }
  static bool isEqual(const llzk::ExpressionValue &lhs, const llzk::ExpressionValue &rhs) {
    if (lhs.getExpr() == getEmptyExpr() || lhs.getExpr() == getTombstoneExpr() ||
        rhs.getExpr() == getEmptyExpr() || rhs.getExpr() == getTombstoneExpr()) {
      return lhs.getExpr() == rhs.getExpr();
    }
    return lhs == rhs;
  }
};

} // namespace llvm
