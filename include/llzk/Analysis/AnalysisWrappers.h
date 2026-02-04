//===-- AnalysisWrappers.h --------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Convenience classes for a frequent pattern of dataflow analysis used in LLZK,
/// where an analysis is run across all `StructDefOp`s contained within a module,
/// where each of those analyses may need to reference the analysis results from
/// other `StructDefOp`s. This pattern reoccurs due to the instantiation of subcomponents
/// within components, which often requires the instantiating component to look up
/// the results of an analysis on the subcomponent. This kind of lookup is not
/// supported through mlir's AnalysisManager, as it only allows lookups on nested operations,
/// not sibling operations. This pattern allows subcomponents to instead use the ModuleOp's
/// analysis manager, allowing components to query analyses for any component in the module.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Analysis/AnalysisUtil.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/Compare.h"
#include "llzk/Util/ErrorHelper.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/AnalysisManager.h>

#include <map>

namespace llzk {

/// @brief This is the base class for a dataflow analysis designed to run on a single struct (i.e.,
/// a single component).
/// @tparam Result The output of the analysis.
/// @tparam Context Any module-level information or configuration needed to run this analysis.
template <typename Result, typename Context> class StructAnalysis {
public:
  /// @brief Assert that this analysis is being run on a StructDefOp and initializes the
  /// analysis with the current StructDefOp and its parent ModuleOp.
  /// @param op The presumed StructDefOp.
  StructAnalysis(mlir::Operation *op) {
    structDefOp = llvm::dyn_cast<component::StructDefOp>(op);
    if (!structDefOp) {
      const char *error_message = "StructAnalysis expects provided op to be a StructDefOp!";
      op->emitError(error_message).report();
      llvm::report_fatal_error(error_message);
    }
    auto maybeModOp = getRootModule(op);
    if (mlir::failed(maybeModOp)) {
      const char *error_message = "StructAnalysis could not find root module from StructDefOp!";
      op->emitError(error_message).report();
      llvm::report_fatal_error(error_message);
    }
    modOp = *maybeModOp;
  }
  virtual ~StructAnalysis() = default;

  /// @brief Perform the analysis and construct the `Result` output.
  /// @param solver The pre-configured dataflow solver. This solver should already have
  /// a liveness analysis run, otherwise this analysis may be a no-op.
  /// @param moduleAnalysisManager The analysis manager of the top-level module. By giving
  /// the struct analysis a reference to the module's analysis manager, we can query analyses of
  /// other structs by querying for a child analysis. Otherwise, a struct's analysis manager cannot
  /// query for the analyses of other operations unless they are nested within the struct.
  /// @param ctx The `Context` given to the analysis. This is presumed to have been created by the
  /// StructAnalysis's parent ModuleAnalysis.
  /// @return `mlir::success()` if the analysis ran without errors, and a `mlir::failure()`
  /// otherwise.
  virtual mlir::LogicalResult runAnalysis(
      mlir::DataFlowSolver &solver, mlir::AnalysisManager &moduleAnalysisManager, const Context &ctx
  ) = 0;

  /// @brief Query if the analysis has constructed a `Result` object for the given `Context`.
  bool constructed(const Context &ctx) const { return res.contains(ctx); }

  /// @brief Access the result iff it has been created for the given `Context` object `ctx`.
  const Result &getResult(const Context &ctx) const {
    ensure(
        constructed(ctx), mlir::Twine(__FUNCTION__) +
                              ": result has not been constructed for struct " +
                              mlir::Twine(getStruct().getName())
    );
    return *res.at(ctx);
  }

protected:
  /// @brief Get the `ModuleOp` that is the parent of the `StructDefOp` that is under analysis.
  mlir::ModuleOp getModule() const { return modOp; }

  /// @brief Get the current `StructDefOp` that is under analysis.
  component::StructDefOp getStruct() const { return structDefOp; }

  /// @brief Initialize the final `Result` object.
  void setResult(const Context &ctx, Result &&r) {
    auto [_, inserted] = res.insert(std::make_pair(ctx, std::make_unique<Result>(r)));
    ensure(inserted, "Result already initialized");
  }

private:
  mlir::ModuleOp modOp;
  component::StructDefOp structDefOp;
  std::unordered_map<Context, std::unique_ptr<Result>> res;
};

template <typename Context>
concept ContextType = requires(const Context &a, const Context &b) {
  { a == b } -> std::convertible_to<bool>;
  { std::hash<Context> {}(a) } -> std::convertible_to<std::size_t>;
};

/// @brief An empty struct that is used for convenience for analyses that do not
/// require any context.
struct NoContext {};

/// @brief Any type that is a subclass of `StructAnalysis` and provided a
/// `Context` that matches `ContextType`.
template <typename Analysis, typename Result, typename Context>
concept StructAnalysisType = requires {
  requires std::is_base_of<StructAnalysis<Result, Context>, Analysis>::value;
  requires ContextType<Context>;
};

/// @brief An analysis wrapper that runs the given `StructAnalysisTy` struct analysis over
/// all of the struct contained within the module. Through the use of the `Context` object, this
/// analysis facilitates the sharing of common data and analyses across struct analyses.
/// @tparam Result The result of each `StructAnalysis`.
/// @tparam Context The context shared between `StructAnalysis` analyses.
/// @tparam StructAnalysisType The analysis run on all the contained module's structs.
template <typename Result, typename Context, StructAnalysisType<Result, Context> StructAnalysisTy>
class ModuleAnalysis {

  /// @brief Per-struct results mapping.
  using StructResults = std::map<
      component::StructDefOp, std::reference_wrapper<const Result>,
      NamedOpLocationLess<component::StructDefOp>>;

  /// @brief A map of this module's structs to the result of the `StructAnalysis`
  /// on that struct for a given configuration (denoted by the `Context` object).
  /// The inner `StructResults` is implemented as an ordered map to control
  /// sorting order for iteration.
  using ResultMap = std::unordered_map<Context, StructResults>;

public:
  /// @brief Asserts that the analysis is being run on a `ModuleOp`.
  /// @note Derived classes may also use the `Analysis(mlir::Operation*, mlir::AnalysisManager&)`
  /// constructor that is allowed by classes that are constructed using the
  /// `AnalysisManager::getAnalysis<Analysis>()` method.
  ModuleAnalysis(mlir::Operation *op) {
    if (modOp = llvm::dyn_cast<mlir::ModuleOp>(op); !modOp) {
      auto error_message = "ModuleAnalysis expects provided op to be an mlir::ModuleOp!";
      op->emitError(error_message).report();
      llvm::report_fatal_error(error_message);
    }
  }
  virtual ~ModuleAnalysis() = default;

  /// @brief Run the `StructAnalysisTy` struct analysis on all child structs.
  /// @param am The module-level analysis manager that will be passed to
  /// `StructAnalysis::runAnalysis`. This analysis manager should be the same analysis manager used
  /// to construct this analysis.
  virtual void runAnalysis(mlir::AnalysisManager &am) { constructChildAnalyses(am); }

  /// @brief Runs the analysis if the results do not already exist.
  void ensureAnalysisRun(mlir::AnalysisManager &am) {
    if (!constructed()) {
      runAnalysis(am);
    }
  }

  /// @brief Check if the results of this analysis have been created for the currently
  /// available context.
  bool constructed() const { return results.contains(getContext()); }

  /// @brief Checks if `op` has a result contained in the current result map.
  bool hasResult(component::StructDefOp op) const {
    return constructed() && results.at(getContext()).contains(op);
  }

  /// @brief Asserts that `op` has a result and returns it.
  const Result &getResult(component::StructDefOp op) const {
    ensureResultCreated(op);
    return results.at(getContext()).at(op).get();
  }

  /// @brief Get the results for the current context.
  const StructResults &getCurrentResults() const {
    ensure(constructed(), "results are not yet constructed for the current context");
    return results.at(getContext());
  }

  mlir::DataFlowSolver &getSolver() { return solver; }

protected:
  mlir::DataFlowSolver solver;

  /// @brief Initialize the shared dataflow solver with any common analyses required
  /// by the contained struct analyses.
  /// @param solver
  virtual void initializeSolver() = 0;

  /// @brief Return the current `Context` object. The context contains parameters
  /// that configure or pass information to the analysis.
  virtual const Context &getContext() const = 0;

  /// @brief Construct and run the `StructAnalysisTy` analyses on each `StructDefOp` contained
  /// in the `ModuleOp` that is being subjected to this analysis.
  /// @param am The module's analysis manager.
  void constructChildAnalyses(mlir::AnalysisManager &am) {
    dataflow::markAllOpsAsLive(solver, modOp);

    // The analysis is run at the module level so that lattices are computed
    // for global functions as well.
    initializeSolver();
    auto res = solver.initializeAndRun(modOp);
    ensure(res.succeeded(), "solver failed to run on module!");

    const Context &ctx = getContext();
    // Force construction of empty results here so `getCurrentResults()` on
    // a module with no inner structs returns no results rather than an assertion
    // failure.
    results[ctx] = {};
    modOp.walk([this, &am, &ctx](component::StructDefOp s) mutable {
      auto &childAnalysis = am.getChildAnalysis<StructAnalysisTy>(s);
      // Don't re-run the analysis if we already have the results.
      // The analysis may have been run as part of a nested analysis.
      if (!childAnalysis.constructed(ctx)) {
        mlir::LogicalResult childAnalysisRes = childAnalysis.runAnalysis(solver, am, ctx);

        if (mlir::failed(childAnalysisRes)) {
          auto error_message = "StructAnalysis failed to run for " + mlir::Twine(s.getName());
          s->emitError(error_message).report();
          llvm::report_fatal_error(error_message);
        }
      }

      auto [_, inserted] = results[ctx].insert(
          std::make_pair(s, std::reference_wrapper(childAnalysis.getResult(ctx)))
      );
      ensure(inserted, "struct location conflict");
      return mlir::WalkResult::skip();
    });
  }

private:
  mlir::ModuleOp modOp;
  ResultMap results;

  /// @brief Ensures that the given struct has a result.
  /// @param op The struct to ensure has a result.
  void ensureResultCreated(component::StructDefOp op) const {
    ensure(hasResult(op), "Result does not exist for StructDefOp " + mlir::Twine(op.getName()));
  }
};

} // namespace llzk
