//===-- ForbiddenPreconditionInfluence.h ------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains an analysis and utilities for determining if a `verif`
/// precondition is dependent, via control-flow or data-flow, on forbidden sources
/// (i.e., struct members or function return values).
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Verif/IR/Ops.h"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Interfaces/CallInterfaces.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>

namespace llzk::verif {

/// Sources of information that are not allowed in contract preconditions. These
/// are generally results of the target, so having preconditions over them doesn't
/// make sense.
enum class ForbiddenPreconditionInfluence : uint8_t {
  None = 0,
  StructMember = 1 << 0,
  FunctionReturn = 1 << 1,
};

inline ForbiddenPreconditionInfluence
operator|(ForbiddenPreconditionInfluence lhs, ForbiddenPreconditionInfluence rhs) {
  return static_cast<ForbiddenPreconditionInfluence>(
      static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs)
  );
}

inline ForbiddenPreconditionInfluence &
operator|=(ForbiddenPreconditionInfluence &lhs, ForbiddenPreconditionInfluence rhs) {
  lhs = lhs | rhs;
  return lhs;
}

/// Return true when the influence set contains at least one classification.
inline bool any(ForbiddenPreconditionInfluence influence) {
  return static_cast<uint8_t>(influence) != 0;
}

/// Return true when `influence` contains the requested `flag`.
inline bool
hasInfluence(ForbiddenPreconditionInfluence influence, ForbiddenPreconditionInfluence flag) {
  return (static_cast<uint8_t>(influence) & static_cast<uint8_t>(flag)) != 0;
}

namespace detail {

using Influence = ForbiddenPreconditionInfluence;

/// Cache key for one interprocedural callable-result summary query.
///
/// The analyzer memoizes result summaries by callable body, the caller-provided
/// forbidden-influence classification of each argument, and the specific result
/// number being queried. Reusing this key avoids re-analyzing the same helper
/// function under identical argument influence assumptions.
struct CallableSummaryKey {
  mlir::Operation *callable {};
  llvm::SmallVector<uint8_t> argInfluences;
  unsigned resultNumber {};

  bool operator==(const CallableSummaryKey &other) const {
    return callable == other.callable && resultNumber == other.resultNumber &&
           argInfluences == other.argInfluences;
  }
};

struct CallableSummaryKeyInfo : llvm::DenseMapInfo<CallableSummaryKey> {
  static CallableSummaryKey getEmptyKey() {
    return {llvm::DenseMapInfo<mlir::Operation *>::getEmptyKey(), {}, 0};
  }

  static CallableSummaryKey getTombstoneKey() {
    return {llvm::DenseMapInfo<mlir::Operation *>::getTombstoneKey(), {}, 0};
  }

  static unsigned getHashValue(const CallableSummaryKey &key) {
    return llvm::hash_combine(
        key.callable, key.resultNumber,
        llvm::hash_combine_range(key.argInfluences.begin(), key.argInfluences.end())
    );
  }

  static bool isEqual(const CallableSummaryKey &lhs, const CallableSummaryKey &rhs) {
    return lhs == rhs;
  }
};

/// Interprocedural verifier-local analysis for forbidden precondition influence.
///
/// This analysis answers a narrow policy question for `verif.require_*`: whether
/// an SSA value may depend, through data flow or SCF control flow, on struct
/// members or target-function return values. It computes memoized callable
/// summaries so helper calls can be checked transitively without extending the
/// general-purpose `SourceRefAnalysis`.
class ForbiddenInfluenceAnalyzer {
public:
  /// Create a verifier-local analyzer for one LLZK module.
  explicit ForbiddenInfluenceAnalyzer(mlir::ModuleOp owningModule) : module(owningModule) {}

  /// Classify the forbidden influence reaching a value inside a contract body.
  Influence analyzeContractValue(verif::ContractOp contract, mlir::Value value);

  /// Summarize the forbidden influence of one callable result under the given
  /// argument influences.
  Influence analyzeCallableResult(
      mlir::CallableOpInterface callableOp, llvm::ArrayRef<Influence> argInfluences,
      unsigned resultNumber
  );

private:
  /// Callable-local recursive walker used while analyzing one contract or
  /// callable summary.
  ///
  /// An `AnalysisFrame` owns the value cache and recursion tracking for a single
  /// callable body under a fixed set of entry-argument influences. The parent
  /// `ForbiddenInfluenceAnalyzer` handles interprocedural summary caching, while
  /// this frame performs the intra-body traversal over SSA values, block
  /// arguments, calls, and SCF region results.
  class AnalysisFrame {
  public:
    /// Seed a callable-local analysis frame with the current argument influences.
    AnalysisFrame(
        ForbiddenInfluenceAnalyzer &parentAnalyzer, mlir::CallableOpInterface callableOp,
        llvm::ArrayRef<Influence> argInfluences
    );

    /// Recursively classify the forbidden influence reaching a single SSA value.
    Influence analyzeValue(mlir::Value value);

  private:
    /// Recover the forbidden influence reaching a region block argument.
    Influence analyzeBlockArgument(mlir::BlockArgument blockArg);

    /// Summarize the forbidden influence produced by a call result.
    Influence analyzeCallResult(mlir::CallOpInterface call, unsigned resultNumber);

    /// Summarize the forbidden influence produced by an `scf.if` result.
    Influence analyzeIfResult(mlir::scf::IfOp ifOp, unsigned resultNumber);

    /// Summarize the forbidden influence produced by an `scf.for` result.
    Influence analyzeForResult(mlir::scf::ForOp forOp, unsigned resultNumber);

    /// Summarize the forbidden influence produced by an `scf.while` result.
    Influence analyzeWhileResult(mlir::scf::WhileOp whileOp, unsigned resultNumber);

    ForbiddenInfluenceAnalyzer &analyzer;
    llvm::DenseMap<mlir::Value, Influence> valueCache;
    llvm::DenseSet<mlir::Value> activeValues;
  };

  /// Classify whether a contract entry argument is an allowed input or a
  /// forbidden target-function return value.
  static Influence classifyContractArgument(verif::ContractOp contract, mlir::BlockArgument arg);

  mlir::ModuleOp module;
  llvm::DenseMap<CallableSummaryKey, Influence, CallableSummaryKeyInfo> callableSummaryCache;
  llvm::DenseSet<CallableSummaryKey, CallableSummaryKeyInfo> activeSummaries;
};

} // namespace detail

/// Analyze whether a contract value depends on forbidden precondition sources.
inline ForbiddenPreconditionInfluence analyzeForbiddenPreconditionInfluence(
    mlir::ModuleOp module, verif::ContractOp contract, mlir::Value value
) {
  return detail::ForbiddenInfluenceAnalyzer(module).analyzeContractValue(contract, value);
}

/// Analyze whether a callable result depends on forbidden precondition sources
/// under a caller-provided argument influence summary.
inline ForbiddenPreconditionInfluence analyzeForbiddenPreconditionCallableResult(
    mlir::ModuleOp module, mlir::CallableOpInterface callableOp,
    llvm::ArrayRef<ForbiddenPreconditionInfluence> argInfluences, unsigned resultNumber
) {
  return detail::ForbiddenInfluenceAnalyzer(module).analyzeCallableResult(
      callableOp, argInfluences, resultNumber
  );
}

} // namespace llzk::verif
