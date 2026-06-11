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

#pragma once

#include "llzk/Dialect/Verif/IR/Ops.h"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Interfaces/CallInterfaces.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallVector.h>

#include <optional>

namespace llzk::verif {

/// Sources of information that are not allowed in contract preconditions. These
/// are generally results of the target, so having preconditions over them doesn't
/// make sense.
enum class ForbiddenPreconditionInfluence : uint8_t {
  None = 0,
  StructMember = 1 << 0,
  FunctionReturn = 1 << 1,
};

/// Summary of forbidden precondition influence along with representative source
/// locations for each forbidden kind.
struct ForbiddenPreconditionInfluenceInfo {
  ForbiddenPreconditionInfluence influence = ForbiddenPreconditionInfluence::None;
  llvm::SmallSetVector<mlir::Location, 2> structMemberLocs = {};

  bool operator==(const ForbiddenPreconditionInfluenceInfo &other) const {
    return influence == other.influence && structMemberLocs == other.structMemberLocs;
  }

  static ForbiddenPreconditionInfluenceInfo None() { return {}; }

  static ForbiddenPreconditionInfluenceInfo StructMember() {
    return {.influence = ForbiddenPreconditionInfluence::StructMember};
  }

  static ForbiddenPreconditionInfluenceInfo FunctionReturn() {
    return {.influence = ForbiddenPreconditionInfluence::FunctionReturn};
  }
};

inline llvm::hash_code hash_value(const ForbiddenPreconditionInfluenceInfo &info) {
  return llvm::hash_combine(
      info.influence,
      llvm::hash_combine_range(info.structMemberLocs.begin(), info.structMemberLocs.end())
  );
}

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

/// Merge two forbidden-influence summaries, preserving the first known source
/// location for each forbidden kind.
inline ForbiddenPreconditionInfluenceInfo mergeInfluenceInfo(
    ForbiddenPreconditionInfluenceInfo lhs, const ForbiddenPreconditionInfluenceInfo &rhs
) {
  lhs.influence |= rhs.influence;
  lhs.structMemberLocs.insert(rhs.structMemberLocs.begin(), rhs.structMemberLocs.end());
  return lhs;
}

template <typename T, typename... Args>
inline ForbiddenPreconditionInfluenceInfo
mergeInfluenceInfo(const T &first, const T &next, Args... args) {
  T merged = mergeInfluenceInfo(first, next);
  return mergeInfluenceInfo(merged, args...);
}

namespace detail {

using Influence = ForbiddenPreconditionInfluence;
using InfluenceInfo = ForbiddenPreconditionInfluenceInfo;

/// Cache key for one interprocedural callable-result summary query.
///
/// The analyzer memoizes result summaries by callable body, the caller-provided
/// forbidden-influence classification of each argument, and the specific result
/// number being queried. Reusing this key avoids re-analyzing the same helper
/// function under identical argument influence assumptions.
struct CallableSummaryKey {
  mlir::Operation *callable {};
  llvm::SmallVector<InfluenceInfo> argInfluences;
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

/// One included precondition that becomes illegal under a specific caller
/// binding.
///
/// The include verifier tracks each failing nested precondition separately so a
/// single `verif.include` diagnostic can report every included `require_*`
/// that becomes illegal under the caller's operand binding.
struct IncludedContractFailure {
  verif::PreconditionOpInterface precondition {};
  InfluenceInfo influenceInfo = InfluenceInfo::None();
};

/// Summary of all included-contract precondition failures under a specific
/// caller binding.
///
/// The include verifier memoizes whether a contract remains valid when its
/// entry arguments are seeded with the forbidden-influence classification of
/// the corresponding `verif.include` operands. If failures occur, this
/// structure records each offending callee precondition together with the
/// merged influence information that caused it to become illegal.
struct IncludedContractSummary {
  llvm::SmallVector<IncludedContractFailure> failures;

  explicit operator bool() const { return !failures.empty(); }
};

/// Cache key for one interprocedural included-contract summary query.
///
/// The analyzer memoizes included-contract failures by callee contract and the
/// caller-provided forbidden-influence classification of each include operand.
/// Reusing this key avoids re-analyzing the same included contract under
/// identical argument influence assumptions.
struct IncludedContractSummaryKey {
  mlir::Operation *contract {};
  llvm::SmallVector<InfluenceInfo> argInfluences;

  bool operator==(const IncludedContractSummaryKey &other) const {
    return contract == other.contract && argInfluences == other.argInfluences;
  }
};

struct IncludedContractSummaryKeyInfo : llvm::DenseMapInfo<IncludedContractSummaryKey> {
  static IncludedContractSummaryKey getEmptyKey() {
    return {llvm::DenseMapInfo<mlir::Operation *>::getEmptyKey(), {}};
  }

  static IncludedContractSummaryKey getTombstoneKey() {
    return {llvm::DenseMapInfo<mlir::Operation *>::getTombstoneKey(), {}};
  }

  static unsigned getHashValue(const IncludedContractSummaryKey &key) {
    return llvm::hash_combine(
        key.contract, llvm::hash_combine_range(key.argInfluences.begin(), key.argInfluences.end())
    );
  }

  static bool
  isEqual(const IncludedContractSummaryKey &lhs, const IncludedContractSummaryKey &rhs) {
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
  InfluenceInfo analyzeContractValue(verif::ContractOp contract, mlir::Value value);

  /// Classify the forbidden influence reaching a precondition op, including
  /// both the condition operand and any enclosing SCF control ancestors.
  InfluenceInfo
  analyzePreconditionOp(verif::ContractOp contract, verif::PreconditionOpInterface preCondOp);

  /// Summarize the forbidden influence of one callable result under the given
  /// argument influences.
  InfluenceInfo analyzeCallableResult(
      mlir::CallableOpInterface callableOp, llvm::ArrayRef<InfluenceInfo> argInfluences,
      unsigned resultNumber
  );

  /// Check whether an included contract becomes invalid under caller-provided
  /// operand influences, returning every failing callee precondition if so.
  IncludedContractSummary analyzeIncludedContract(
      verif::ContractOp calleeContract, llvm::ArrayRef<InfluenceInfo> argInfluences
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
        llvm::ArrayRef<InfluenceInfo> argInfluenceInfos
    );

    /// Recursively classify the forbidden influence reaching a single SSA value.
    InfluenceInfo analyzeValue(mlir::Value value);

    /// Classify the forbidden influence reaching a precondition op, including
    /// its enclosing SCF control-flow ancestors.
    InfluenceInfo analyzePreconditionOp(verif::PreconditionOpInterface preCondOp);

    /// Analyze an included contract under the current frame's operand bindings.
    IncludedContractSummary analyzeIncludeOp(verif::IncludeOp includeOp);

  private:
    /// Collect forbidden influence from SCF control ancestors that guard the
    /// execution of `op`.
    InfluenceInfo analyzeControlAncestors(mlir::Operation *op);

    /// Classify the control dependence introduced by one enclosing SCF op.
    InfluenceInfo analyzeAncestorControl(mlir::Operation *ancestor, mlir::Operation *nestedOp);

    /// Recover the forbidden influence reaching a region block argument.
    InfluenceInfo analyzeBlockArgument(mlir::BlockArgument blockArg);

    /// Summarize the forbidden influence produced by a call result.
    InfluenceInfo analyzeCallResult(mlir::CallOpInterface call, mlir::OpResult callRes);

    /// Summarize the forbidden influence produced by an `scf.if` result.
    InfluenceInfo analyzeIfResult(mlir::scf::IfOp ifOp, mlir::OpResult ifRes);

    /// Summarize the forbidden influence produced by an `scf.for` result.
    InfluenceInfo analyzeForResult(mlir::scf::ForOp forOp, mlir::OpResult forRes);

    /// Summarize the forbidden influence produced by an
    /// `scf.execute_region` result.
    InfluenceInfo
    analyzeExecuteRegionResult(mlir::scf::ExecuteRegionOp execOp, mlir::OpResult execRes);

    /// Summarize the forbidden influence produced by an `scf.while` result.
    InfluenceInfo analyzeWhileResult(mlir::scf::WhileOp whileOp, mlir::OpResult whileRes);

    ForbiddenInfluenceAnalyzer &analyzer;
    llvm::DenseMap<mlir::Value, InfluenceInfo> valueCache;
    llvm::DenseMap<mlir::Operation *, InfluenceInfo> controlAncestorCache;
    llvm::DenseSet<mlir::Value> activeValues;
  };

  /// Classify whether a contract entry argument is an allowed input or a
  /// forbidden target-function return value.
  static InfluenceInfo
  classifyContractArgument(verif::ContractOp contract, mlir::BlockArgument arg);

  mlir::ModuleOp module;
  llvm::DenseMap<CallableSummaryKey, InfluenceInfo, CallableSummaryKeyInfo> callableSummaryCache;
  llvm::DenseSet<CallableSummaryKey, CallableSummaryKeyInfo> activeSummaries;
  llvm::DenseMap<
      IncludedContractSummaryKey, IncludedContractSummary, IncludedContractSummaryKeyInfo>
      includedContractSummaryCache;
  llvm::DenseSet<IncludedContractSummaryKey, IncludedContractSummaryKeyInfo>
      activeIncludedSummaries;
  llvm::DenseMap<ContractOp, AnalysisFrame> cachedFrames;
};

} // namespace detail

/// Analyze whether a contract value depends on forbidden precondition sources
/// and recover representative source locations for any forbidden influence.
inline ForbiddenPreconditionInfluenceInfo analyzeForbiddenPreconditionInfluenceInfo(
    mlir::ModuleOp module, verif::ContractOp contract, mlir::Value value
) {
  return detail::ForbiddenInfluenceAnalyzer(module).analyzeContractValue(contract, value);
}

/// Analyze whether a precondition op depends on forbidden sources, including
/// both its condition operand and enclosing SCF control ancestors.
inline ForbiddenPreconditionInfluenceInfo analyzeForbiddenPreconditionOpInfluenceInfo(
    mlir::ModuleOp module, verif::ContractOp contract, verif::PreconditionOpInterface preCondOp
) {
  return detail::ForbiddenInfluenceAnalyzer(module).analyzePreconditionOp(contract, preCondOp);
}

/// Analyze whether a contract value depends on forbidden precondition sources.
inline ForbiddenPreconditionInfluence analyzeForbiddenPreconditionInfluence(
    mlir::ModuleOp module, verif::ContractOp contract, mlir::Value value
) {
  return analyzeForbiddenPreconditionInfluenceInfo(module, contract, value).influence;
}

/// Analyze whether a callable result depends on forbidden precondition sources
/// under a caller-provided argument influence summary.
inline ForbiddenPreconditionInfluenceInfo analyzeForbiddenPreconditionCallableResultInfo(
    mlir::ModuleOp module, mlir::CallableOpInterface callableOp,
    llvm::ArrayRef<ForbiddenPreconditionInfluenceInfo> argInfluences, unsigned resultNumber
) {
  return detail::ForbiddenInfluenceAnalyzer(module).analyzeCallableResult(
      callableOp, argInfluences, resultNumber
  );
}

/// Analyze whether a callable result depends on forbidden precondition sources
/// under a caller-provided argument influence summary.
inline ForbiddenPreconditionInfluence analyzeForbiddenPreconditionCallableResult(
    mlir::ModuleOp module, mlir::CallableOpInterface callableOp,
    llvm::ArrayRef<ForbiddenPreconditionInfluenceInfo> argInfluences, unsigned resultNumber
) {
  return analyzeForbiddenPreconditionCallableResultInfo(
             module, callableOp, argInfluences, resultNumber
  )
      .influence;
}

/// Analyze whether including a contract with caller-provided operand influence
/// summaries would trigger a forbidden precondition failure in the callee.
inline detail::IncludedContractSummary analyzeForbiddenIncludedContractSummary(
    mlir::ModuleOp module, verif::ContractOp calleeContract,
    llvm::ArrayRef<ForbiddenPreconditionInfluenceInfo> argInfluences
) {
  return detail::ForbiddenInfluenceAnalyzer(module).analyzeIncludedContract(
      calleeContract, argInfluences
  );
}

} // namespace llzk::verif
