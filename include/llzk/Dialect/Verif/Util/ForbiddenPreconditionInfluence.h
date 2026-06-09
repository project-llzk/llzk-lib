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

inline bool any(ForbiddenPreconditionInfluence influence) {
  return static_cast<uint8_t>(influence) != 0;
}

inline bool
hasInfluence(ForbiddenPreconditionInfluence influence, ForbiddenPreconditionInfluence flag) {
  return (static_cast<uint8_t>(influence) & static_cast<uint8_t>(flag)) != 0;
}

namespace detail {

using Influence = ForbiddenPreconditionInfluence;

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

class ForbiddenInfluenceAnalyzer {
public:
  explicit ForbiddenInfluenceAnalyzer(mlir::ModuleOp module) : module(module) {}

  Influence analyzeContractValue(verif::ContractOp contract, mlir::Value value);

  Influence analyzeCallableResult(
      mlir::Operation *callable, llvm::ArrayRef<Influence> argInfluences, unsigned resultNumber
  );

private:
  class AnalysisFrame {
  public:
    AnalysisFrame(
        ForbiddenInfluenceAnalyzer &analyzer, mlir::Operation *callableOp,
        llvm::ArrayRef<Influence> argInfluences
    );

    Influence analyzeValue(mlir::Value value);

  private:
    Influence analyzeBlockArgument(mlir::BlockArgument blockArg);

    Influence analyzeCallResult(mlir::CallOpInterface call, unsigned resultNumber);

    Influence analyzeIfResult(mlir::scf::IfOp ifOp, unsigned resultNumber);

    Influence analyzeForResult(mlir::scf::ForOp forOp, unsigned resultNumber);

    Influence analyzeWhileResult(mlir::scf::WhileOp whileOp, unsigned resultNumber);

    ForbiddenInfluenceAnalyzer &analyzer;
    llvm::DenseMap<mlir::Value, Influence> valueCache;
    llvm::DenseSet<mlir::Value> activeValues;
  };

  static Influence classifyContractArgument(
      verif::ContractOp contract, mlir::Block *entryBlock, mlir::BlockArgument arg
  );

  mlir::ModuleOp module;
  llvm::DenseMap<CallableSummaryKey, Influence, CallableSummaryKeyInfo> callableSummaryCache;
  llvm::DenseSet<CallableSummaryKey, CallableSummaryKeyInfo> activeSummaries;
};

} // namespace detail

inline ForbiddenPreconditionInfluence analyzeForbiddenPreconditionInfluence(
    mlir::ModuleOp module, verif::ContractOp contract, mlir::Value value
) {
  return detail::ForbiddenInfluenceAnalyzer(module).analyzeContractValue(contract, value);
}

inline ForbiddenPreconditionInfluence analyzeForbiddenPreconditionCallableResult(
    mlir::ModuleOp module, mlir::Operation *callable,
    llvm::ArrayRef<ForbiddenPreconditionInfluence> argInfluences, unsigned resultNumber
) {
  return detail::ForbiddenInfluenceAnalyzer(module).analyzeCallableResult(
      callable, argInfluences, resultNumber
  );
}

} // namespace llzk::verif
