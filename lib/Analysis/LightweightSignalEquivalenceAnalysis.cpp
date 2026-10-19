//===- LightweightSignalEquivalenceAnalysis.cpp -----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from mlir/lib/Analysis/DataFlow/SparseAnalysis.cpp.
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a basic signal equivalence analysis
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/LightweightSignalEquivalenceAnalysis.h"

#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Shared/ValueCopy.h"
#include "llzk/Dialect/Struct/IR/Ops.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

#include <optional>

#define DEBUG_TYPE "llzk-signal-equivalence"

using namespace mlir;
using namespace llzk::component;
using namespace llzk::pod;

namespace llzk {

namespace {

struct PodObservation {
  Value root;
  SmallVector<StringAttr> path;
  Operation *rootObservation = nullptr;
};

/// Return whether \p value is used by \p op or one of its nested operations.
static bool operationUsesValue(Operation *op, Value value) {
  return llvm::any_of(value.getUses(), [op](OpOperand &use) {
    Operation *owner = use.getOwner();
    return owner == op || op->isProperAncestor(owner);
  });
}

/// Visit the operations strictly between two same-block anchors.
template <typename Callback>
static bool allBetween(Operation *lhs, Operation *rhs, Callback &&callback) {
  if (lhs->getBlock() != rhs->getBlock()) {
    return false;
  }
  if (lhs == rhs) {
    return true;
  }

  Operation *first = lhs->isBeforeInBlock(rhs) ? lhs : rhs;
  Operation *last = first == lhs ? rhs : lhs;
  for (Operation *cur = first->getNextNode(); cur != last; cur = cur->getNextNode()) {
    if (!callback(cur)) {
      return false;
    }
  }
  return true;
}

/// Check that the selected direct record of a POD cannot have changed between
/// two observations. Uses nested in another operation are intentionally treated
/// as unknown barriers.
static bool
podRecordUnmodified(Value storage, StringAttr selectedRecord, Operation *lhs, Operation *rhs) {
  return allBetween(lhs, rhs, [&](Operation *op) {
    if (!operationUsesValue(op, storage)) {
      return true;
    }

    // A nested-region use is not one of the locally understood copy operations.
    for (OpOperand &use : storage.getUses()) {
      if (op->isProperAncestor(use.getOwner())) {
        return false;
      }
    }

    if (auto read = dyn_cast<ReadPodOp>(op)) {
      return read.getPodRef() == storage;
    }
    if (auto write = dyn_cast<WritePodOp>(op)) {
      if (write.getPodRef() == storage) {
        return write.getRecordNameAttr() != selectedRecord;
      }
      return write.getValue() == storage;
    }
    if (auto alloc = dyn_cast<NewPodOp>(op)) {
      return llvm::is_contained(alloc.getInitialValues(), storage);
    }
    if (auto write = dyn_cast<MemberWriteOp>(op)) {
      return write.getVal() == storage;
    }
    return false;
  });
}

/// Return whether \p op contains a nested write that may target the member read
/// by \p read. Struct values are identity-bearing handles, so a write through a
/// different SSA value cannot be assumed to target a distinct component.
static bool containsPotentiallyAliasingMemberWrite(Operation *op, MemberReadOp read) {
  WalkResult result = op->walk([&](MemberWriteOp write) {
    if (write.getOperation() != op && write.getMemberNameAttr() == read.getMemberNameAttr()) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

/// Find the reaching same-block member write for a read. The search stops at an
/// unknown use of the component or a potentially aliasing same-member write,
/// and only an exact component/member match can be forwarded.
static MemberWriteOp findReachingMemberWrite(MemberReadOp read) {
  if (read.getTableOffset().has_value()) {
    return nullptr;
  }

  Value component = read.getComponent();
  for (Operation *cur = read->getPrevNode(); cur; cur = cur->getPrevNode()) {
    if (auto memberWrite = dyn_cast<MemberWriteOp>(cur)) {
      if (memberWrite.getComponent() == component) {
        if (memberWrite.getMemberNameAttr() != read.getMemberNameAttr()) {
          continue;
        }
        if (memberWrite.getVal().getType() != read.getVal().getType()) {
          return nullptr;
        }
        return memberWrite;
      }

      // A struct handle copied through a POD, struct member, or another
      // identity-preserving operation may denote the same component despite
      // having a distinct SSA value.
      if (memberWrite.getMemberNameAttr() == read.getMemberNameAttr()) {
        return nullptr;
      }
    }

    if (containsPotentiallyAliasingMemberWrite(cur, read)) {
      return nullptr;
    }

    if (!operationUsesValue(cur, component)) {
      continue;
    }

    for (OpOperand &use : component.getUses()) {
      if (cur->isProperAncestor(use.getOwner())) {
        return nullptr;
      }
    }

    if (auto memberRead = dyn_cast<MemberReadOp>(cur)) {
      if (memberRead.getComponent() == component) {
        continue;
      }
      return nullptr;
    }
    return nullptr;
  }
  return nullptr;
}

/// Normalize a POD storage value whose selected suffix is observed at
/// \p observation. Every traversed copy edge is checked before following it.
static bool normalizePodStorage(
    Value storage, Operation *observation, SmallVector<StringAttr> &path, PodObservation &result
) {
  if (path.empty()) {
    return false;
  }

  if (auto nestedRead = storage.getDefiningOp<ReadPodOp>()) {
    if (!isa<PodType>(nestedRead.getType()) ||
        !podRecordUnmodified(storage, path.front(), nestedRead, observation)) {
      return false;
    }
    path.insert(path.begin(), nestedRead.getRecordNameAttr());
    return normalizePodStorage(nestedRead.getPodRef(), nestedRead, path, result);
  }

  if (auto memberRead = storage.getDefiningOp<MemberReadOp>()) {
    if (!isa<PodType>(memberRead.getType()) ||
        !podRecordUnmodified(storage, path.front(), memberRead, observation)) {
      return false;
    }
    MemberWriteOp write = findReachingMemberWrite(memberRead);
    if (!write || !isa<PodType>(write.getVal().getType())) {
      return false;
    }
    return normalizePodStorage(write.getVal(), write, path, result);
  }

  result.root = storage;
  result.path = path;
  result.rootObservation = observation;
  return true;
}

static std::optional<PodObservation> normalizePodRead(ReadPodOp read) {
  PodObservation result;
  SmallVector<StringAttr> path {read.getRecordNameAttr()};
  if (!normalizePodStorage(read.getPodRef(), read, path, result)) {
    return std::nullopt;
  }
  return result;
}

/// Forward an exact, non-offset member read to its reaching write. POD-valued
/// reads are normalized separately so their copy point remains visible.
static Value forwardScalarMemberRead(Value value) {
  auto read = value.getDefiningOp<MemberReadOp>();
  if (!read || isa<PodType>(read.getType())) {
    return value;
  }
  if (MemberWriteOp write = findReachingMemberWrite(read)) {
    return write.getVal();
  }
  return value;
}

} // namespace

LightweightSignalEquivalenceAnalysis::LightweightSignalEquivalenceAnalysis(Operation *) {}

bool LightweightSignalEquivalenceAnalysis::areSignalsEquivalent(Value v1, Value v2) {
  if (v1 == v2) {
    return true;
  }
  if (equivalentSignals.isEquivalent(v1, v2)) {
    return true;
  }

  Value originalV1 = v1;
  Value originalV2 = v2;
  v1 = forwardScalarMemberRead(v1);
  v2 = forwardScalarMemberRead(v2);
  LLVM_DEBUG(llvm::outs() << "Asking for equivalence between " << v1 << " and " << v2 << '\n');
  if (v1 == v2) {
    equivalentSignals.unionSets(originalV1, originalV2);
    return true;
  }
  if (equivalentSignals.isEquivalent(v1, v2)) {
    equivalentSignals.unionSets(originalV1, originalV2);
    return true;
  }

  Operation *o1 = v1.getDefiningOp();
  Operation *o2 = v2.getDefiningOp();

  if (o1 == nullptr || o2 == nullptr) {
    return false;
  }

  auto podRead1 = dyn_cast<ReadPodOp>(o1);
  auto podRead2 = dyn_cast<ReadPodOp>(o2);
  if (podRead1 || podRead2) {
    if (!podRead1 || !podRead2) {
      return false;
    }
    // A POD read of a mutable aggregate creates an independent snapshot. Matching
    // the source record does not prove those snapshots equivalent after either
    // result is mutated, so leave such values to a mutation-aware analysis.
    if (llzk::requiresValueCopy(podRead1.getType()) ||
        llzk::requiresValueCopy(podRead2.getType())) {
      return false;
    }
    std::optional<PodObservation> obs1 = normalizePodRead(podRead1);
    std::optional<PodObservation> obs2 = normalizePodRead(podRead2);
    if (!obs1 || !obs2 || obs1->root != obs2->root || obs1->path != obs2->path ||
        !podRecordUnmodified(
            obs1->root, obs1->path.front(), obs1->rootObservation, obs2->rootObservation
        )) {
      return false;
    }
    equivalentSignals.unionSets(originalV1, originalV2);
    return true;
  }

  // Allocations denote distinct mutable POD states, even if their operations
  // happen to have identical names and operands.
  if (isa<NewPodOp>(o1) || isa<NewPodOp>(o2)) {
    return false;
  }

  if (o1->getName() != o2->getName()) {
    return false;
  }

  if (o1->getNumOperands() != o2->getNumOperands()) {
    return false;
  }

  for (size_t i = 0; i < o1->getNumOperands(); i++) {
    if (!areSignalsEquivalent(o1->getOperand(i), o2->getOperand(i))) {
      return false;
    }
  }

  equivalentSignals.unionSets(originalV1, originalV2);
  return true;
}
} // namespace llzk
