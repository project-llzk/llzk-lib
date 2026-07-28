//===-- LLZKAggregateScalarizationPass.cpp ----------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-aggregate-scalarization` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Array/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/POD/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "llzk/Dialect/Verif/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>

namespace llzk {
#define GEN_PASS_DEF_AGGREGATESCALARIZATIONPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk::array;
using namespace llzk::component;
using namespace llzk::function;
using namespace llzk::pod;
using namespace llzk::verif;

namespace {

struct AggregateProfile {
  llvm::SmallVector<size_t> countsByDepth;

  void addAtDepth(size_t depth) {
    if (countsByDepth.size() <= depth) {
      countsByDepth.resize(depth + 1, 0);
    }
    ++countsByDepth[depth];
  }

  size_t maxDepth() const {
    for (size_t i = countsByDepth.size(); i != 0; --i) {
      if (countsByDepth[i - 1] != 0) {
        return i - 1;
      }
    }
    return 0;
  }

  bool empty() const {
    return llvm::all_of(countsByDepth, [](size_t count) { return count == 0; });
  }

  bool strictlyImprovesFrom(const AggregateProfile &before) const {
    size_t maxSize = std::max(countsByDepth.size(), before.countsByDepth.size());
    for (size_t i = maxSize; i != 0; --i) {
      size_t depth = i - 1;
      size_t afterCount = depth < countsByDepth.size() ? countsByDepth[depth] : 0;
      size_t beforeCount = depth < before.countsByDepth.size() ? before.countsByDepth[depth] : 0;
      if (afterCount != beforeCount) {
        return afterCount < beforeCount;
      }
    }
    return false;
  }
};

static LogicalResult collectAggregateProfileForType(
    Type type, Operation *origin, SymbolTableCollection &tables, AggregateProfile &profile,
    llvm::SmallDenseSet<StructDefOp, 8> &visitedStructs, size_t depth
) {
  if (auto arrayType = dyn_cast<ArrayType>(type)) {
    if (arrayType.hasStaticShape()) {
      profile.addAtDepth(depth);
      return collectAggregateProfileForType(
          arrayType.getElementType(), origin, tables, profile, visitedStructs, depth + 1
      );
    }
    return success();
  }

  if (auto podType = dyn_cast<PodType>(type)) {
    profile.addAtDepth(depth);
    for (RecordAttr record : podType.getRecords()) {
      if (failed(collectAggregateProfileForType(
              record.getType(), origin, tables, profile, visitedStructs, depth + 1
          ))) {
        return failure();
      }
    }
    return success();
  }

  auto structType = dyn_cast<StructType>(type);
  if (!structType) {
    return success();
  }

  FailureOr<StructDefOp> structDef = llzk::verifyStructTypeResolution(tables, structType, origin);
  if (failed(structDef)) {
    return failure();
  }
  if (!visitedStructs.insert(*structDef).second) {
    return success();
  }

  for (MemberDefOp member : structDef->getMemberDefs()) {
    if (failed(collectAggregateProfileForType(
            member.getType(), origin, tables, profile, visitedStructs, depth
        ))) {
      visitedStructs.erase(*structDef);
      return failure();
    }
  }
  visitedStructs.erase(*structDef);
  return success();
}

static LogicalResult collectAggregateProfileForTypes(
    ArrayRef<Type> types, Operation *origin, SymbolTableCollection &tables,
    AggregateProfile &profile
) {
  llvm::SmallDenseSet<StructDefOp, 8> visitedStructs;
  for (Type type : types) {
    if (failed(collectAggregateProfileForType(
            type, origin, tables, profile, visitedStructs, /*depth=*/0
        ))) {
      return failure();
    }
  }
  return success();
}

static FailureOr<AggregateProfile> collectAggregateProfile(ModuleOp module) {
  SymbolTableCollection tables;
  AggregateProfile profile;

  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (auto structDef = dyn_cast<StructDefOp>(op)) {
      llvm::SmallVector<Type> memberTypes;
      for (MemberDefOp member : structDef.getMemberDefs()) {
        memberTypes.push_back(member.getType());
      }
      if (failed(collectAggregateProfileForTypes(memberTypes, structDef, tables, profile))) {
        return WalkResult::interrupt();
      }
    } else if (auto funcDef = dyn_cast<FuncDefOp>(op)) {
      llvm::SmallVector<Type> types;
      llvm::append_range(types, funcDef.getArgumentTypes());
      llvm::append_range(types, funcDef.getResultTypes());
      if (failed(collectAggregateProfileForTypes(types, funcDef, tables, profile))) {
        return WalkResult::interrupt();
      }
    } else if (auto contract = dyn_cast<ContractOp>(op)) {
      llvm::SmallVector<Type> types;
      llvm::append_range(types, contract.getArgumentTypes());
      llvm::append_range(types, contract.getResultTypes());
      if (failed(collectAggregateProfileForTypes(types, contract, tables, profile))) {
        return WalkResult::interrupt();
      }
    } else if (auto createArray = dyn_cast<CreateArrayOp>(op)) {
      if (failed(collectAggregateProfileForTypes(
              ArrayRef<Type> {createArray.getType()}, createArray, tables, profile
          ))) {
        return WalkResult::interrupt();
      }
    } else if (auto newPod = dyn_cast<NewPodOp>(op)) {
      if (failed(collectAggregateProfileForTypes(
              ArrayRef<Type> {newPod.getType()}, newPod, tables, profile
          ))) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    return failure();
  }
  return profile;
}

class AggregateScalarizationPass
    : public llzk::impl::AggregateScalarizationPassBase<AggregateScalarizationPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    constexpr unsigned kMaxIterations = 8;

    FailureOr<AggregateProfile> beforeProfile = collectAggregateProfile(module);
    if (failed(beforeProfile)) {
      signalPassFailure();
      return;
    }
    if (beforeProfile->empty()) {
      markAllAnalysesPreserved();
      return;
    }

    for (unsigned iteration = 0; iteration < kMaxIterations; ++iteration) {
      OpPassManager pm(ModuleOp::getOperationName());
      pm.addPass(llzk::array::createArrayToScalarPass());
      pm.addPass(llzk::pod::createPodToScalarPass());
      pm.addPass(createCanonicalizerPass());
      if (failed(runPipeline(pm, module))) {
        signalPassFailure();
        return;
      }

      FailureOr<AggregateProfile> afterProfile = collectAggregateProfile(module);
      if (failed(afterProfile)) {
        signalPassFailure();
        return;
      }
      if (afterProfile->empty()) {
        return;
      }
      if (!afterProfile->strictlyImprovesFrom(*beforeProfile)) {
        module.emitError()
            << "aggregate scalarization round did not make semantic progress toward scalar-only IR";
        signalPassFailure();
        return;
      }
      beforeProfile = std::move(afterProfile);
    }

    module.emitError() << "aggregate scalarization exceeded the iteration limit while "
                          "still making progress toward scalar-only IR";
    signalPassFailure();
  }
};

} // namespace
