//===-- FlatteningContext.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Instantiation history and loop-preservation state for flattening.
///
//===----------------------------------------------------------------------===//

#include "FlatteningContext.h"

#include "TemplateInstantiation.h"

#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/TypeHelper.h"
#include "llzk/Util/Walk.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Utils/Utils.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Matchers.h>

#include <llvm/Support/Debug.h>

#include <functional>

#define DEBUG_TYPE "llzk-flatten"

using namespace mlir;
using namespace llzk;
using namespace llzk::component;
using namespace llzk::function;
using namespace llzk::polymorphic;
using namespace llzk::polymorphic::detail;

namespace llzk::polymorphic::detail::flattening {

/// Compute the trip count from the loop itself, independently of any trusted annotation.
std::optional<APInt> getConstantTripCount(LoopLikeOpInterface loopOp) {
  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp.getOperation());
      whileOp && matchPattern(whileOp.getConditionOp().getCondition(), m_Zero())) {
    return APInt(64, 0);
  }
  std::optional<OpFoldResult> lbVal = loopOp.getSingleLowerBound();
  std::optional<OpFoldResult> ubVal = loopOp.getSingleUpperBound();
  std::optional<OpFoldResult> stepVal = loopOp.getSingleStep();
  if (!lbVal || !ubVal || !stepVal) {
    return std::nullopt;
  }
  bool isSigned = true;
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp.getOperation())) {
    isSigned = !forOp.getUnsignedCmp();
  }
  return constantTripCount(*lbVal, *ubVal, *stepVal, isSigned, scf::computeUbMinusLb);
}

/// Explain why a supported loop must remain rolled, using annotations before computed bounds.
static std::optional<std::string> getUnrollSkipReason(Operation *op, uint64_t limit) {
  if (!isa<scf::ForOp, scf::WhileOp, affine::AffineForOp>(op)) {
    return std::nullopt;
  }
  uint64_t tripCount;
  auto bounds = op->getAttrOfType<LoopBoundsAttr>(LoopBoundsAttr::name);
  if (bounds) {
    if (bounds.getStep() == 0) {
      return std::nullopt; // The annotated unroller diagnoses invalid steps separately.
    }
    llvm::DynamicAPInt count =
        llvm::ceilDiv(bounds.getUpper() - bounds.getLower(), bounds.getStep());
    auto checked = checkedToUInt64(std::max(count, llvm::DynamicAPInt(0)));
    if (!checked) {
      llvm::consumeError(checked.takeError());
      return "llzk.loopbounds trip count does not fit in uint64_t; leaving loop unchanged";
    }
    tripCount = *checked;
  } else {
    if (isa<scf::WhileOp>(op)) {
      return std::nullopt;
    }
    auto count = getConstantTripCount(llvm::cast<LoopLikeOpInterface>(op));
    if (!count) {
      return std::nullopt;
    }
    if (count->getActiveBits() > 64) {
      return "computed loop trip count does not fit in uint64_t; leaving loop unchanged";
    }
    tripCount = count->getZExtValue();
  }
  if (tripCount > limit) {
    return ("loop trip count " + Twine(tripCount) + " exceeds max-unroll limit of " + Twine(limit) +
            "; leaving loop unchanged")
        .str();
  }
  if (bounds && tripCount != 0) {
    Value iv;
    bool isSigned = true;
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      iv = forOp.getInductionVar();
      isSigned = !forOp.getUnsignedCmp();
    } else if (auto affineForOp = dyn_cast<affine::AffineForOp>(op)) {
      iv = affineForOp.getInductionVar();
    }
    if (iv && !iv.use_empty()) {
      Type ivType = iv.getType();
      unsigned width =
          ivType.isIndex() ? IndexType::kInternalStorageBitWidth : ivType.getIntOrFloatBitWidth();
      const llvm::DynamicAPInt &lower = bounds.getLower();
      auto last = lower + toDynamicAPInt(tripCount - 1) * bounds.getStep();
      // All intermediate induction values lie between these endpoints.
      for (const auto &value : {lower, last}) {
        auto encoded = checkedToAPInt(value, width, isSigned);
        if (!encoded) {
          llvm::consumeError(encoded.takeError());
          return "llzk.loopbounds induction value does not fit the loop induction type; "
                 "leaving loop unchanged";
        }
      }
    }
  }
  return std::nullopt;
}

std::optional<SymbolRefAttr>
FunctionInstantiationCache::lookup(FuncDefOp sourceFunc, ArrayAttr concreteParamKey) const {
  auto found = entries.find(sourceFunc.getOperation());
  if (found == entries.end()) {
    return std::nullopt;
  }
  for (const Entry &candidate : found->second) {
    if (candidate.concreteParamKey == concreteParamKey) {
      return candidate.functionPath;
    }
  }
  return std::nullopt;
}

void FunctionInstantiationCache::record(
    FuncDefOp sourceFunc, ArrayAttr concreteParamKey, SymbolRefAttr functionPath
) {
  assert(
      !lookup(sourceFunc, concreteParamKey).has_value() && "function instantiation cached twice"
  );
  entries[sourceFunc.getOperation()].push_back(Entry {concreteParamKey, functionPath});
}

bool ConversionTracker::shouldSkipUnroll(Operation *op) {
  auto reason = getUnrollSkipReason(op, unrollIterationLimit);
  if (!reason) {
    return false;
  }
  if (unrollWarnings.insert({op->getLoc(), StringAttr::get(op->getContext(), *reason)}).second) {
    op->emitWarning(*reason);
  }
  return true;
}

bool ConversionTracker::mustPreserveLoop(Operation *op) {
  if (!op) {
    return false;
  }
  for (Operation *p = op; p && !isa<SymbolOpInterface>(p); p = p->getParentOp()) {
    if (shouldSkipUnroll(p)) {
      return true;
    }
  }
  return walkContains<Operation *>(*op, [this](Operation *nested) {
    return shouldSkipUnroll(nested);
  });
}

DenseSet<Operation *> ConversionTracker::getPreservedLoopOps(ModuleOp modOp) {
  DenseSet<Operation *> preserved;
  modOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!shouldSkipUnroll(op)) {
      return WalkResult::advance();
    }
    op->walk([&preserved](Operation *nested) {
      if (nested->getNumRegions() != 0) {
        preserved.insert(nested);
      }
    });
    for (Operation *p = op->getParentOp(); p && !isa<SymbolOpInterface>(p); p = p->getParentOp()) {
      preserved.insert(p);
    }
    return WalkResult::skip();
  });
  return preserved;
}

void ConversionTracker::recordInstantiation(SymbolRefAttr funcName) {
  funcInstantiations.insert(funcName);
  modified = true;
}

void ConversionTracker::recordInstantiation(StructType oldType, StructType newType) {
  assert(!isNullOrEmpty(oldType.getParams()) && "cannot instantiate with no params");

  auto forwardResult = structInstantiations.try_emplace(oldType, newType);
  if (forwardResult.second) {
    // Insertion was successful
    // ASSERT: The reverse map does not contain this mapping either
    assert(!reverseInstantiations.contains(newType));
    reverseInstantiations[newType] = oldType;
    // Set the modified flag
    modified = true;
  } else {
    // ASSERT: If a mapping already existed for `oldType` it must be `newType`
    assert(forwardResult.first->getSecond() == newType);
    // ASSERT: The reverse mapping is already present as well
    assert(reverseInstantiations.lookup(newType) == oldType);
  }
  assert(structInstantiations.size() == reverseInstantiations.size());
}

std::optional<StructType> ConversionTracker::getInstantiation(StructType oldType) const {
  auto cachedResult = structInstantiations.find(oldType);
  if (cachedResult != structInstantiations.end()) {
    return cachedResult->second;
  }
  return std::nullopt;
}

std::optional<StructType> ConversionTracker::getPreimage(StructType newType) const {
  auto cachedResult = reverseInstantiations.find(newType);
  if (cachedResult != reverseInstantiations.end()) {
    return cachedResult->second;
  }
  return std::nullopt;
}

std::optional<SymbolRefAttr> ConversionTracker::lookupPartialFuncInstantiation(
    FuncDefOp sourceFunc, ArrayAttr concreteParamKey
) const {
  return partialFuncInstantiations.lookup(sourceFunc, concreteParamKey);
}

void ConversionTracker::recordPartialFuncInstantiation(
    FuncDefOp sourceFunc, ArrayAttr concreteParamKey, TemplateOp templateOp, FuncDefOp functionOp
) {
  partialFuncInstantiations.record(
      sourceFunc, concreteParamKey,
      asSymbolRefAttr(
          SmallVector<FlatSymbolRefAttr> {
              FlatSymbolRefAttr::get(templateOp.getSymNameAttr()),
              FlatSymbolRefAttr::get(functionOp.getSymNameAttr()),
          }
      )
  );
}

DenseSet<SymbolRefAttr> ConversionTracker::getInstantiatedDefinitionNames() const {
  DenseSet<SymbolRefAttr> instantiatedNames = funcInstantiations;
  for (const auto &[origRemoteTy, _] : structInstantiations) {
    instantiatedNames.insert(origRemoteTy.getNameRef());
  }
  return instantiatedNames;
}

void ConversionTracker::reportDelayedDiagnostics(StructType newType, CallOp caller) {
  auto res = delayedDiagnostics.find(newType);
  if (res != delayedDiagnostics.end()) {
    llzk::polymorphic::detail::reportDelayedDiagnostics(caller, std::move(res->second));

    // Emitting a Diagnostic consumes it (per DiagnosticEngine::emit) so remove them from the map.
    // Unfortunately, this means if the key StructType is the result of instantiation at multiple
    // `compute()` calls it will only be reported at one of those locations, not all.
    delayedDiagnostics.erase(newType);
  }
}

std::optional<SymbolRefAttr> ConversionTracker::getFullFuncInstantiation(
    FuncDefOp sourceFunc, ArrayAttr concreteParamKey
) const {
  return fullFuncInstantiations.lookup(sourceFunc, concreteParamKey);
}

void ConversionTracker::recordFullFuncInstantiation(
    FuncDefOp sourceFunc, ArrayAttr concreteParamKey, SymbolRefAttr instantiatedPath
) {
  fullFuncInstantiations.record(sourceFunc, concreteParamKey, instantiatedPath);
}

bool ConversionTracker::isLegalConversion(Type oldType, Type newType, const char *patName) const {
  std::function<bool(Type, Type)> checkInstantiations = [&](Type oTy, Type nTy) {
    // Check if `oTy` is a struct with a known instantiation to `nTy`
    if (StructType oldStructType = llvm::dyn_cast<StructType>(oTy)) {
      // Note: The values in `structInstantiations` must be no-parameter struct types
      // so there is no need for recursive check, simple equality is sufficient.
      if (this->structInstantiations.lookup(oldStructType) == nTy) {
        return true;
      }
    }
    // Check if `nTy` is the result of a struct instantiation and if the pre-image of
    // that instantiation (i.e., the parameterized version of the instantiated struct)
    // is a more concrete unification of `oTy`.
    if (StructType newStructType = llvm::dyn_cast<StructType>(nTy)) {
      if (auto preImage = this->reverseInstantiations.lookup(newStructType)) {
        if (isMoreConcreteUnification(oTy, preImage, checkInstantiations)) {
          return true;
        }
      }
    }
    return false;
  };

  if (isMoreConcreteUnification(oldType, newType, checkInstantiations)) {
    return true;
  }
  LLVM_DEBUG(
      llvm::dbgs() << "[" << patName << "] Cannot replace old type " << oldType << " with new type "
                   << newType
                   << " because it does not define a compatible and more concrete type.\n";
  );
  return false;
}

} // namespace llzk::polymorphic::detail::flattening
