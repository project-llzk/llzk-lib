//===-- WildcardArraySpecializationPass.cpp ---------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-specialize-wildcard-arrays` pass.
///
//===----------------------------------------------------------------------===//

#include "SharedImpl.h"

#include "llzk/Analysis/SymbolDefTree.h"
#include "llzk/Analysis/SymbolUseGraph.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Shared/TypeConversionPatterns.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/Debug.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolLookup.h"
#include "llzk/Util/SymbolTableLLZK.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>

// Include the generated base pass class definitions.
namespace llzk::polymorphic {
#define GEN_PASS_DEF_WILDCARDARRAYSPECIALIZATIONPASS
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h.inc"
} // namespace llzk::polymorphic

#define DEBUG_TYPE "llzk-specialize-wildcard-arrays"

using namespace mlir;
using namespace llzk;
using namespace llzk::array;
using namespace llzk::component;
using namespace llzk::function;
using namespace llzk::polymorphic;
using namespace llzk::polymorphic::detail;

namespace {

/// Tracks specializations created during the pass so later rewrites can validate
/// that any type change is a legal move toward a more concrete program.
class ConversionTracker {
  bool modified = false;
  DenseMap<StructType, SmallVector<StructType>> structSpecializations;
  DenseMap<StructType, StructType> reverseSpecializations;
  DenseSet<SymbolRefAttr> funcInstantiations;

public:
  bool isModified() const { return modified; }
  void resetModifiedFlag() { modified = false; }
  void updateModifiedFlag(bool currStepModified) { modified |= currStepModified; }

  void recordInstantiation(SymbolRefAttr funcName) {
    funcInstantiations.insert(funcName);
    modified = true;
  }

  void recordSpecialization(StructType oldType, StructType newType) {
    assert(isNullOrEmpty(oldType.getParams()) && "wildcard-array specialization expects plain key");
    SmallVector<StructType> &specializations = structSpecializations[oldType];
    if (llvm::is_contained(specializations, newType)) {
      assert(reverseSpecializations.lookup(newType) == oldType);
      return;
    }
    specializations.push_back(newType);
    auto [it, inserted] = reverseSpecializations.try_emplace(newType, oldType);
    (void)it;
    (void)inserted;
    assert(inserted || it->second == oldType);
    modified = true;
  }

  DenseSet<SymbolRefAttr> getInstantiatedDefinitionNames() const {
    DenseSet<SymbolRefAttr> instantiatedNames = funcInstantiations;
    for (const auto &[origRemoteTy, _] : structSpecializations) {
      instantiatedNames.insert(origRemoteTy.getNameRef());
    }
    return instantiatedNames;
  }

  bool isLegalConversion(Type oldType, Type newType, const char *patName) const {
    std::function<bool(Type, Type)> checkSpecializations = [&](Type oTy, Type nTy) {
      if (StructType oldStructType = llvm::dyn_cast<StructType>(oTy)) {
        auto specializationIt = structSpecializations.find(oldStructType);
        if (specializationIt != structSpecializations.end() &&
            llvm::is_contained(specializationIt->second, nTy)) {
          return true;
        }
      }
      if (StructType newStructType = llvm::dyn_cast<StructType>(nTy)) {
        if (StructType preImage = reverseSpecializations.lookup(newStructType)) {
          if (isMoreConcreteUnification(oTy, preImage, checkSpecializations)) {
            return true;
          }
        }
      }
      return false;
    };

    if (!isMoreConcreteUnification(oldType, newType, checkSpecializations)) {
      LLVM_DEBUG({
        llvm::dbgs() << '[' << patName << "] invalid type conversion from " << oldType << " to "
                     << newType << '\n';
      });
      return false;
    }
    return true;
  }

  bool areLegalConversions(TypeRange oldTypes, TypeRange newTypes, const char *patName) const {
    return oldTypes.size() == newTypes.size() &&
           llvm::all_of(llvm::zip_equal(oldTypes, newTypes), [&](auto pair) {
      return isLegalConversion(std::get<0>(pair), std::get<1>(pair), patName);
    });
  }
};

/// Turns pattern match failures into hard pass failures with diagnostics.
struct MatchFailureListener : public RewriterBase::Listener {
  bool hadFailure = false;

  void notifyMatchFailure(Location loc, function_ref<void(Diagnostic &)> reasonCallback) override {
    InFlightDiagnostic diag = emitError(loc);
    reasonCallback(*diag.getUnderlyingDiagnostic());
    diag.report();
    hadFailure = true;
  }
};

static LogicalResult
applyAndFoldGreedily(ModuleOp modOp, ConversionTracker &tracker, RewritePatternSet &&patterns) {
  bool currStepModified = false;
  MatchFailureListener failureListener;
  LogicalResult result = applyPatternsGreedily(
      modOp->getRegion(0), std::move(patterns),
      GreedyRewriteConfig {.maxIterations = 20, .listener = &failureListener, .fold = true},
      &currStepModified
  );
  tracker.updateModifiedFlag(currStepModified);
  return failure(result.failed() || failureListener.hadFailure);
}

/// Records every wildcard array type replaced while specializing one call site.
struct WildcardArraySpecializationInfo {
  DenseMap<ArrayType, ArrayType> replacements;
  SmallVector<std::pair<ArrayType, ArrayType>> ordered;

  bool empty() const { return ordered.empty(); }

  LogicalResult record(ArrayType oldTy, ArrayType newTy) {
    auto it = replacements.find(oldTy);
    if (it == replacements.end()) {
      replacements.try_emplace(oldTy, newTy);
      ordered.emplace_back(oldTy, newTy);
      return success();
    }
    return success(it->second == newTy);
  }

  SmallVector<Attribute> getConcreteTypeAttrs() const {
    SmallVector<Attribute> attrs;
    attrs.reserve(ordered.size());
    for (const auto &[_, newTy] : ordered) {
      attrs.push_back(TypeAttr::get(newTy));
    }
    return attrs;
  }
};

/// Returns whether `type` contains an `array.type` with at least one dynamic
/// dimension anywhere in the nested type structure.
static bool containsWildcardArrayDims(Type type) {
  if (ArrayType arrTy = llvm::dyn_cast<ArrayType>(type)) {
    if (llvm::any_of(arrTy.getDimensionSizes(), [](Attribute dim) {
      if (IntegerAttr intAttr = llvm::dyn_cast<IntegerAttr>(dim)) {
        return isDynamic(intAttr);
      }
      return false;
    })) {
      return true;
    }
    return containsWildcardArrayDims(arrTy.getElementType());
  }
  if (StructType structTy = llvm::dyn_cast<StructType>(type)) {
    if (ArrayAttr params = structTy.getParams()) {
      return llvm::any_of(params.getValue(), [](Attribute attr) {
        if (TypeAttr typeAttr = llvm::dyn_cast<TypeAttr>(attr)) {
          return containsWildcardArrayDims(typeAttr.getValue());
        }
        return false;
      });
    }
  }
  if (FunctionType funcTy = llvm::dyn_cast<FunctionType>(type)) {
    return llvm::any_of(funcTy.getInputs(), containsWildcardArrayDims) ||
           llvm::any_of(funcTy.getResults(), containsWildcardArrayDims);
  }
  return false;
}

/// Collects every wildcard array in `oldTy` that becomes concrete in `newTy`.
/// Returns failure if the types do not describe the same overall structure.
static LogicalResult collectWildcardArraySpecializations(
    Type oldTy, Type newTy, WildcardArraySpecializationInfo &out,
    std::optional<StructType> ignoredStructType = std::nullopt
) {
  if (ignoredStructType.has_value() && oldTy == *ignoredStructType &&
      llvm::isa<StructType>(newTy)) {
    return success();
  }
  if (!typesUnify(oldTy, newTy)) {
    return failure();
  }
  if (FunctionType oldFuncTy = llvm::dyn_cast<FunctionType>(oldTy)) {
    FunctionType newFuncTy = llvm::dyn_cast<FunctionType>(newTy);
    if (!newFuncTy || oldFuncTy.getNumInputs() != newFuncTy.getNumInputs() ||
        oldFuncTy.getNumResults() != newFuncTy.getNumResults()) {
      return failure();
    }
    for (auto [oldInput, newInput] :
         llvm::zip_equal(oldFuncTy.getInputs(), newFuncTy.getInputs())) {
      if (failed(collectWildcardArraySpecializations(oldInput, newInput, out, ignoredStructType))) {
        return failure();
      }
    }
    for (auto [oldResult, newResult] :
         llvm::zip_equal(oldFuncTy.getResults(), newFuncTy.getResults())) {
      if (failed(
              collectWildcardArraySpecializations(oldResult, newResult, out, ignoredStructType)
          )) {
        return failure();
      }
    }
    return success();
  }
  if (StructType oldStructTy = llvm::dyn_cast<StructType>(oldTy)) {
    StructType newStructTy = llvm::dyn_cast<StructType>(newTy);
    if (!newStructTy) {
      return failure();
    }
    ArrayAttr oldParams = oldStructTy.getParams();
    ArrayAttr newParams = newStructTy.getParams();
    ArrayRef<Attribute> oldAttrs = oldParams ? oldParams.getValue() : ArrayRef<Attribute> {};
    ArrayRef<Attribute> newAttrs = newParams ? newParams.getValue() : ArrayRef<Attribute> {};
    if (oldAttrs.size() != newAttrs.size()) {
      return failure();
    }
    for (auto [oldAttr, newAttr] : llvm::zip_equal(oldAttrs, newAttrs)) {
      if (TypeAttr oldTypeAttr = llvm::dyn_cast<TypeAttr>(oldAttr)) {
        TypeAttr newTypeAttr = llvm::dyn_cast<TypeAttr>(newAttr);
        if (!newTypeAttr ||
            failed(collectWildcardArraySpecializations(
                oldTypeAttr.getValue(), newTypeAttr.getValue(), out, ignoredStructType
            ))) {
          return failure();
        }
      }
    }
    return success();
  }
  ArrayType oldArrTy = llvm::dyn_cast<ArrayType>(oldTy);
  ArrayType newArrTy = llvm::dyn_cast<ArrayType>(newTy);
  if (!oldArrTy || !newArrTy) {
    return success();
  }
  if (oldArrTy.getDimensionSizes().size() != newArrTy.getDimensionSizes().size() ||
      failed(collectWildcardArraySpecializations(
          oldArrTy.getElementType(), newArrTy.getElementType(), out, ignoredStructType
      ))) {
    return failure();
  }

  bool changed = false;
  for (auto [oldDim, newDim] :
       llvm::zip_equal(oldArrTy.getDimensionSizes(), newArrTy.getDimensionSizes())) {
    if (auto oldInt = llvm::dyn_cast<IntegerAttr>(oldDim); oldInt && isDynamic(oldInt)) {
      if (auto newInt = llvm::dyn_cast<IntegerAttr>(newDim); newInt && !isDynamic(newInt)) {
        changed = true;
      }
    }
  }
  if (!changed) {
    return success();
  }
  return out.record(oldArrTy, newArrTy);
}

static bool functionTypeIsMoreConcrete(
    FunctionType oldTy, FunctionType newTy, const ConversionTracker &tracker, const char *patName,
    std::optional<StructType> ignoredStructType = std::nullopt
) {
  auto isCompatible = [&](Type oldType, Type newType) {
    if (ignoredStructType.has_value() && oldType == *ignoredStructType &&
        llvm::isa<StructType>(newType)) {
      return true;
    }
    return tracker.isLegalConversion(oldType, newType, patName);
  };

  return oldTy.getNumInputs() == newTy.getNumInputs() &&
         oldTy.getNumResults() == newTy.getNumResults() &&
         llvm::all_of(llvm::zip_equal(oldTy.getInputs(), newTy.getInputs()), [&](auto pair) {
    return isCompatible(std::get<0>(pair), std::get<1>(pair));
  }) && llvm::all_of(llvm::zip_equal(oldTy.getResults(), newTy.getResults()), [&](auto pair) {
    return isCompatible(std::get<0>(pair), std::get<1>(pair));
  });
}

/// Rewrites wildcard arrays, and optionally one enclosing struct type, to the
/// concrete types inferred for a specialization.
class WildcardArrayTypeConverter : public TypeConverter {
  const DenseMap<ArrayType, ArrayType> &arrayReplacements_;
  std::optional<StructType> oldStructType_;
  std::optional<StructType> newStructType_;

public:
  WildcardArrayTypeConverter(
      const DenseMap<ArrayType, ArrayType> &arrayReplacements,
      std::optional<StructType> oldStructType = std::nullopt,
      std::optional<StructType> newStructType = std::nullopt
  )
      : TypeConverter(), arrayReplacements_(arrayReplacements), oldStructType_(oldStructType),
        newStructType_(newStructType) {
    addConversion([](Type inputTy) { return inputTy; });

    addConversion([this](ArrayType inputTy) -> Type {
      Type newElemTy = this->convertType(inputTy.getElementType());
      auto it = arrayReplacements_.find(inputTy);
      if (it != arrayReplacements_.end()) {
        ArrayType replacement = it->second;
        if (replacement.getElementType() != newElemTy) {
          return replacement.cloneWith(newElemTy);
        }
        return replacement;
      }
      if (newElemTy != inputTy.getElementType()) {
        return inputTy.cloneWith(newElemTy);
      }
      return inputTy;
    });

    addConversion([this](StructType inputTy) -> Type {
      if (oldStructType_.has_value() && newStructType_.has_value() && inputTy == *oldStructType_) {
        return *newStructType_;
      }
      if (ArrayAttr params = inputTy.getParams()) {
        SmallVector<Attribute> updated;
        bool changed = false;
        for (Attribute attr : params.getValue()) {
          if (TypeAttr typeAttr = llvm::dyn_cast<TypeAttr>(attr)) {
            Type newTy = this->convertType(typeAttr.getValue());
            updated.push_back(TypeAttr::get(newTy));
            changed |= newTy != typeAttr.getValue();
          } else {
            updated.push_back(attr);
          }
        }
        if (changed) {
          return StructType::get(
              inputTy.getNameRef(), ArrayAttr::get(inputTy.getContext(), updated)
          );
        }
      }
      return inputTy;
    });
  }
};

/// Produces a more precise cast result type when the input carries concrete
/// array sizes for dimensions that were wildcarded in the result.
static std::optional<Type> refineCastResultArrayWildcards(Type resultTy, Type inputTy) {
  ArrayType resultArrTy = llvm::dyn_cast<ArrayType>(resultTy);
  ArrayType inputArrTy = llvm::dyn_cast<ArrayType>(inputTy);
  if (!resultArrTy || !inputArrTy) {
    return std::nullopt;
  }
  if (resultArrTy.getDimensionSizes().size() != inputArrTy.getDimensionSizes().size() ||
      !typesUnify(resultArrTy.getElementType(), inputArrTy.getElementType())) {
    return std::nullopt;
  }

  SmallVector<Attribute> refinedDims;
  bool changed = false;
  for (auto [resultDim, inputDim] :
       llvm::zip_equal(resultArrTy.getDimensionSizes(), inputArrTy.getDimensionSizes())) {
    if (auto resultInt = llvm::dyn_cast<IntegerAttr>(resultDim);
        resultInt && isDynamic(resultInt)) {
      if (auto inputInt = llvm::dyn_cast<IntegerAttr>(inputDim); inputInt && !isDynamic(inputInt)) {
        refinedDims.push_back(inputDim);
        changed = true;
        continue;
      }
    }
    refinedDims.push_back(resultDim);
  }

  if (!changed) {
    return std::nullopt;
  }
  return resultArrTy.cloneWith(resultArrTy.getElementType(), refinedDims);
}

/// Template-scoped references name template parameters rather than concrete
/// symbols, so they cannot be specialized here.
static bool calleeReferencesTemplateParam(CallOp op) {
  SymbolRefAttr callee = op.getCalleeAttr();
  if (!callee || callee.getNestedReferences().size() != 1) {
    return false;
  }
  TemplateOp parentTemplate = getParentOfType<TemplateOp>(op);
  if (!parentTemplate) {
    return false;
  }
  return parentTemplate.hasConstNamed<TemplateParamOp>(callee.getRootReference());
}

namespace Cleanup {

/// Shared state for post-specialization cleanup helpers.
class CleanupBase {
public:
  SymbolTableCollection tables;

  CleanupBase(ModuleOp root, const SymbolDefTree &symDefTree, const SymbolUseGraph &symUseGraph)
      : rootMod(root), defTree(symDefTree), useGraph(symUseGraph) {}

protected:
  ModuleOp rootMod;
  const SymbolDefTree &defTree;
  const SymbolUseGraph &useGraph;
};

static bool isErasableDefinition(Operation *op) {
  if (llvm::isa<StructDefOp>(op)) {
    return true;
  }
  if (function::FuncDefOp fdef = llvm::dyn_cast<function::FuncDefOp>(op)) {
    return !fdef.isInStruct();
  }
  return false;
}

/// Removes parameterized definitions whose specialized replacements now cover
/// every remaining use.
struct FromEraseSet : public CleanupBase {
  FromEraseSet(
      ModuleOp root, const SymbolDefTree &symDefTree, const SymbolUseGraph &symUseGraph,
      DenseSet<SymbolRefAttr> &&tryToErasePaths
  )
      : CleanupBase(root, symDefTree, symUseGraph) {
    for (SymbolRefAttr path : tryToErasePaths) {
      Operation *lookupFrom = rootMod.getOperation();
      auto res = lookupSymbolIn(tables, path, Within(), lookupFrom);
      assert(succeeded(res) && "inputs must be valid symbol references");
      assert(isErasableDefinition(res->get()) && "inputs must be cleanup candidates");
      if (!res->viaInclude()) {
        tryToErase.insert(llvm::cast<SymbolOpInterface>(res->get()));
      }
    }
  }

  LogicalResult eraseUnusedDefinitions() {
    for (SymbolOpInterface sym : tryToErase) {
      collectSafeToErase(sym);
    }
    for (auto &it : llvm::make_early_inc_range(visitedPlusSafetyResult)) {
      if (!it.second || !tryToErase.contains(it.first)) {
        visitedPlusSafetyResult.erase(it.first);
      }
    }
    for (auto &[sym, _] : visitedPlusSafetyResult) {
      sym.erase();
    }
    return success();
  }

  const DenseSet<SymbolOpInterface> &getTryToEraseSet() const { return tryToErase; }

private:
  DenseSet<SymbolOpInterface> tryToErase;
  DenseMap<SymbolOpInterface, bool> visitedPlusSafetyResult;
  DenseMap<const SymbolUseGraphNode *, SymbolOpInterface> lookupCache;

  bool collectSafeToErase(SymbolOpInterface check) {
    assert(check);
    auto visited = visitedPlusSafetyResult.find(check);
    if (visited != visitedPlusSafetyResult.end()) {
      return visited->second;
    }
    if (isErasableDefinition(check.getOperation()) && !tryToErase.contains(check)) {
      visitedPlusSafetyResult[check] = false;
      return false;
    }
    visitedPlusSafetyResult[check] = true;
    if (collectSafeToErase(defTree.lookupNode(check))) {
      const auto *useNode = useGraph.lookupNode(check);
      if (!useNode || collectSafeToErase(useNode)) {
        return true;
      }
    }
    visitedPlusSafetyResult[check] = false;
    return false;
  }

  bool collectSafeToErase(const SymbolDefTreeNode *check) {
    assert(check);
    if (const SymbolDefTreeNode *p = check->getParent()) {
      if (SymbolOpInterface checkOp = p->getOp()) {
        return collectSafeToErase(checkOp);
      }
    }
    return true;
  }

  bool collectSafeToErase(const SymbolUseGraphNode *check) {
    assert(check);
    for (const SymbolUseGraphNode *p : check->predecessorIter()) {
      if (SymbolOpInterface checkOp = cachedLookup(p)) {
        if (!collectSafeToErase(checkOp)) {
          return false;
        }
      }
    }
    return true;
  }

  SymbolOpInterface cachedLookup(const SymbolUseGraphNode *node) {
    assert(node && "must provide a node");
    auto fromCache = lookupCache.find(node);
    if (fromCache != lookupCache.end()) {
      return fromCache->second;
    }
    auto lookupRes = node->lookupSymbol(tables);
    assert(succeeded(lookupRes) && "graph contains node with invalid path");
    assert(lookupRes->get() != nullptr && "lookup must return an Operation");
    SymbolOpInterface actualRes =
        lookupRes->viaInclude() ? nullptr : llvm::cast<SymbolOpInterface>(lookupRes->get());
    lookupCache[node] = actualRes;
    return actualRes;
  }
};

} // namespace Cleanup

static LogicalResult erasePreimageOfInstantiations(
    ModuleOp rootMod, const ConversionTracker &tracker, const SymbolDefTree &symDefTree,
    const SymbolUseGraph &symUseGraph
) {
  Cleanup::FromEraseSet cleaner(
      rootMod, symDefTree, symUseGraph, tracker.getInstantiatedDefinitionNames()
  );
  LogicalResult res = cleaner.eraseUnusedDefinitions();
  if (failed(res)) {
    return res;
  }
  rootMod->walk([&cleaner, &symUseGraph](Operation *walkedOp) {
    SymbolOpInterface op = llvm::dyn_cast<SymbolOpInterface>(walkedOp);
    if (!op || !cleaner.getTryToEraseSet().contains(op)) {
      return;
    }
    if (const SymbolUseGraphNode *node = symUseGraph.lookupNode(op);
        node && node->hasPredecessor()) {
      op.emitWarning("Parameterized definition still has uses!").report();
    }
  });
  return success();
}

namespace CastRefinement {

/// Refines `poly.unifiable_cast` result types by replacing wildcard array
/// dimensions with concrete dimensions inferred from the operand type.
class UpdateUnifiableCastResultType final : public OpRewritePattern<UnifiableCastOp> {
  ConversionTracker &tracker_;

public:
  UpdateUnifiableCastResultType(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  LogicalResult matchAndRewrite(UnifiableCastOp op, PatternRewriter &rewriter) const override {
    std::optional<Type> refinedResultTy =
        refineCastResultArrayWildcards(op.getResult().getType(), op.getInput().getType());
    if (!refinedResultTy.has_value() || *refinedResultTy == op.getResult().getType()) {
      return failure();
    }
    if (!tracker_.isLegalConversion(
            op.getResult().getType(), *refinedResultTy, "UpdateUnifiableCastResultType"
        )) {
      return failure();
    }
    rewriter.modifyOpInPlace(op, [&]() { op.getResult().setType(*refinedResultTy); });
    return success();
  }
};

LogicalResult run(ModuleOp modOp, ConversionTracker &tracker) {
  MLIRContext *ctx = modOp.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<UpdateUnifiableCastResultType>(ctx, tracker);
  return applyAndFoldGreedily(modOp, tracker, std::move(patterns));
}

} // namespace CastRefinement

namespace WildcardFunctionSpecialization {

static SymbolRefAttr replaceLeafReference(SymbolRefAttr symRef, StringRef newLeafName) {
  SmallVector<FlatSymbolRefAttr> pieces = getPieces(symRef);
  assert(!pieces.empty() && "symbol reference must have at least one piece");
  pieces.back() = FlatSymbolRefAttr::get(StringAttr::get(symRef.getContext(), newLeafName));
  return asSymbolRefAttr(pieces);
}

static std::string
buildWildcardSpecializationName(StringRef baseName, const WildcardArraySpecializationInfo &info) {
  return BuildShortTypeString::from(baseName.str(), info.getConcreteTypeAttrs());
}

/// Retargets calls nested inside a specialized struct body to the corresponding
/// specialized struct member definition.
class CallStructFuncPattern : public OpConversionPattern<CallOp> {
public:
  CallStructFuncPattern(TypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<CallOp>(converter, ctx, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(
      CallOp op, OpAdaptor adapter, ConversionPatternRewriter &rewriter
  ) const override {
    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), newResultTypes))) {
      return op->emitError("Could not convert Op result types.");
    }

    SymbolRefAttr calleeAttr = op.getCalleeAttr();
    if (op.calleeIsStructCompute()) {
      if (StructType newStTy = getIfSingleton<StructType>(newResultTypes)) {
        calleeAttr = appendLeaf(newStTy.getNameRef(), calleeAttr.getLeafReference());
      }
    } else if (op.calleeIsStructConstrain()) {
      if (StructType newStTy = getAtIndex<StructType>(adapter.getArgOperands().getTypes(), 0)) {
        calleeAttr = appendLeaf(newStTy.getNameRef(), calleeAttr.getLeafReference());
      }
    }

    replaceOpWithNewOp<CallOp>(
        rewriter, op, newResultTypes, calleeAttr, adapter.getMapOperands(),
        op.getNumDimsPerMapAttr(), adapter.getArgOperands()
    );
    return success();
  }
};

/// Updates struct member declarations after the surrounding struct has been
/// specialized to concrete wildcard array types.
class MemberDefOpPattern : public OpConversionPattern<MemberDefOp> {
public:
  MemberDefOpPattern(TypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<MemberDefOp>(converter, ctx, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(MemberDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    Type oldMemberType = op.getType();
    Type newMemberType = getTypeConverter()->convertType(oldMemberType);
    if (oldMemberType == newMemberType) {
      return failure();
    }
    rewriter.modifyOpInPlace(op, [&op, &newMemberType]() { op.setType(newMemberType); });
    return success();
  }
};

static LogicalResult verifyNestedCallSymbols(FuncDefOp func) {
  SymbolTableCollection tables;
  WalkResult result = func.walk([&tables](CallOp nestedCall) {
    return WalkResult(nestedCall.verifySymbolUses(tables));
  });
  return failure(result.wasInterrupted());
}

static LogicalResult applyWildcardSpecializationConversions(
    FuncDefOp newFunc, const WildcardArraySpecializationInfo &info
) {
  MLIRContext *ctx = newFunc.getContext();
  WildcardArrayTypeConverter tyConv(info.replacements);
  ConversionTarget target = newConverterDefinedTarget<>(tyConv, ctx);
  RewritePatternSet patterns = newGeneralRewritePatternSet(tyConv, ctx, target);
  if (failed(applyFullConversion(newFunc, target, std::move(patterns)))) {
    return failure();
  }
  return verifyNestedCallSymbols(newFunc);
}

static LogicalResult applyWildcardSpecializationConversions(
    StructDefOp newStruct, StructType oldStructType, StructType newStructType,
    const WildcardArraySpecializationInfo &info
) {
  MLIRContext *ctx = newStruct.getContext();
  WildcardArrayTypeConverter tyConv(info.replacements, oldStructType, newStructType);
  ConversionTarget target = newConverterDefinedTarget<>(tyConv, ctx);
  RewritePatternSet patterns = newGeneralRewritePatternSet(tyConv, ctx, target);
  patterns.add<CallStructFuncPattern, MemberDefOpPattern>(tyConv, ctx);
  return applyFullConversion(newStruct, target, std::move(patterns));
}

static FailureOr<SymbolRefAttr> getOrCreateSpecializedFreeFunc(
    CallOp op, PatternRewriter &rewriter, SymbolTableCollection &symTables, FuncDefOp targetFunc,
    const WildcardArraySpecializationInfo &info
) {
  ModuleOp parentModule = getParentOfType<ModuleOp>(targetFunc);
  assert(parentModule && "free function must be nested in a module");

  std::string newFuncName = buildWildcardSpecializationName(targetFunc.getSymName(), info);
  FuncDefOp newFunc;
  if (Operation *existing = symTables.getSymbolTable(parentModule).lookup(newFuncName)) {
    newFunc = llvm::dyn_cast<FuncDefOp>(existing);
    if (!newFunc) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag.append("specialized function name collision for '", newFuncName, '\'');
      });
    }
  } else {
    newFunc = targetFunc.clone();
    newFunc.setSymName(newFuncName);
    symTables.getSymbolTable(parentModule).insert(newFunc, Block::iterator(targetFunc));
    if (failed(applyWildcardSpecializationConversions(newFunc, info))) {
      newFunc.erase();
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag.append("failure while creating wildcard-specialized function '", newFuncName, '\'');
      });
    }
  }

  return replaceLeafReference(op.getCalleeAttr(), newFunc.getSymName());
}

static FailureOr<StructType> getOrCreateSpecializedStruct(
    CallOp op, PatternRewriter &rewriter, SymbolTableCollection &symTables,
    StructDefOp targetStruct, const WildcardArraySpecializationInfo &info,
    ConversionTracker &tracker
) {
  ModuleOp parentModule = getParentOfType<ModuleOp>(targetStruct);
  assert(parentModule && "struct definition must be nested in a module");

  StructType oldStructType = targetStruct.getType();
  std::string newStructName = buildWildcardSpecializationName(targetStruct.getSymName(), info);
  StructDefOp newStruct;
  if (Operation *existing = symTables.getSymbolTable(parentModule).lookup(newStructName)) {
    newStruct = llvm::dyn_cast<StructDefOp>(existing);
    if (!newStruct) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag.append("specialized struct name collision for '", newStructName, '\'');
      });
    }
  } else {
    newStruct = targetStruct.clone();
    newStruct.setSymName(newStructName);
    symTables.getSymbolTable(parentModule).insert(newStruct, Block::iterator(targetStruct));
    StructType newStructType = newStruct.getType();
    if (failed(
            applyWildcardSpecializationConversions(newStruct, oldStructType, newStructType, info)
        )) {
      newStruct.erase();
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag.append("failure while creating wildcard-specialized struct '", newStructName, '\'');
      });
    }
  }

  StructType newStructType = newStruct.getType();
  tracker.recordSpecialization(oldStructType, newStructType);
  return newStructType;
}

/// Specializes calls whose target signature still contains wildcard arrays but
/// whose call-site signature has become concrete enough to resolve them.
class SpecializeWildcardCallOp final : public OpRewritePattern<CallOp> {
  ConversionTracker &tracker_;

public:
  SpecializeWildcardCallOp(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern<CallOp>(ctx), tracker_(tracker) {}

  LogicalResult matchAndRewrite(CallOp op, PatternRewriter &rewriter) const override {
    if (calleeReferencesTemplateParam(op)) {
      return failure();
    }

    SymbolTableCollection symTables;
    FailureOr<SymbolLookupResult<FuncDefOp>> targetRes = op.getCalleeTarget(symTables);
    if (failed(targetRes)) {
      return failure();
    }
    FuncDefOp targetFunc = targetRes->get();
    if (llvm::isa<TemplateOp>(targetFunc->getParentOp())) {
      return failure();
    }

    FunctionType targetSig = targetFunc.getFunctionType();
    FunctionType callSig = op.getTypeSignature();
    StructDefOp targetStruct =
        targetFunc.isInStruct() ? getParentOfType<StructDefOp>(targetFunc) : StructDefOp();
    std::optional<StructType> ignoredStructType =
        targetStruct ? std::optional<StructType>(targetStruct.getType()) : std::nullopt;
    if (!containsWildcardArrayDims(targetSig) ||
        !functionTypeIsMoreConcrete(
            targetSig, callSig, tracker_, "SpecializeWildcardCallOp", ignoredStructType
        )) {
      return failure();
    }

    WildcardArraySpecializationInfo info;
    if (failed(collectWildcardArraySpecializations(targetSig, callSig, info, ignoredStructType)) ||
        info.empty()) {
      return failure();
    }

    if (!targetFunc.isInStruct()) {
      FailureOr<SymbolRefAttr> newCalleeAttr =
          getOrCreateSpecializedFreeFunc(op, rewriter, symTables, targetFunc, info);
      if (failed(newCalleeAttr)) {
        return failure();
      }
      SmallVector<Type> newResultTypes;
      FailureOr<SymbolLookupResult<FuncDefOp>> specializedFuncRes =
          lookupTopLevelSymbol<FuncDefOp>(symTables, *newCalleeAttr, op);
      if (failed(specializedFuncRes)) {
        return failure();
      }
      newResultTypes.append(
          specializedFuncRes->get().getFunctionType().getResults().begin(),
          specializedFuncRes->get().getFunctionType().getResults().end()
      );
      if (!tracker_.areLegalConversions(
              op.getResultTypes(), newResultTypes, "SpecializeWildcardCallOp"
          )) {
        return failure();
      }
      tracker_.recordInstantiation(op.getCalleeAttr());
      tracker_.updateModifiedFlag(true);
      replaceOpWithNewOp<CallOp>(
          rewriter, op, TypeRange(newResultTypes), *newCalleeAttr,
          CallOp::toVectorOfValueRange(op.getMapOperands()), op.getNumDimsPerMapAttr(),
          op.getArgOperands()
      );
      return success();
    }

    assert(targetStruct && "struct function must have a parent struct");
    if (llvm::isa<TemplateOp>(targetStruct->getParentOp())) {
      return failure();
    }

    FailureOr<StructType> newStructTypeRes =
        getOrCreateSpecializedStruct(op, rewriter, symTables, targetStruct, info, tracker_);
    if (failed(newStructTypeRes)) {
      return failure();
    }
    StructType newStructType = *newStructTypeRes;

    SmallVector<Type> newResultTypes;
    if (targetFunc.nameIsConstrain()) {
      if (op.getArgOperands().empty() || op.getArgOperands().front().getType() != newStructType) {
        return failure();
      }
    } else {
      WildcardArrayTypeConverter tyConv(info.replacements, targetStruct.getType(), newStructType);
      if (failed(tyConv.convertTypes(op.getResultTypes(), newResultTypes))) {
        return failure();
      }
      if (!tracker_.areLegalConversions(
              op.getResultTypes(), newResultTypes, "SpecializeWildcardCallOp"
          )) {
        return failure();
      }
    }

    tracker_.updateModifiedFlag(true);
    SymbolRefAttr newCalleeAttr =
        appendLeaf(newStructType.getNameRef(), op.getCallee().getLeafReference());
    replaceOpWithNewOp<CallOp>(
        rewriter, op, TypeRange(newResultTypes), newCalleeAttr,
        CallOp::toVectorOfValueRange(op.getMapOperands()), op.getNumDimsPerMapAttr(),
        op.getArgOperands()
    );
    return success();
  }
};

/// Rebinds `constrain` calls onto a previously-created specialized struct when
/// the receiver type already points at that specialized definition.
class RetargetStructConstrainCall final : public OpRewritePattern<CallOp> {
  ConversionTracker &tracker_;

public:
  RetargetStructConstrainCall(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern<CallOp>(ctx), tracker_(tracker) {}

  LogicalResult matchAndRewrite(CallOp op, PatternRewriter &rewriter) const override {
    if (!op.calleeIsStructConstrain() || op.getArgOperands().empty()) {
      return failure();
    }

    StructType selfType = llvm::dyn_cast<StructType>(op.getArgOperands().front().getType());
    if (!selfType) {
      return failure();
    }

    SymbolTableCollection symTables;
    FailureOr<SymbolLookupResult<FuncDefOp>> targetRes = op.getCalleeTarget(symTables);
    if (failed(targetRes)) {
      return failure();
    }
    FuncDefOp targetFunc = targetRes->get();
    StructDefOp targetStruct = getParentOfType<StructDefOp>(targetFunc);
    if (!targetStruct || selfType == targetStruct.getType()) {
      return failure();
    }

    SymbolRefAttr newCalleeAttr =
        appendLeaf(selfType.getNameRef(), op.getCallee().getLeafReference());
    FailureOr<SymbolLookupResult<FuncDefOp>> specializedRes =
        lookupTopLevelSymbol<FuncDefOp>(symTables, newCalleeAttr, op);
    if (failed(specializedRes)) {
      return failure();
    }

    tracker_.updateModifiedFlag(true);
    replaceOpWithNewOp<CallOp>(
        rewriter, op, TypeRange(op.getResultTypes()), newCalleeAttr,
        CallOp::toVectorOfValueRange(op.getMapOperands()), op.getNumDimsPerMapAttr(),
        op.getArgOperands()
    );
    return success();
  }
};

LogicalResult run(ModuleOp modOp, ConversionTracker &tracker) {
  MLIRContext *ctx = modOp.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<RetargetStructConstrainCall>(ctx, tracker);
  patterns.add<SpecializeWildcardCallOp>(ctx, tracker);
  MatchFailureListener failureListener;
  walkAndApplyPatterns(modOp, std::move(patterns), &failureListener);
  return failure(failureListener.hadFailure);
}

} // namespace WildcardFunctionSpecialization

/// Drives wildcard-array cast refinement and callable specialization until the
/// module reaches a fixpoint, then cleans up replaced parameterized symbols.
class PassImpl : public llzk::polymorphic::impl::WildcardArraySpecializationPassBase<PassImpl> {
public:
  using Base = WildcardArraySpecializationPassBase<PassImpl>;
  using Base::Base;

private:
  void runOnOperation() override {
    ModuleOp modOp = getOperation();
    ConversionTracker tracker;
    unsigned loopCount = 0;
    do {
      ++loopCount;
      if (loopCount > iterationLimit) {
        llvm::errs() << DEBUG_TYPE << " exceeded the limit of " << iterationLimit
                     << " iterations!\n";
        signalPassFailure();
        return;
      }
      tracker.resetModifiedFlag();

      if (failed(CastRefinement::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while refining wildcard array cast results\n";
        signalPassFailure();
        return;
      }
      if (failed(WildcardFunctionSpecialization::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE
                     << " failed while specializing wildcard-array function signatures\n";
        signalPassFailure();
        return;
      }
    } while (tracker.isModified());

    if (failed(erasePreimageOfInstantiations(
            modOp, tracker, getAnalysis<SymbolDefTree>(), getAnalysis<SymbolUseGraph>()
        ))) {
      signalPassFailure();
    }
  }
};

} // namespace
