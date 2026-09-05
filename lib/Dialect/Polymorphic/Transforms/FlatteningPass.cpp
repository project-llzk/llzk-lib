//===-- LLZKFlatteningPass.cpp - Implements -llzk-flatten pass --*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-flatten` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/SymbolDefTree.h"
#include "llzk/Analysis/SymbolUseGraph.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Dialect/String/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/Concepts.h"
#include "llzk/Util/Debug.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolLookup.h"
#include "llzk/Util/SymbolTableLLZK.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/LoopUtils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Utils/Utils.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DepthFirstIterator.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>

#include <cstdint>

// Include the generated base pass class definitions.
namespace llzk::polymorphic {
#define GEN_PASS_DEF_FLATTENINGPASS
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h.inc"
} // namespace llzk::polymorphic

#include "SharedImpl.h"

#define DEBUG_TYPE "llzk-flatten"

using namespace mlir;
using namespace llzk;
using namespace llzk::array;
using namespace llzk::component;
using namespace llzk::constrain;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::polymorphic;
using namespace llzk::polymorphic::detail;

namespace {

static void reportDelayedDiagnostics(CallOp caller, SmallVector<Diagnostic> &&diagnostics) {
  DiagnosticEngine &engine = caller.getContext()->getDiagEngine();
  for (Diagnostic &diag : diagnostics) {
    // Update any notes referencing an UnknownLoc to use the CallOp location.
    for (Diagnostic &note : diag.getNotes()) {
      assert(note.getNotes().empty() && "notes cannot have notes attached");
      if (llvm::isa<UnknownLoc>(note.getLocation())) {
        note = std::move(Diagnostic(caller.getLoc(), note.getSeverity()).append(note.str()));
      }
    }
    // Report. Based on InFlightDiagnostic::report().
    engine.emit(std::move(diag));
  }
}

class ConversionTracker {
  /// Published result of one successful partial-function conversion.
  ///
  /// The source operation and concrete key live in the surrounding map; these names are only the
  /// post-insertion symbol path needed to retarget a later exact cache hit.
  struct PartialFuncInstantiation {
    ArrayAttr concreteParamKey;
    StringAttr templateName;
    StringAttr functionName;
  };

  /// Tracks if some step performed a modification of the code such that another pass should be run.
  bool modified;
  /// Maps original remote (i.e., use site) type to new remote type. A partially specialized result
  /// retains its remaining parameters.
  DenseMap<StructType, StructType> structInstantiations;
  /// Maps each instantiated type to its canonical source type for legal conversion checks. The
  /// absent and explicitly empty parameter-list spellings share a forward-cache entry; only the
  /// canonical spelling is stored in this reverse map.
  DenseMap<StructType, StructType> reverseInstantiations;
  /// Tracks original free function definitions for which instantiated clones were created.
  DenseSet<SymbolRefAttr> funcInstantiations;
  /// Successful partial functions keyed by their source operation and exact concrete bindings.
  /// The rendered symbol names are only values; they are never used as cache identity.
  /// The tracker outlives each Step 2 rewrite pass so exact hits remain reusable across fixpoint
  /// iterations; the cache is cleared only before cleanup can erase its source operations.
  DenseMap<Operation *, SmallVector<PartialFuncInstantiation>> partialFuncInstantiations;
  /// Maps new remote type (i.e., the values in 'structInstantiations') to a list of Diagnostic
  /// to report at the location(s) of the compute() that causes the instantiation to the StructType.
  DenseMap<StructType, SmallVector<Diagnostic>> delayedDiagnostics;
  /// Root-relative names of original structs in templates with `poly.expr` bindings and no
  /// `poly.param` operations. This coarse filter avoids symbol-table lookup for ordinary concrete
  /// types; operation identity below remains authoritative when names can repeat across roots.
  DenseSet<SymbolRefAttr> originalStructNamesWithExprsAndNoParams;
  DenseSet<Operation *> originalStructsWithExprsAndNoParams;

public:
  /// Index eligible source definitions before rewriting begins. Generated templates must not
  /// become eligible merely because an earlier specialization removed their parameters.
  explicit ConversionTracker(ModuleOp root) : modified(false) {
    root.walk([this](TemplateOp templateOp) {
      if (templateOp.hasConstOps<TemplateParamOp>() || !templateOp.hasConstOps<TemplateExprOp>()) {
        return;
      }
      for (StructDefOp structDef : templateOp.getBodyRegion().front().getOps<StructDefOp>()) {
        originalStructNamesWithExprsAndNoParams.insert(structDef.getType().getNameRef());
        originalStructsWithExprsAndNoParams.insert(structDef.getOperation());
      }
    });
  }

  bool isModified() const { return modified; }
  void resetModifiedFlag() { modified = false; }
  void updateModifiedFlag(bool currStepModified) { modified |= currStepModified; }

  /// Return whether `type` may name an eligible original definition.
  bool mayBeOriginalStructWithExprsAndNoParams(StructType type) const {
    return originalStructNamesWithExprsAndNoParams.contains(type.getNameRef());
  }

  /// Return whether `structDef` is an eligible original definition.
  bool isOriginalStructWithExprsAndNoParams(StructDefOp structDef) const {
    return originalStructsWithExprsAndNoParams.contains(structDef.getOperation());
  }

  void recordInstantiation(StructType oldType, StructType newType) {
    StructType canonicalOldType =
        getStructTypeWithParams(oldType.getNameRef(), oldType.getParams());
    auto forwardResult = structInstantiations.try_emplace(canonicalOldType, newType);
    if (forwardResult.second) {
      // Keep the empty-list spelling out of the reverse identity, since it is only a forward alias.
      reverseInstantiations.try_emplace(newType, canonicalOldType);
      modified = true;
    } else {
      // ASSERT: If a mapping already existed for `canonicalOldType` it must be `newType`.
      assert(forwardResult.first->getSecond() == newType);
    }

    // StructType distinguishes a null parameter attribute from an explicit empty ArrayAttr even
    // though both spell the same zero-parameter source. Preserve both forward spellings so uses
    // written either way find the specialization created from the other spelling.
    if (oldType != canonicalOldType) {
      auto aliasResult = structInstantiations.try_emplace(oldType, newType);
      if (aliasResult.second) {
        modified = true;
      } else {
        // ASSERT: If the alternate parameter-list spelling already existed it must map to
        // `newType`.
        assert(aliasResult.first->getSecond() == newType);
      }
    }

    assert(reverseInstantiations.contains(newType));
    assert(structInstantiations.size() >= reverseInstantiations.size());
  }

  /// Return the instantiated type of the given StructType, if any.
  std::optional<StructType> getInstantiation(StructType oldType) const {
    auto cachedResult = structInstantiations.find(oldType);
    if (cachedResult != structInstantiations.end()) {
      return cachedResult->second;
    }
    StructType canonicalOldType =
        getStructTypeWithParams(oldType.getNameRef(), oldType.getParams());
    if (canonicalOldType != oldType) {
      cachedResult = structInstantiations.find(canonicalOldType);
      if (cachedResult != structInstantiations.end()) {
        return cachedResult->second;
      }
    }
    return std::nullopt;
  }

  /// Record that the given free function was instantiated.
  void recordInstantiation(SymbolRefAttr funcName) {
    funcInstantiations.insert(funcName);
    modified = true;
  }

  /// Return the successfully converted partial function for this exact source/key pair, if any.
  std::optional<SymbolRefAttr>
  lookupPartialFuncInstantiation(FuncDefOp sourceFunc, ArrayAttr concreteParamKey) const {
    auto found = partialFuncInstantiations.find(sourceFunc.getOperation());
    if (found == partialFuncInstantiations.end()) {
      return std::nullopt;
    }
    for (const PartialFuncInstantiation &candidate : found->second) {
      if (candidate.concreteParamKey == concreteParamKey) {
        SmallVector<FlatSymbolRefAttr> calleeSuffix {
            FlatSymbolRefAttr::get(candidate.templateName),
            FlatSymbolRefAttr::get(candidate.functionName),
        };
        return asSymbolRefAttr(calleeSuffix);
      }
    }
    return std::nullopt;
  }

  /// Publish a successful partial conversion after insertion and body conversion have completed.
  void recordPartialFuncInstantiation(
      FuncDefOp sourceFunc, ArrayAttr concreteParamKey, TemplateOp templateOp, FuncDefOp functionOp
  ) {
    assert(
        !lookupPartialFuncInstantiation(sourceFunc, concreteParamKey).has_value() &&
        "partial function instantiation already cached"
    );
    partialFuncInstantiations[sourceFunc.getOperation()].push_back(
        PartialFuncInstantiation {
            concreteParamKey,
            templateOp.getSymNameAttr(),
            functionOp.getSymNameAttr(),
        }
    );
  }

  /// No partial-function cache entry is read after cleanup starts. Clear source-operation keys
  /// before cleanup can erase their definitions.
  void clearPartialFuncInstantiations() { partialFuncInstantiations.clear(); }

  /// Collect the fully-qualified names of all structs and free functions that were instantiated.
  DenseSet<SymbolRefAttr> getInstantiatedDefinitionNames() const {
    DenseSet<SymbolRefAttr> instantiatedNames = funcInstantiations;
    for (const auto &[origRemoteTy, _] : structInstantiations) {
      instantiatedNames.insert(origRemoteTy.getNameRef());
    }
    return instantiatedNames;
  }

  void reportDelayedDiagnostics(StructType newType, CallOp caller) {
    auto res = delayedDiagnostics.find(newType);
    if (res != delayedDiagnostics.end()) {
      ::reportDelayedDiagnostics(caller, std::move(res->second));

      // Emitting a Diagnostic consumes it (per DiagnosticEngine::emit) so remove them from the map.
      // Unfortunately, this means if the key StructType is the result of instantiation at multiple
      // `compute()` calls it will only be reported at one of those locations, not all.
      delayedDiagnostics.erase(newType);
    }
  }

  SmallVector<Diagnostic> &delayedDiagnosticSet(StructType newType) {
    return delayedDiagnostics[newType];
  }

  /// Check if the type conversion is legal, i.e., the new type unifies with and is more concrete
  /// than the old type with additional allowance for the results of struct flattening conversions.
  bool isLegalConversion(Type oldType, Type newType, const char *patName) const {
    std::function<bool(Type, Type)> checkInstantiations = [&](Type oTy, Type nTy) {
      // Check if `oTy` is a struct with a known instantiation to `nTy`
      if (StructType oldStructType = llvm::dyn_cast<StructType>(oTy)) {
        // The map records the exact result type, including any parameters retained by a partial
        // specialization, so direct equality is sufficient.
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
        llvm::dbgs() << "[" << patName << "] Cannot replace old type " << oldType
                     << " with new type " << newType
                     << " because it does not define a compatible and more concrete type.\n";
    );
    return false;
  }

  template <typename T, typename U>
  inline bool areLegalConversions(T oldTypes, U newTypes, const char *patName) const {
    return llvm::all_of(
        llvm::zip_equal(oldTypes, newTypes), [this, &patName](std::tuple<Type, Type> oldThenNew) {
      return this->isLegalConversion(std::get<0>(oldThenNew), std::get<1>(oldThenNew), patName);
    }
    );
  }
};

template <typename Impl, typename Op, typename... HandledAttrs>
class SymbolUserHelper : public OpConversionPattern<Op> {
private:
  const DenseMap<Attribute, Attribute> &paramNameToValue;

  SymbolUserHelper(
      TypeConverter &converter, MLIRContext *ctx, unsigned patternBenefit,
      const DenseMap<Attribute, Attribute> &paramNameToInstantiatedValue
  )
      : OpConversionPattern<Op>(converter, ctx, patternBenefit),
        paramNameToValue(paramNameToInstantiatedValue) {}

public:
  using OpAdaptor = typename mlir::OpConversionPattern<Op>::OpAdaptor;

  virtual Attribute getNameAttr(Op) const = 0;

  virtual LogicalResult handleDefaultRewrite(
      Attribute, Op op, OpAdaptor, ConversionPatternRewriter &, Attribute a
  ) const {
    return op->emitOpError().append("expected value with type ", op.getType(), " but found ", a);
  }

  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "[SymbolUserHelper] op: " << op << '\n');
    auto res = this->paramNameToValue.find(getNameAttr(op));
    if (res == this->paramNameToValue.end()) {
      LLVM_DEBUG(llvm::dbgs() << "[SymbolUserHelper] no instantiation for " << op << '\n');
      return failure();
    }
    llvm::TypeSwitch<Attribute, LogicalResult> TS(res->second);
    llvm::TypeSwitch<Attribute, LogicalResult> *ptr = &TS;

    ((ptr = &(ptr->template Case<HandledAttrs>([&](HandledAttrs a) {
      return static_cast<const Impl *>(this)->handleRewrite(res->first, op, adaptor, rewriter, a);
    }))),
     ...);

    return TS.Default([&](Attribute a) {
      return handleDefaultRewrite(res->first, op, adaptor, rewriter, a);
    });
  }
  friend Impl;
};

class ClonedBodyConstReadOpPattern
    : public SymbolUserHelper<
          ClonedBodyConstReadOpPattern, ConstReadOp, IntegerAttr, FeltConstAttr> {
  SmallVector<Diagnostic> &diagnostics;

  using super =
      SymbolUserHelper<ClonedBodyConstReadOpPattern, ConstReadOp, IntegerAttr, FeltConstAttr>;

public:
  ClonedBodyConstReadOpPattern(
      TypeConverter &converter, MLIRContext *ctx,
      const DenseMap<Attribute, Attribute> &paramNameToInstantiatedValue,
      SmallVector<Diagnostic> &instantiationDiagnostics
  )
      // benefit>0 so this applies instead of GeneralTypeReplacePattern<ConstReadOp>
      : super(converter, ctx, /*patternBenefit=*/1, paramNameToInstantiatedValue),
        diagnostics(instantiationDiagnostics) {}

  Attribute getNameAttr(ConstReadOp op) const override { return op.getConstNameAttr(); }

  LogicalResult handleRewrite(
      Attribute sym, ConstReadOp op, OpAdaptor, ConversionPatternRewriter &rewriter, IntegerAttr a
  ) const {
    APInt attrValue = a.getValue();
    Type origResTy = op.getType();
    Type newResTy = getTypeConverter()->convertType(origResTy);
    if (!newResTy) {
      return op->emitOpError().append("could not convert result type ", origResTy);
    }

    if (FeltType ty = llvm::dyn_cast<FeltType>(newResTy)) {
      replaceOpWithNewOp<FeltConstantOp>(
          rewriter, op, FeltConstAttr::get(getContext(), attrValue, ty)
      );
      return success();
    }

    if (llvm::isa<IndexType>(newResTy)) {
      replaceOpWithNewOp<arith::ConstantIndexOp>(rewriter, op, fromAPInt(attrValue));
      return success();
    }

    if (newResTy.isSignlessInteger(1)) {
      // Treat 0 as false and any other value as true (but give a warning if it's not 1)
      if (attrValue.isZero()) {
        replaceOpWithNewOp<arith::ConstantIntOp>(rewriter, op, false, newResTy);
        return success();
      }
      if (!attrValue.isOne()) {
        Location opLoc = op.getLoc();
        Diagnostic diag(opLoc, DiagnosticSeverity::Warning);
        diag << "Interpreting non-zero value " << stringWithoutType(a) << " as true";
        if (getContext()->shouldPrintOpOnDiagnostic()) {
          diag.attachNote(opLoc) << "see current operation: " << *op;
        }
        diag.attachNote(UnknownLoc::get(getContext()))
            << "when instantiating '" << StructDefOp::getOperationName() << "' parameter \"" << sym
            << "\" for this call";
        diagnostics.push_back(std::move(diag));
      }
      replaceOpWithNewOp<arith::ConstantIntOp>(rewriter, op, true, newResTy);
      return success();
    }
    return op->emitOpError().append("unexpected result type ", newResTy);
  }

  LogicalResult handleRewrite(
      Attribute, ConstReadOp op, OpAdaptor, ConversionPatternRewriter &rewriter, FeltConstAttr a
  ) const {
    Type origResTy = op.getType();
    Type newResTy = getTypeConverter()->convertType(origResTy);
    FeltType feltType = llvm::dyn_cast_or_null<FeltType>(newResTy);
    if (!feltType) {
      return op->emitOpError().append(
          "expected a concrete felt result type after conversion, but found ",
          newResTy ? newResTy : origResTy
      );
    }

    FailureOr<Attribute> materialized =
        materializeTemplateParamValue(a, std::optional<Type>(feltType));
    if (failed(materialized)) {
      return op->emitOpError().append(
          "felt constant ", a, " is incompatible with converted result type ", feltType
      );
    }
    replaceOpWithNewOp<FeltConstantOp>(rewriter, op, llvm::cast<FeltConstAttr>(*materialized));
    return success();
  }
};

/// Apply known template bindings throughout types and type-valued attributes. This converts type
/// variables, array dimensions and element types, parameterized struct arguments, and POD
/// record types. Unbound parameters and otherwise unchanged types retain their original form.
class TemplateParamTypeConverter : public TypeConverter {
  const DenseMap<Attribute, Attribute> &paramNameToValue;

protected:
  Attribute convertIfPossible(Attribute attr) const {
    auto res = paramNameToValue.find(attr);
    return (res != paramNameToValue.end()) ? res->second : attr;
  }

public:
  /// Build a converter that substitutes exactly the known template bindings and recursively
  /// propagates those substitutions through supported compound types.
  explicit TemplateParamTypeConverter(const DenseMap<Attribute, Attribute> &paramNameToConcrete)
      : TypeConverter(), paramNameToValue(paramNameToConcrete) {
    addConversion([](Type type) { return type; });
    addConversion([this](TypeVarType inputTy) -> Type {
      if (TypeAttr tyAttr = llvm::dyn_cast<TypeAttr>(convertIfPossible(inputTy.getNameRef()))) {
        Type convertedType = tyAttr.getValue();
        if (isConcreteType(convertedType)) {
          return convertedType;
        }
      }
      return inputTy;
    });

    addConversion([this](ArrayType inputTy) {
      SmallVector<Attribute> updatedDims;
      bool changed = false;
      for (Attribute dim : inputTy.getDimensionSizes()) {
        Attribute converted = convertIfPossible(dim);
        updatedDims.push_back(converted);
        changed |= converted != dim;
      }
      Type updatedElement = convertType(inputTy.getElementType());
      if (!changed && updatedElement == inputTy.getElementType()) {
        return inputTy;
      }
      return flattenArrayElementType(
          inputTy.cloneWith(inputTy.getElementType(), updatedDims), updatedElement
      );
    });

    addConversion([this](StructType inputTy) -> StructType {
      ArrayAttr params = inputTy.getParams();
      if (!params) {
        return inputTy;
      }
      SmallVector<Attribute> updatedParams;
      bool changed = false;
      for (Attribute param : params) {
        Attribute converted = convertAttr(param);
        updatedParams.push_back(converted);
        changed |= converted != param;
      }
      return changed ? getStructTypeWithParams(
                           inputTy.getNameRef(), inputTy.getContext(), updatedParams
                       )
                     : inputTy;
    });

    addConversion([this](pod::PodType inputTy) -> pod::PodType {
      SmallVector<pod::RecordAttr> updatedRecords;
      bool changed = false;
      for (pod::RecordAttr record : inputTy.getRecords()) {
        Type converted = convertType(record.getType());
        updatedRecords.push_back(
            converted == record.getType()
                ? record
                : pod::RecordAttr::get(inputTy.getContext(), record.getName(), converted)
        );
        changed |= converted != record.getType();
      }
      return changed ? pod::PodType::get(inputTy.getContext(), updatedRecords) : inputTy;
    });
  }

  /// Recursively convert a type-valued attribute; otherwise replace an exact bound parameter.
  Attribute convertAttr(Attribute attr) const {
    if (TypeAttr tyAttr = llvm::dyn_cast<TypeAttr>(attr)) {
      Type convertedTy = convertType(tyAttr.getValue());
      if (convertedTy != tyAttr.getValue()) {
        return TypeAttr::get(convertedTy);
      }
    }
    return convertIfPossible(attr);
  }
};

/// Return whether an operation still has a non-concrete operand or result type after applying the
/// currently known template bindings. Non-read operations may remain in a reduced expression with
/// such types; the evaluator must wait until a later specialization makes them concrete.
static bool hasUnresolvedOperationType(Operation *op, const TypeConverter &tyConv) {
  auto hasUnresolvedType = [&tyConv](Type type) {
    Type converted = tyConv.convertType(type);
    return !converted || !isConcreteType(converted);
  };
  return llvm::any_of(op->getOperandTypes(), hasUnresolvedType) ||
         llvm::any_of(op->getResultTypes(), hasUnresolvedType);
}

/// Clone a referenced template expression and apply every currently concrete value and type
/// binding. A reduced template no longer owns the removed parameter declarations, so its retained
/// expression must contain neither reads nor type variables for those parameters. Return an empty
/// result when a known value still has a non-concrete converted type, deferring that specialization
/// attempt rather than discarding a binding that cannot yet be materialized.
static FailureOr<std::optional<TemplateExprOp>> cloneDeferredExpr(
    TemplateExprOp exprOp, const DenseMap<Attribute, Attribute> &paramNameToConcrete,
    SmallVector<Diagnostic> &diagnostics
) {
  MLIRContext *ctx = exprOp.getContext();
  TemplateParamTypeConverter tyConv(paramNameToConcrete);
  // A known read cannot remain after its binding is removed, so defer the whole clone if its
  // converted result type is still symbolic. Other operations can remain symbolic in a reduced
  // expression; evaluateExpr checks those operation types before attempting a fold.
  WalkResult unresolvedDependency = exprOp.walk([&](ConstReadOp readOp) {
    if (!paramNameToConcrete.contains(readOp.getConstNameAttr())) {
      return WalkResult::advance();
    }
    return hasUnresolvedOperationType(readOp, tyConv) ? WalkResult::interrupt()
                                                      : WalkResult::advance();
  });
  if (unresolvedDependency.wasInterrupted()) {
    return std::optional<TemplateExprOp>();
  }

  TemplateExprOp clonedExpr = llvm::cast<TemplateExprOp>(exprOp->clone());
  ConversionTarget target = newConverterDefinedTarget<>(tyConv, ctx);
  target.addDynamicallyLegalOp<ConstReadOp>([&](ConstReadOp op) {
    return !paramNameToConcrete.contains(op.getConstNameAttr()) && defaultLegalityCheck(tyConv, op);
  });

  RewritePatternSet patterns = newGeneralRewritePatternSet<>(tyConv, ctx, target);
  patterns.add<ClonedBodyConstReadOpPattern>(tyConv, ctx, paramNameToConcrete, diagnostics);
  if (failed(applyFullConversion(clonedExpr, target, std::move(patterns)))) {
    clonedExpr->destroy();
    return failure();
  }
  return std::make_optional(clonedExpr);
}

/// Patterns can use this listener and call notifyMatchFailure(..) for failures where the entire
/// pass must fail, i.e., where instantiation would introduce an illegal type conversion.
struct MatchFailureListener : public RewriterBase::Listener {
  bool hadFailure = false;

  ~MatchFailureListener() override {}

  void notifyMatchFailure(Location loc, function_ref<void(Diagnostic &)> reasonCallback) override {
    hadFailure = true;

    InFlightDiagnostic diag = emitError(loc);
    reasonCallback(*diag.getUnderlyingDiagnostic());
    diag.report();
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

/// Return true if the given attribute value is concrete for the purposes of struct instantiation.
template <bool AllowStructParams = true> bool isConcreteAttr(Attribute a) {
  return classifyAttrConcreteness(a, AllowStructParams) == AttrConcreteness::Concrete;
}

static SymbolRefAttr
convertCalleeSymRefs(SymbolRefAttr callee, const DenseMap<Attribute, Attribute> &paramNameToValue) {
  auto it = paramNameToValue.find(FlatSymbolRefAttr::get(callee.getRootReference()));
  if (it == paramNameToValue.end()) {
    return callee;
  }

  auto tyAttr = llvm::dyn_cast<TypeAttr>(it->second);
  if (!tyAttr) {
    return callee;
  }

  auto structTy = llvm::dyn_cast<StructType>(tyAttr.getValue());
  if (!structTy) {
    return callee;
  }

  SmallVector<FlatSymbolRefAttr> newPieces = getPieces(structTy.getNameRef());
  llvm::append_range(newPieces, callee.getNestedReferences());
  return asSymbolRefAttr(newPieces);
}

/// Rewrite callees in a cloned target using concrete type bindings. Materialize explicit
/// nested-call arguments for removed bindings before the clone enters a reduced template or a
/// parent module.
static void
convertCalleesInPlace(Operation *op, const DenseMap<Attribute, Attribute> &paramNameToValue) {
  TemplateParamTypeConverter tyConv(paramNameToValue);
  op->walk([&paramNameToValue, &tyConv](CallOp callOp) {
    callOp.setCalleeAttr(convertCalleeSymRefs(callOp.getCalleeAttr(), paramNameToValue));

    ArrayAttr templateParams = callOp.getTemplateParamsAttr();
    if (!templateParams) {
      return;
    }
    SmallVector<Attribute> convertedParams;
    convertedParams.reserve(templateParams.size());
    bool changed = false;
    for (Attribute param : templateParams) {
      Attribute converted = tyConv.convertAttr(param);
      convertedParams.push_back(converted);
      changed |= converted != param;
    }
    if (changed) {
      callOp.setTemplateParamsAttr(ArrayAttr::get(callOp.getContext(), convertedParams));
    }
  });
}

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

/// Evaluate a single template expression. An unresolved value or operation-type dependency defers
/// evaluation; malformed, incompatible, effectful, non-speculatable, or non-foldable concrete
/// expressions are semantic errors.
static FailureOr<std::optional<Attribute>>
evaluateExpr(TemplateExprOp exprOp, const DenseMap<Attribute, Attribute> &paramNameToConcrete) {
  TemplateParamTypeConverter tyConv(paramNameToConcrete);
  // Deferral depends on the expression's complete value and type dependency set, not operation
  // order. Do not diagnose a non-foldable prefix while a later read or operation type still
  // requires partial instantiation.
  WalkResult unresolvedParam = exprOp.walk([&](Operation *op) {
    if (auto constReadOp = llvm::dyn_cast<ConstReadOp>(op);
        constReadOp && !paramNameToConcrete.contains(constReadOp.getConstNameAttr())) {
      return WalkResult::interrupt();
    }
    return hasUnresolvedOperationType(op, tyConv) ? WalkResult::interrupt() : WalkResult::advance();
  });
  if (unresolvedParam.wasInterrupted()) {
    return std::optional<Attribute>();
  }

  // Map from SSA value in the expr body to its concrete Attribute.
  DenseMap<Value, Attribute> valueMap;
  for (Operation &bodyOp : exprOp.getInitializerRegion().front()) {
    if (auto yieldOp = llvm::dyn_cast<YieldOp>(bodyOp)) {
      auto it = valueMap.find(yieldOp.getVal());
      if (it != valueMap.end()) {
        return std::make_optional(it->second);
      }
      yieldOp.emitOpError("cannot evaluate yielded value as a concrete template constant");
      return failure();
    }

    if (auto constReadOp = llvm::dyn_cast<ConstReadOp>(bodyOp)) {
      auto it = paramNameToConcrete.find(constReadOp.getConstNameAttr());
      if (it == paramNameToConcrete.end()) {
        return std::optional<Attribute>();
      }
      // If the attribute type is `FeltType` but it's stored as an IntegerAttr, promote to
      // a `FeltConstAttr`.
      Attribute val = it->second;
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(val)) {
        if (auto feltTy = llvm::dyn_cast<FeltType>(constReadOp.getResult().getType())) {
          val = FeltConstAttr::get(bodyOp.getContext(), intAttr.getValue(), feltTy);
        }
      }
      valueMap[constReadOp.getResult()] = val;
      continue;
    }

    // Gather constant attributes for all operands.
    SmallVector<Attribute> operandAttrs;
    operandAttrs.reserve(bodyOp.getNumOperands());
    for (Value operand : bodyOp.getOperands()) {
      auto it = valueMap.find(operand);
      if (it == valueMap.end()) {
        bodyOp.emitOpError("cannot evaluate operand as a concrete template constant");
        return failure();
      }
      operandAttrs.push_back(it->second);
    }

    // A successful zero-result fold can otherwise discard an operation with nested effects.
    // Accept only folds whose operation is safe to speculate and free of memory effects.
    const bool isFoldDiscardable = mlir::isPure(&bodyOp);
    SmallVector<OpFoldResult> foldResults;
    if (failed(bodyOp.fold(operandAttrs, foldResults)) ||
        foldResults.size() != bodyOp.getNumResults()) {
      bodyOp.emitOpError("cannot fold concrete template expression");
      return failure();
    }
    if (!isFoldDiscardable) {
      bodyOp.emitOpError(
          "cannot evaluate concrete template expression with memory effects or unsafe speculation"
      );
      return failure();
    }
    for (auto [result, fr] : llvm::zip_equal(bodyOp.getResults(), foldResults)) {
      if (Attribute a = llvm::dyn_cast<Attribute>(fr)) {
        valueMap[result] = a;
        continue;
      }
      if (Value foldedValue = llvm::dyn_cast<Value>(fr)) {
        auto it = valueMap.find(foldedValue);
        if (it != valueMap.end()) {
          valueMap[result] = it->second;
          continue;
        }
      }
      bodyOp.emitOpError("template expression fold did not produce a constant attribute");
      return failure();
    }
  }
  exprOp.emitOpError("initializer has no yield operation");
  return failure();
}

/// Return whether `target` may reference `exprOp`. Symbol-use analysis stops at symbol-table
/// boundaries, so inspect target regions separately. If analysis cannot prove that `target` has no
/// use, conservatively treat the expression as referenced.
static bool targetMayReferenceTemplateExpr(Operation *target, TemplateExprOp exprOp) {
  // Symbol-use traversal stops at the StructDefOp symbol-table boundary, so inspect its direct
  // member types separately without descending into nested symbol scopes.
  if (auto structDef = llvm::dyn_cast<StructDefOp>(target)) {
    SymbolRefAttr exprRef = FlatSymbolRefAttr::get(exprOp.getSymNameAttr());
    SymbolTableCollection tables;
    for (MemberDefOp memberDef : structDef.getBodyRegion().getOps<MemberDefOp>()) {
      for (SymbolRefAttr usedRef : getSymbolsUsedIn(memberDef.getType())) {
        if (usedRef == exprRef ||
            tables.lookupNearestSymbolFrom(memberDef.getOperation(), usedRef) ==
                exprOp.getOperation()) {
          return true;
        }
      }
    }
  }
  if (!symbolKnownUseEmpty(exprOp.getOperation(), target)) {
    return true;
  }
  return llvm::any_of(target->getRegions(), [&](Region &region) {
    return !symbolKnownUseEmpty(exprOp.getOperation(), &region);
  });
}

/// Evaluate `TemplateExprOp`s referenced by `target` whose dependencies are concrete, adding their
/// values to `paramNameToConcrete`. Skip unreferenced expressions. A failed result is a fatal
/// conversion or evaluation error; a successful empty optional means a known binding's converted
/// type remains non-concrete, so the caller should make no progress after the complete scan. A
/// successful value contains detached, converted clones for expressions that still depend on
/// remaining parameters, which the caller must insert or destroy. Any concrete but malformed or
/// non-foldable expression is a failure.
static FailureOr<std::optional<SmallVector<TemplateExprOp>>> evaluateTemplateExprs(
    TemplateOp templateOp, Operation *target, DenseMap<Attribute, Attribute> &paramNameToConcrete,
    SmallVector<Diagnostic> &deferredExprDiagnostics
) {
  LLVM_DEBUG(
      llvm::dbgs() << "[evaluateTemplateExprs] before: " << debug::toStringList(paramNameToConcrete)
                   << '\n'
  );
  SmallVector<TemplateExprOp> deferredExprs;
  auto destroyDeferredExprs = [&]() {
    for (TemplateExprOp exprOp : deferredExprs) {
      exprOp->destroy();
    }
    deferredExprs.clear();
  };
  bool hasUnresolvedExpression = false;
  for (TemplateExprOp exprOp : templateOp.getConstOps<TemplateExprOp>()) {
    if (!targetMayReferenceTemplateExpr(target, exprOp)) {
      continue;
    }
    // Evaluation and preservation must observe the same concrete type substitutions. In
    // particular, a type-variable binding can make an otherwise non-foldable cast an identity
    // cast, so folding the original expression would be route-dependent.
    FailureOr<std::optional<TemplateExprOp>> convertedExpr =
        cloneDeferredExpr(exprOp, paramNameToConcrete, deferredExprDiagnostics);
    if (failed(convertedExpr)) {
      destroyDeferredExprs();
      return failure();
    }
    if (!convertedExpr->has_value()) {
      // A known binding with a non-concrete converted type defers this specialization, but
      // independent referenced expressions still need to be checked for fatal evaluation errors.
      hasUnresolvedExpression = true;
      continue;
    }
    TemplateExprOp convertedExprOp = **convertedExpr;
    FailureOr<std::optional<Attribute>> result = evaluateExpr(convertedExprOp, paramNameToConcrete);
    if (failed(result)) {
      convertedExprOp->destroy();
      destroyDeferredExprs();
      return failure();
    }
    if (*result) {
      convertedExprOp->destroy();
      Attribute value = result->value();
      auto exprNameAttr = FlatSymbolRefAttr::get(exprOp.getSymNameAttr());
      paramNameToConcrete.try_emplace(exprNameAttr, value);
      LLVM_DEBUG(
          llvm::dbgs() << "[evaluateTemplateExprs] expr @" << exprOp.getSymName()
                       << " evaluated to " << value << '\n'
      );
    } else {
      // Keep the converted detached clone. The caller transfers it into the reduced template, so
      // later specialization starts from the same representation that was just evaluated.
      deferredExprs.push_back(convertedExprOp);
    }
  }
  if (hasUnresolvedExpression) {
    destroyDeferredExprs();
    return std::optional<SmallVector<TemplateExprOp>>();
  }
  LLVM_DEBUG(
      llvm::dbgs() << "[evaluateTemplateExprs] after: " << debug::toStringList(paramNameToConcrete)
                   << '\n'
  );
  return std::optional<SmallVector<TemplateExprOp>>(std::move(deferredExprs));
}

static inline bool tableOffsetIsntSymbol(MemberReadOp op) {
  return !llvm::isa_and_present<SymbolRefAttr>(op.getTableOffset().value_or(nullptr));
}

/// Materialize symbolic member table offsets only from integer template bindings. Member tables are
/// index-addressed, so other concrete attribute kinds emit diagnostics instead of being coerced.
class ClonedMemberReadOpPattern
    : public SymbolUserHelper<ClonedMemberReadOpPattern, MemberReadOp, IntegerAttr> {
  using super = SymbolUserHelper<ClonedMemberReadOpPattern, MemberReadOp, IntegerAttr>;

public:
  ClonedMemberReadOpPattern(
      TypeConverter &converter, MLIRContext *ctx,
      const DenseMap<Attribute, Attribute> &paramNameToInstantiatedValue
  )
      // benefit>0 so this applies instead of GeneralTypeReplacePattern<MemberReadOp>
      : super(converter, ctx, /*patternBenefit=*/1, paramNameToInstantiatedValue) {}

  Attribute getNameAttr(MemberReadOp op) const override {
    return op.getTableOffset().value_or(nullptr);
  }

  LogicalResult handleRewrite(
      Attribute, MemberReadOp op, OpAdaptor, ConversionPatternRewriter &rewriter, IntegerAttr a
  ) const {
    rewriter.modifyOpInPlace(op, [&]() {
      op.setTableOffsetAttr(rewriter.getIndexAttr(fromAPInt(a.getValue())));
    });

    return success();
  }

  LogicalResult handleDefaultRewrite(
      Attribute, MemberReadOp op, OpAdaptor, ConversionPatternRewriter &, Attribute a
  ) const override {
    return op->emitOpError().append(
        "table offset requires an integer template value, but found ", a
    );
  }

  LogicalResult matchAndRewrite(
      MemberReadOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    LLVM_DEBUG(llvm::dbgs() << "[ClonedMemberReadOpPattern]   MemberReadOp: " << op << '\n';);
    if (tableOffsetIsntSymbol(op)) {
      return failure();
    }

    return super::matchAndRewrite(op, adaptor, rewriter);
  }
};

namespace Step1_InstantiateStructs {

/// Clone a `StructDefOp` for one specialization, replacing every known template binding in the
/// cloned definition while retaining unresolved bindings.
class StructCloner {
  ConversionTracker &tracker_;
  ModuleOp rootMod;
  SymbolTableCollection symTables;
  bool reportMissing = true;

  class MappedTypeConverter : public TemplateParamTypeConverter {
    StructType origTy;
    StructType newTy;

  public:
    /// Convert `originalType` to `newType` and apply known template bindings to nested types.
    MappedTypeConverter(
        StructType originalType, StructType newType,
        const DenseMap<Attribute, Attribute> &paramNameToInstantiatedValue
    )
        : TemplateParamTypeConverter(paramNameToInstantiatedValue), origTy(originalType),
          newTy(newType) {

      addConversion([this](StructType inputTy) {
        LLVM_DEBUG(llvm::dbgs() << "[MappedTypeConverter] convert " << inputTy << '\n');

        // Check for replacement of the full type
        if (inputTy == this->origTy) {
          return this->newTy;
        }
        // Check for replacement of parameter symbol names with concrete values
        if (ArrayAttr inputTyParams = inputTy.getParams()) {
          SmallVector<Attribute> updated;
          for (Attribute a : inputTyParams) {
            updated.push_back(convertAttr(a));
          }
          return getStructTypeWithParams(inputTy.getNameRef(), inputTy.getContext(), updated);
        }
        // Otherwise, return the type unchanged
        return inputTy;
      });

      addConversion([this](ArrayType inputTy) {
        // Check for replacement of parameter symbol names with concrete values
        ArrayRef<Attribute> dimSizes = inputTy.getDimensionSizes();
        if (!dimSizes.empty()) {
          SmallVector<Attribute> updated;
          for (Attribute a : dimSizes) {
            updated.push_back(convertIfPossible(a));
          }
          return ArrayType::get(this->convertType(inputTy.getElementType()), updated);
        }
        // Otherwise, return the type unchanged
        return inputTy;
      });
    }
  };

  FailureOr<StructType> genClone(StructType typeAtCaller, ArrayRef<Attribute> typeAtCallerParams) {
    LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   attempting clone of " << typeAtCaller << '\n');
    // Find the StructDefOp for the original StructType
    FailureOr<SymbolLookupResult<StructDefOp>> r =
        typeAtCaller.getDefinition(symTables, rootMod, reportMissing);
    if (failed(r)) {
      LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   skip: cannot find StructDefOp \n");
      return failure(); // getDefinition() already emits a sufficient error message
    }
    LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   found definition\n";);

    StructDefOp origStruct = r->get();
    StructType typeAtDef = origStruct.getType();
    MLIRContext *ctx = origStruct.getContext();
    TemplateOp parentTemplate = getParentOfType<TemplateOp>(origStruct);
    if (!parentTemplate) {
      LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   skip: struct is not in a template\n");
      return failure();
    }
    ModuleOp parentModule = getParentOfType<ModuleOp>(parentTemplate);
    if (!parentModule) {
      LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   skip: template is not in a module\n");
      return failure();
    }

    // Concrete template bindings for this specialization. Parameter values populate the map first;
    // expression values are added once their dependencies become concrete.
    DenseMap<Attribute, Attribute> paramNameToConcrete;
    // Reduced from `typeAtCallerParams` to contain only the non-concrete Attributes.
    ArrayAttr reducedCallerParams = nullptr;
    SmallVector<Attribute> nonConcreteParams;
    size_t concreteParamCount = 0;
    {
      ArrayAttr paramNamesAttr = typeAtDef.getParams();
      ArrayRef<Attribute> paramNames =
          paramNamesAttr ? paramNamesAttr.getValue() : ArrayRef<Attribute> {};

      // pre-conditions
      assert(paramNames.size() == typeAtCallerParams.size());
      assert(paramNames.size() == llvm::range_size(parentTemplate.getConstOps<TemplateParamOp>()));

      for (auto [paramName, next] : llvm::zip_equal(paramNames, typeAtCallerParams)) {
        if (isConcreteAttr<false>(next)) {
          paramNameToConcrete[paramName] = next;
          ++concreteParamCount;
        } else {
          nonConcreteParams.push_back(next);
        }
      }
      // post-conditions
      assert(nonConcreteParams.size() + concreteParamCount == paramNames.size());
      if (!nonConcreteParams.empty()) {
        reducedCallerParams = ArrayAttr::get(ctx, nonConcreteParams);
      }
    }

    if (auto cached = tracker_.getInstantiation(typeAtCaller)) {
      return *cached;
    }

    // This list will be used to build the new remote/external type.
    SmallVector<FlatSymbolRefAttr> typeAtCallerSymPieces = getPieces(typeAtCaller.getNameRef());
    typeAtCallerSymPieces.pop_back(); // drop struct name

    // Evaluate any poly.expr symbols whose param dependencies are now concrete; add them to the
    // map so ClonedBodyConstReadOpPattern can replace uses of those symbols too.
    size_t bindingsBeforeExprEvaluation = paramNameToConcrete.size();
    SmallVector<Diagnostic> deferredExprDiagnostics;
    FailureOr<std::optional<SmallVector<TemplateExprOp>>> exprEvaluation = evaluateTemplateExprs(
        parentTemplate, origStruct.getOperation(), paramNameToConcrete, deferredExprDiagnostics
    );
    if (failed(exprEvaluation)) {
      return failure();
    }
    if (!exprEvaluation->has_value()) {
      return failure();
    }
    SmallVector<TemplateExprOp> deferredExprs = std::move(**exprEvaluation);
    bool expressionMaterialized = paramNameToConcrete.size() > bindingsBeforeExprEvaluation;
    if (concreteParamCount == 0 && !expressionMaterialized) {
      LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   skip: no specialization progress\n");
      for (TemplateExprOp exprOp : deferredExprs) {
        exprOp->destroy();
      }
      return failure();
    }

    FailureOr<InstantiationLayout> layoutResult =
        buildInstantiationLayout(parentTemplate, ArrayAttr(), paramNameToConcrete);
    if (failed(layoutResult)) {
      for (TemplateExprOp exprOp : deferredExprs) {
        exprOp->destroy();
      }
      return failure();
    }
    InstantiationLayout layout = std::move(*layoutResult);
    assert(layout.remainingNames.size() == nonConcreteParams.size());

    if (layout.remainingNames.empty() && !deferredExprs.empty()) {
      deferredExprs.front().emitOpError(
          "cannot complete instantiation while a template expression remains deferred"
      );
      for (TemplateExprOp exprOp : deferredExprs) {
        exprOp->destroy();
      }
      return failure();
    }

    // Clone the original struct.
    StructDefOp newStruct = origStruct.clone();
    convertCalleesInPlace(newStruct, paramNameToConcrete);
    // Keep the inserted owner so a failed body conversion cannot publish a partial clone.
    Operation *insertedOwner = nullptr;
    if (layout.remainingNames.empty()) { // FULL INSTANTIATION CASE
      // Set name of the new struct by prepending its name with instantiated template name.
      newStruct.setSymName(
          (layout.templateNameWithAttrs + mlir::Twine('_') + newStruct.getSymName()).str()
      );
      // Insert 'newStruct' into the parent ModuleOp of the original TemplateOp. Use the
      // `SymbolTable::insert()` function so that the name will be made unique if necessary.
      symTables.getSymbolTable(parentModule).insert(newStruct, Block::iterator(parentTemplate));
      insertedOwner = newStruct.getOperation();
      // Drop the old template name from the list.
      typeAtCallerSymPieces.pop_back();
    } else { // PARTIAL INSTANTIATION CASE
      // Clone the template and set instantiated name.
      TemplateOp newTemplate = parentTemplate.cloneWithoutRegions();
      newTemplate.setSymName(layout.templateNameWithAttrs);
      setInstantiationNamePattern(newTemplate, layout.namePattern);
      assert(newTemplate->getNumRegions() > 0 && "region exists"); // it just doesn't have a block
      newTemplate.getBodyRegion().emplaceBlock();

      // Clone preserved const param/expr ops.
      for (Attribute name : layout.remainingNames) {
        FlatSymbolRefAttr nameSym = llvm::dyn_cast<FlatSymbolRefAttr>(name);
        assert(nameSym && "expected FlatSymbolRefAttr");

        Operation *symOp = symTables.getSymbolTable(parentTemplate).lookup(nameSym.getAttr());
        assert(symOp && "symbol must exist");
        newTemplate.insert(newTemplate.begin(), symOp->clone());
      }
      for (TemplateExprOp exprOp : deferredExprs) {
        newTemplate.getBodyRegion().front().push_back(exprOp.getOperation());
      }

      // Insert the struct into the detached template with a local table. The long-lived
      // collection must not cache a table for a template that may be erased on failure.
      SymbolTable newTemplateSymbols(newTemplate);
      newTemplateSymbols.insert(newStruct);
      symTables.getSymbolTable(parentModule).insert(newTemplate, Block::iterator(parentTemplate));
      insertedOwner = newTemplate.getOperation();

      // Replace the old template name in the list with the new one (get template name after
      // symbol table insertion since it may be modified to make it unique).
      typeAtCallerSymPieces.back() = FlatSymbolRefAttr::get(newTemplate.getSymNameAttr());
    }

    // Retrieve the new type AFTER inserting since the struct name may be appended to make
    // it unique and use the remaining non-concrete parameters from the original type.
    StructType newLocalType = newStruct.getType(reducedCallerParams);
    typeAtCallerSymPieces.push_back(
        FlatSymbolRefAttr::get(newLocalType.getNameRef().getLeafReference())
    );
    StructType newRemoteType =
        StructType::get(asSymbolRefAttr(typeAtCallerSymPieces), newLocalType.getParams());
    LLVM_DEBUG({
      llvm::dbgs() << "[StructCloner]   original def type: " << typeAtDef << '\n';
      llvm::dbgs() << "[StructCloner]   cloned def type: " << newStruct.getType() << '\n';
      llvm::dbgs() << "[StructCloner]   original remote type: " << typeAtCaller << '\n';
      llvm::dbgs() << "[StructCloner]   cloned local type: " << newLocalType << '\n';
      llvm::dbgs() << "[StructCloner]   cloned remote type: " << newRemoteType << '\n';
    });

    SmallVector<Diagnostic> conversionDiagnostics;

    // Within the new struct, replace all references to the original StructType (i.e., the
    // locally-parameterized version) with the new locally-parameterized StructType,
    // and replace all uses of the removed struct parameters with the concrete values.
    MappedTypeConverter tyConv(typeAtDef, newStruct.getType(), paramNameToConcrete);
    ConversionTarget target =
        newConverterDefinedTarget<EmitEqualityOp>(tyConv, ctx, tableOffsetIsntSymbol);
    target.addDynamicallyLegalOp<ConstReadOp>([&paramNameToConcrete](ConstReadOp op) {
      // Legal if it's not in the map of concrete attribute instantiations
      return !paramNameToConcrete.contains(op.getConstNameAttr());
    });

    RewritePatternSet patterns = newGeneralRewritePatternSet<EmitEqualityOp>(tyConv, ctx, target);
    patterns.add<ClonedBodyConstReadOpPattern>(
        tyConv, ctx, paramNameToConcrete, conversionDiagnostics
    );
    patterns.add<ClonedMemberReadOpPattern>(tyConv, ctx, paramNameToConcrete);
    if (failed(applyFullConversion(newStruct, target, std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   instantiating body of struct failed \n");
      // Erase the published owner through its parent table so the block and its symbol-table entry
      // are removed together. A partial template's detached table is local and has already died.
      symTables.getSymbolTable(parentModule).erase(insertedOwner);
      return failure();
    }

    // Publish diagnostics only after the generated owner has passed conversion.
    if (!deferredExprDiagnostics.empty() || !conversionDiagnostics.empty()) {
      SmallVector<Diagnostic> &diagnostics = tracker_.delayedDiagnosticSet(newLocalType);
      diagnostics.append(
          std::make_move_iterator(deferredExprDiagnostics.begin()),
          std::make_move_iterator(deferredExprDiagnostics.end())
      );
      diagnostics.append(
          std::make_move_iterator(conversionDiagnostics.begin()),
          std::make_move_iterator(conversionDiagnostics.end())
      );
    }
    tracker_.recordInstantiation(typeAtCaller, newRemoteType);
    return newRemoteType;
  }

public:
  StructCloner(ConversionTracker &tracker, ModuleOp root)
      : tracker_(tracker), rootMod(root), symTables() {}

  FailureOr<StructType> createInstantiatedClone(StructType orig) {
    LLVM_DEBUG(llvm::dbgs() << "[StructCloner] orig: " << orig << '\n');
    if (ArrayAttr params = orig.getParams(); params && !params.empty()) {
      return genClone(orig, params.getValue());
    }

    // A type with no arguments is normally already concrete. It still needs a clone when its
    // original definition's template has no parameters and the struct references an expression.
    if (!tracker_.mayBeOriginalStructWithExprsAndNoParams(orig)) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "[StructCloner]   skip: definition was not originally in a template with expressions "
             "and no parameters\n"
      );
      return failure();
    }
    SymbolTableCollection tables;
    FailureOr<SymbolLookupResult<StructDefOp>> definition =
        orig.getDefinition(tables, rootMod, /*reportMissing=*/false);
    if (failed(definition)) {
      LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   skip: cannot find definition\n");
      return failure();
    }
    StructDefOp structDef = definition->get();
    if (!tracker_.isOriginalStructWithExprsAndNoParams(structDef)) {
      LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   skip: definition is not an original source\n");
      return failure();
    }
    TemplateOp parentTemplate = getParentOfType<TemplateOp>(structDef);
    if (!parentTemplate || parentTemplate.getConstOps<TemplateExprOp>().empty() ||
        !llvm::any_of(parentTemplate.getConstOps<TemplateExprOp>(), [&](TemplateExprOp exprOp) {
      return targetMayReferenceTemplateExpr(structDef.getOperation(), exprOp);
    })) {
      LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   skip: no referenced expression\n");
      return failure();
    }
    return genClone(orig, ArrayRef<Attribute> {});
  }

  void enableReportMissing() { reportMissing = true; }

  void disableReportMissing() { reportMissing = false; }
};

class DisableReportMissing;

class ParameterizedStructUseTypeConverter : public TypeConverter {
  ConversionTracker &tracker_;
  StructCloner cloner;

  friend DisableReportMissing;

public:
  ParameterizedStructUseTypeConverter(ConversionTracker &tracker, ModuleOp root)
      : TypeConverter(), tracker_(tracker), cloner(tracker, root) {

    addConversion([](Type inputTy) { return inputTy; });

    addConversion([this](StructType inputTy) -> StructType {
      LLVM_DEBUG(
          llvm::dbgs() << "[ParameterizedStructUseTypeConverter] attempting conversion of "
                       << inputTy << '\n';
      );
      // First check for a cached entry
      if (auto opt = tracker_.getInstantiation(inputTy)) {
        return opt.value();
      }

      // Otherwise, try to create a clone of the struct with instantiated params. If that can't be
      // done, return the original type to indicate that it's still legal (for this step at least).
      FailureOr<StructType> cloneRes = cloner.createInstantiatedClone(inputTy);
      if (failed(cloneRes)) {
        return inputTy;
      }
      StructType newTy = cloneRes.value();
      LLVM_DEBUG(
          llvm::dbgs() << "[ParameterizedStructUseTypeConverter] instantiating " << inputTy
                       << " as " << newTy << '\n'
      );
      return newTy;
    });

    addConversion([this](ArrayType inputTy) {
      return inputTy.cloneWith(convertType(inputTy.getElementType()));
    });
  }
};

/// Rebuild struct `compute` and `constrain` calls after their struct types change. Retarget the
/// callee to the converted struct while preserving affine-map and explicit template arguments.
class CallStructFuncPattern : public OpConversionPattern<CallOp> {
  ConversionTracker &tracker_;

public:
  CallStructFuncPattern(TypeConverter &converter, MLIRContext *ctx, ConversionTracker &tracker)
      // benefit>0 so this applies instead of CallOpClassReplacePattern
      : OpConversionPattern<CallOp>(converter, ctx, /*benefit=*/1), tracker_(tracker) {}

  LogicalResult matchAndRewrite(
      CallOp op, OpAdaptor adapter, ConversionPatternRewriter &rewriter
  ) const override {
    LLVM_DEBUG(llvm::dbgs() << "[CallStructFuncPattern] CallOp: " << op << '\n');

    // Convert the result types of the CallOp
    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), newResultTypes))) {
      return op->emitError("Could not convert Op result types.");
    }
    LLVM_DEBUG({
      llvm::dbgs() << "[CallStructFuncPattern]   newResultTypes: "
                   << debug::toStringList(newResultTypes) << '\n';
    });

    // Update the callee to reflect the new struct target if necessary. These checks are based on
    // `CallOp::calleeIsStructC*()` but the types must not come from the CallOp in this case.
    // Instead they must come from the converted versions.
    SymbolRefAttr calleeAttr = op.getCalleeAttr();
    if (op.calleeIsStructCompute()) {
      if (StructType newStTy = getIfSingleton<StructType>(newResultTypes)) {
        LLVM_DEBUG(llvm::dbgs() << "[CallStructFuncPattern]   newStTy: " << newStTy << '\n');
        calleeAttr = appendLeaf(newStTy.getNameRef(), calleeAttr.getLeafReference());
        tracker_.reportDelayedDiagnostics(newStTy, op);
      }
    } else if (op.calleeIsStructConstrain()) {
      if (StructType newStTy = getAtIndex<StructType>(adapter.getArgOperands().getTypes(), 0)) {
        LLVM_DEBUG(llvm::dbgs() << "[CallStructFuncPattern]   newStTy: " << newStTy << '\n');
        calleeAttr = appendLeaf(newStTy.getNameRef(), calleeAttr.getLeafReference());
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "[CallStructFuncPattern] replaced " << op);
    ArrayAttr templateParamsAttr = op.getTemplateParamsAttr();
    ArrayRef<Attribute> templateParams =
        templateParamsAttr ? templateParamsAttr.getValue() : ArrayRef<Attribute>();
    CallOp newOp = replaceOpWithNewOp<CallOp>(
        rewriter, op, newResultTypes, calleeAttr, adapter.getMapOperands(),
        op.getNumDimsPerMapAttr(), adapter.getArgOperands(), templateParams
    );
    (void)newOp; // tell compiler it's intentionally unused in release builds
    LLVM_DEBUG(llvm::dbgs() << " with " << newOp << '\n');
    return success();
  }
};

/// Convert a `MemberDefOp`'s type property even when no body operation reads or writes it.
class MemberDefOpPattern : public OpConversionPattern<MemberDefOp> {
public:
  MemberDefOpPattern(TypeConverter &converter, MLIRContext *ctx)
      // benefit>0 so this applies instead of GeneralTypeReplacePattern<MemberDefOp>
      : OpConversionPattern<MemberDefOp>(converter, ctx, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(
      MemberDefOp op, OpAdaptor /*adapter*/, ConversionPatternRewriter &rewriter
  ) const override {
    LLVM_DEBUG(llvm::dbgs() << "[MemberDefOpPattern] MemberDefOp: " << op << '\n');

    Type oldMemberType = op.getType();
    Type newMemberType = getTypeConverter()->convertType(oldMemberType);
    if (oldMemberType == newMemberType) {
      return failure(); // nothing changed
    }
    rewriter.modifyOpInPlace(op, [&op, &newMemberType]() { op.setType(newMemberType); });
    return success();
  }
};

/// Disables reporting of missing struct symbols during legality checks to avoid showing error
/// diagnostics that are not actually errors.
class DisableReportMissing : public LegalityCheckCallback {
  ParameterizedStructUseTypeConverter &tyConv;

public:
  explicit DisableReportMissing(ParameterizedStructUseTypeConverter &tc) : tyConv(tc) {}

  void checkStarted() override { tyConv.cloner.disableReportMissing(); }

  void checkEnded(bool) override { tyConv.cloner.enableReportMissing(); }
};

LogicalResult run(ModuleOp modOp, ConversionTracker &tracker) {
  MLIRContext *ctx = modOp.getContext();
  ParameterizedStructUseTypeConverter tyConv(tracker, modOp);
  DisableReportMissing drm(tyConv);
  ConversionTarget target = newConverterDefinedTargetWithCallback<>(tyConv, ctx, drm);
  // Keep source templates with expressions and no parameters out of type conversion. TypeConverter
  // caches results by Type, so preserving a source type based on the enclosing operation would
  // leak into external uses of the same type and hide a cached specialization.
  target.addLegalOp<TemplateOp>();
  target.markOpRecursivelyLegal<TemplateOp>([](TemplateOp op) {
    return !op.hasConstOps<TemplateParamOp>() && op.hasConstOps<TemplateExprOp>();
  });
  RewritePatternSet patterns = newGeneralRewritePatternSet(tyConv, ctx, target);
  patterns.add<CallStructFuncPattern>(tyConv, ctx, tracker);
  patterns.add<MemberDefOpPattern>(tyConv, ctx);
  return applyPartialConversion(modOp, target, std::move(patterns));
}

/// Create an instantiated clone of the struct type specified by the `MAIN_ATTR_NAME` attribute
/// on `modOp`, if it exists, and update the attribute to refer to the new instantiated type.
LogicalResult instantiateMainStruct(ModuleOp modOp, ConversionTracker &tracker) {
  FailureOr<StructType> mainTypeOpt = getMainInstanceType(modOp);
  if (failed(mainTypeOpt)) {
    return failure();
  }

  StructType mainType = mainTypeOpt.value();
  if (!mainType || tracker.getInstantiation(mainType)) {
    return success();
  }

  if (isNullOrEmpty(mainType.getParams())) {
    if (!tracker.mayBeOriginalStructWithExprsAndNoParams(mainType)) {
      return success();
    }
    SymbolTableCollection tables;
    FailureOr<SymbolLookupResult<StructDefOp>> definition =
        mainType.getDefinition(tables, modOp, /*reportMissing=*/false);
    if (failed(definition)) {
      return success();
    }
    StructDefOp structDef = definition->get();
    if (!tracker.isOriginalStructWithExprsAndNoParams(structDef)) {
      return success();
    }
    TemplateOp parentTemplate = getParentOfType<TemplateOp>(structDef);
    if (!parentTemplate || parentTemplate.getConstOps<TemplateExprOp>().empty() ||
        !llvm::any_of(parentTemplate.getConstOps<TemplateExprOp>(), [&](TemplateExprOp exprOp) {
      return targetMayReferenceTemplateExpr(structDef.getOperation(), exprOp);
    })) {
      return success();
    }
  }

  StructCloner cloner(tracker, modOp);
  FailureOr<StructType> cloneRes = cloner.createInstantiatedClone(mainType);
  if (failed(cloneRes)) {
    return failure();
  }

  modOp->setAttr(MAIN_ATTR_NAME, TypeAttr::get(cloneRes.value()));
  return success();
}

} // namespace Step1_InstantiateStructs

namespace Step2_InstantiateFunctions {

/// Return the callee-side unification-derived value for a template parameter, if any.
inline static std::optional<Attribute>
inferUnifiedParam(const UnificationMap &unifyResult, SymbolRefAttr paramName) {
  auto it = unifyResult.find({paramName, Side::RHS});
  return (it == unifyResult.end()) ? std::nullopt : std::make_optional(it->second);
}

/// Emit the match failure used when an inferred instantiation violates a template parameter's
/// declared type restriction.
inline static LogicalResult failIncompatibleInferredParam(
    CallOp op, PatternRewriter &rewriter, FlatSymbolRefAttr paramName, TemplateParamOp paramOp
) {
  LLVM_DEBUG(
      llvm::dbgs() << "[InstantiateFuncAtCallOp]  unification for param '" << paramName
                   << "': incompatible with specified param type. MUST FAIL!\n"
  );
  return rewriter.notifyMatchFailure(op, [&paramName, &paramOp](Diagnostic &diag) {
    diag.append("inferred value for parameter '")
        .append(paramName)
        .append("' is incompatible with specified param type")
        .attachNote(paramOp.getLoc())
        .append("template parameter declared here");
  });
}

/// Searches a parameterized callee body for concrete type evidence that resolves a wildcard
/// template parameter, following both local unifiable casts and nested template calls.
class WildcardTypeBodyInferer final {
  SymbolTableCollection &symTables_;
  const DenseMap<Attribute, Attribute> &paramNameToConcrete_;
  SmallVector<std::pair<Operation *, FlatSymbolRefAttr>> activeInferences_;

public:
  WildcardTypeBodyInferer(
      SymbolTableCollection &symTables, const DenseMap<Attribute, Attribute> &paramNameToConcrete
  )
      : symTables_(symTables), paramNameToConcrete_(paramNameToConcrete) {}

  std::optional<Attribute> infer(FuncDefOp func, FlatSymbolRefAttr paramName) {
    if (llvm::any_of(activeInferences_, [&](const auto &e) {
      return e.first == func.getOperation() && e.second == paramName;
    })) {
      return std::nullopt;
    }
    activeInferences_.emplace_back(func.getOperation(), paramName);

    TemplateParamTypeConverter tyConv(paramNameToConcrete_);
    std::optional<Attribute> inferred;
    bool ambiguous = false;

    // Record a concrete candidate unless it conflicts with an earlier one, in which
    // case the wildcard is treated as ambiguous and left unresolved.
    auto noteCandidate = [&inferred, &ambiguous](Attribute candidate) {
      if (!candidate || !isConcreteAttr(candidate)) {
        return WalkResult::advance();
      }
      if (!inferred.has_value()) {
        inferred = candidate;
        return WalkResult::advance();
      }
      if (*inferred != candidate) {
        ambiguous = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    };

    WalkResult walkResult = func.walk([&](Operation *bodyOp) {
      if (auto castOp = llvm::dyn_cast<UnifiableCastOp>(bodyOp)) {
        Type inputTy = tyConv.convertType(castOp.getInput().getType());
        Type resultTy = tyConv.convertType(castOp.getResult().getType());
        if (auto inputTvar = llvm::dyn_cast<TypeVarType>(inputTy);
            inputTvar && inputTvar.getNameRef() == paramName && isConcreteType(resultTy)) {
          return noteCandidate(TypeAttr::get(resultTy));
        }
        if (auto resultTvar = llvm::dyn_cast<TypeVarType>(resultTy);
            resultTvar && resultTvar.getNameRef() == paramName && isConcreteType(inputTy)) {
          return noteCandidate(TypeAttr::get(inputTy));
        }
        return WalkResult::advance();
      }

      auto nestedCall = llvm::dyn_cast<CallOp>(bodyOp);
      if (!nestedCall) {
        return WalkResult::advance();
      }

      FailureOr<SymbolLookupResult<FuncDefOp>> nestedTgtOpt =
          nestedCall.getCalleeTarget(symTables_);
      if (failed(nestedTgtOpt)) {
        return WalkResult::advance();
      }
      FuncDefOp nestedTgt = nestedTgtOpt->get();
      auto nestedTemplate = llvm::dyn_cast<TemplateOp>(nestedTgt->getParentOp());
      if (!nestedTemplate) {
        return WalkResult::advance();
      }

      TypeRange nestedResultTypes = nestedTgt.getFunctionType().getResults();
      for (auto [result, nestedResultTy] :
           llvm::zip_equal(nestedCall.getResults(), nestedResultTypes)) {
        Type convertedResultTy = tyConv.convertType(result.getType());
        auto resultTvar = llvm::dyn_cast<TypeVarType>(convertedResultTy);
        auto nestedTvar = llvm::dyn_cast<TypeVarType>(nestedResultTy);
        if (!resultTvar || !nestedTvar || resultTvar.getNameRef() != paramName) {
          continue;
        }
        if (std::optional<Attribute> candidate = inferFromExplicitNestedCallParams(
                nestedCall, nestedTemplate, nestedTvar.getNameRef(), tyConv
            )) {
          WalkResult candidateResult = noteCandidate(*candidate);
          if (candidateResult.wasInterrupted()) {
            return candidateResult;
          }
          continue;
        }
        if (std::optional<Attribute> candidate = infer(nestedTgt, nestedTvar.getNameRef())) {
          WalkResult candidateResult = noteCandidate(*candidate);
          if (candidateResult.wasInterrupted()) {
            return candidateResult;
          }
        }
      }
      return WalkResult::advance();
    });

    activeInferences_.pop_back();
    if (ambiguous || (walkResult.wasInterrupted() && !inferred.has_value())) {
      return std::nullopt;
    }
    return inferred;
  }

private:
  std::optional<Attribute> inferFromExplicitNestedCallParams(
      CallOp nestedCall, TemplateOp nestedTemplate, FlatSymbolRefAttr nestedParamName,
      const TemplateParamTypeConverter &tyConv
  ) const {
    ArrayAttr nestedCallParams = nestedCall.getTemplateParamsAttr();
    if (isNullOrEmpty(nestedCallParams)) {
      return std::nullopt;
    }

    for (auto [paramOp, attr] :
         llvm::zip_equal(nestedTemplate.getConstOps<TemplateParamOp>(), nestedCallParams)) {
      auto paramName = FlatSymbolRefAttr::get(paramOp.getSymNameAttr());
      if (paramName != nestedParamName) {
        continue;
      }
      Attribute convertedAttr = tyConv.convertAttr(attr);
      return isConcreteAttr(convertedAttr) ? std::make_optional(convertedAttr) : std::nullopt;
    }
    return std::nullopt;
  }
};

/// Rewrite cloned scalar array reads to ranged extract ops when a wildcard element type
/// resolves to a higher-rank array.
class ClonedBodyArrayReadOpPattern final : public OpConversionPattern<ReadArrayOp> {
public:
  using OpConversionPattern<ReadArrayOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ReadArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    Type newResultTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!llvm::isa<ArrayType>(newResultTy)) {
      return failure();
    }
    replaceOpWithNewOp<ExtractArrayOp>(
        rewriter, op, newResultTy, adaptor.getArrRef(), adaptor.getIndices()
    );
    return success();
  }
};

/// Rewrite cloned scalar array writes to ranged inserts when a wildcard element type
/// resolves to a higher-rank array.
class ClonedBodyArrayWriteOpPattern final : public OpConversionPattern<WriteArrayOp> {
public:
  using OpConversionPattern<WriteArrayOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      WriteArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (!llvm::isa<ArrayType>(adaptor.getRvalue().getType())) {
      return failure();
    }
    replaceOpWithNewOp<InsertArrayOp>(
        rewriter, op, adaptor.getArrRef(), adaptor.getIndices(), adaptor.getRvalue()
    );
    return success();
  }
};

/// Apply the given template substitutions throughout a cloned function, then verify every nested
/// `CallOp` against its converted target. Conversion warnings are reported only after both stages
/// succeed, so a rejected clone cannot leak warnings from work that is rolled back.
static LogicalResult applyBodyConversions(
    CallOp op, FuncDefOp newFunc, const DenseMap<Attribute, Attribute> &paramNameToConcrete
) {
  MLIRContext *ctx = op.getContext();
  TemplateParamTypeConverter tyConv(paramNameToConcrete);
  ConversionTarget target = newConverterDefinedTarget<>(tyConv, ctx, tableOffsetIsntSymbol);
  target.addDynamicallyLegalOp<ConstReadOp>([&paramNameToConcrete](ConstReadOp p) {
    // Legal if it's not in the map of concrete attribute instantiations
    return !paramNameToConcrete.contains(p.getConstNameAttr());
  });
  SmallVector<Diagnostic> delayedDiagnostics;
  RewritePatternSet bodyPatterns = newGeneralRewritePatternSet(tyConv, ctx, target);
  bodyPatterns.add<ClonedBodyConstReadOpPattern>(
      tyConv, ctx, paramNameToConcrete, delayedDiagnostics
  );
  bodyPatterns.add<ClonedBodyArrayReadOpPattern, ClonedBodyArrayWriteOpPattern>(tyConv, ctx);
  bodyPatterns.add<ClonedMemberReadOpPattern>(tyConv, ctx, paramNameToConcrete);
  if (failed(applyFullConversion(newFunc, target, std::move(bodyPatterns)))) {
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "[InstantiateFuncAtCallOp]   instantiated clone: " << newFunc << '\n');
  SymbolTableCollection tables;
  WalkResult res = newFunc.walk([&tables](CallOp nestedCall) {
    return WalkResult(nestedCall.verifySymbolUses(tables));
  });
  if (res.wasInterrupted()) {
    return failure();
  }
  ::reportDelayedDiagnostics(op, std::move(delayedDiagnostics));
  return success();
}

/// Specialize calls whose target is a free function inside a `poly.template`. Materialize every
/// known binding, create a full clone or reduced template when a concrete binding or referenced
/// expression makes progress, and leave the call unchanged when no such progress is possible.
class InstantiateFuncAtCallOp final : public OpRewritePattern<CallOp> {
  ConversionTracker &tracker_;

public:
  InstantiateFuncAtCallOp(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern<CallOp>(ctx), tracker_(tracker) {}

  LogicalResult matchAndRewrite(CallOp op, PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "[InstantiateFuncAtCallOp] op: " << op << '\n');

    if (calleeReferencesTemplateParam(op)) {
      return failure();
    }

    // Lookup callee target function
    SymbolTableCollection symTables;
    FailureOr<SymbolLookupResult<FuncDefOp>> callTgtOpt = op.getCalleeTarget(symTables);
    if (failed(callTgtOpt)) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "could not find target function for call";
      });
    }
    FuncDefOp callTgt = callTgtOpt->get();

    // Check if callee is within a TemplateOp
    TemplateOp parentTemplate = llvm::dyn_cast<TemplateOp>(callTgt->getParentOp());
    if (!parentTemplate) {
      return failure(); // nothing to do if not parameterized
    }
    LLVM_DEBUG(
        llvm::dbgs() << "[InstantiateFuncAtCallOp]  target function in template "
                     << parentTemplate.getSymName() << '\n'
    );

    // Perform type unification with tracking to infer the instantiated type(s). Even though
    // `CallOp` verification already checked that caller and callee types unify, the progress of
    // instantiation so far may have brought together a chain of calls across templates where each
    // individual unification check passed due to permissive type variables and/or symbols in the
    // middle but the overall chain does not unify. Hence, this unification may fail and should
    // produce a meaningful error message if it does.
    // See: `test/Transforms/Flattening/instantiate_funcs_fail.llzk`
    FailureOr<UnificationMap> unifyResult = unifyTypeSignature(op, callTgt, rewriter);
    if (failed(unifyResult)) {
      return failure();
    }
    LLVM_DEBUG(
        llvm::dbgs() << "[InstantiateFuncAtCallOp]  unifications of types: "
                     << debug::toStringList(unifyResult.value()) << '\n'
    );

    // Concrete template bindings for this specialization. Parameters are collected first;
    // expression values are added once their dependencies become concrete.
    DenseMap<Attribute, Attribute> paramNameToConcrete;
    if (failed(collectConcreteTemplateParams(
            op, rewriter, symTables, callTgt, parentTemplate, unifyResult.value(),
            paramNameToConcrete
        ))) {
      return failure();
    }

    bool hasConcreteParamBinding =
        llvm::any_of(parentTemplate.getConstOps<TemplateParamOp>(), [&](TemplateParamOp paramOp) {
      return paramNameToConcrete.contains(FlatSymbolRefAttr::get(paramOp.getNameAttr()));
    });
    size_t bindingsBeforeExprEvaluation = paramNameToConcrete.size();
    SmallVector<Diagnostic> deferredExprDiagnostics;
    FailureOr<std::optional<SmallVector<TemplateExprOp>>> exprEvaluation = evaluateTemplateExprs(
        parentTemplate, callTgt.getOperation(), paramNameToConcrete, deferredExprDiagnostics
    );
    if (failed(exprEvaluation)) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "failure while evaluating template expressions";
      });
    }
    if (!exprEvaluation->has_value()) {
      return failure();
    }
    SmallVector<TemplateExprOp> deferredExprs = std::move(**exprEvaluation);
    bool expressionMaterialized = paramNameToConcrete.size() > bindingsBeforeExprEvaluation;
    if (!hasConcreteParamBinding && !expressionMaterialized) {
      LLVM_DEBUG(llvm::dbgs() << "[InstantiateFuncAtCallOp]  skip: no specialization progress\n");
      for (TemplateExprOp exprOp : deferredExprs) {
        exprOp->destroy();
      }
      return failure();
    }

    FailureOr<InstantiationLayout> layoutResult =
        buildInstantiationLayout(parentTemplate, op.getTemplateParamsAttr(), paramNameToConcrete);
    if (failed(layoutResult)) {
      for (TemplateExprOp exprOp : deferredExprs) {
        exprOp->destroy();
      }
      return failure();
    }
    InstantiationLayout layout = std::move(*layoutResult);
    ModuleOp parentModule = getParentOfType<ModuleOp>(parentTemplate);
    assert(parentModule && "TemplateOp must be nested in a ModuleOp");

    SymbolRefAttr originalCalleeAttr = op.getCalleeAttr();
    if (layout.remainingNames.empty() && !deferredExprs.empty()) {
      LogicalResult result = rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "cannot complete instantiation while a template expression remains deferred";
      });
      for (TemplateExprOp exprOp : deferredExprs) {
        exprOp->destroy();
      }
      return result;
    }
    FailureOr<SymbolRefAttr> newCalleeAttr =
        layout.remainingNames.empty()
            ? instantiateFully(
                  op, rewriter, symTables, callTgt, parentTemplate, parentModule,
                  layout.templateNameWithAttrs, paramNameToConcrete
              )
            : instantiatePartially(
                  op, rewriter, symTables, callTgt, parentTemplate, parentModule, layout,
                  paramNameToConcrete, tracker_, deferredExprs, deferredExprDiagnostics
              );
    if (failed(newCalleeAttr)) {
      return failure();
    }

    if (layout.remainingNames.empty()) {
      ::reportDelayedDiagnostics(op, std::move(deferredExprDiagnostics));
    }

    tracker_.recordInstantiation(originalCalleeAttr);

    // Update the CallOp to point to the instantiated function and mark the module as modified.
    rewriter.modifyOpInPlace(op, [&op, &newCalleeAttr, &layout]() {
      LLVM_DEBUG({
        llvm::dbgs() << "[InstantiateFuncAtCallOp]  updating callee from " << op.getCalleeAttr()
                     << " to " << *newCalleeAttr << '\n';
      });
      op.setCalleeAttr(*newCalleeAttr);
      op.setTemplateParamsAttr(layout.rewrittenCallParams);
    });
    tracker_.updateModifiedFlag(true);
    return success();
  }

private:
  /// Re-run call/callee type unification so flattening can surface a useful error if a chain of
  /// partially-instantiated calls stops unifying once earlier substitutions have been applied.
  static FailureOr<UnificationMap>
  unifyTypeSignature(CallOp op, FuncDefOp callTgt, PatternRewriter &rewriter) {
    FailureOr<UnificationMap> unifyResult = op.unifyTypeSignature(callTgt.getFunctionType());
    if (succeeded(unifyResult)) {
      return unifyResult;
    }
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag.append("target function type does not unify with call type ")
          .append(op.getTypeSignature())
          .attachNote(callTgt.getLoc())
          .append("target function declared here");
    });
  }

  /// Populate the concrete subset of template parameters for this specialization using explicit
  /// arguments, signature unification, or body inference for wildcard type parameters.
  static LogicalResult collectConcreteTemplateParams(
      CallOp op, PatternRewriter &rewriter, SymbolTableCollection &symTables, FuncDefOp callTgt,
      TemplateOp parentTemplate, const UnificationMap &unifyResult,
      DenseMap<Attribute, Attribute> &paramNameToConcrete
  ) {
    auto realParams = parentTemplate.getConstOps<TemplateParamOp>();
    ArrayAttr callParams = op.getTemplateParamsAttr();
    LLVM_DEBUG(
        llvm::dbgs() << "[InstantiateFuncAtCallOp]  TemplateParamsAttr: " << callParams << '\n'
    );

    auto recordConcreteParam = [&](FlatSymbolRefAttr paramName, TemplateParamOp paramOp,
                                   Attribute concreteValue) -> LogicalResult {
      if (failed(op.verifyTemplateParamCompatibility(concreteValue, paramOp))) {
        return failIncompatibleInferredParam(op, rewriter, paramName, paramOp);
      }
      paramNameToConcrete[paramName] = concreteValue;
      return success();
    };

    // If there's no template instantiation list, must infer all template parameters.
    if (isNullOrEmpty(callParams)) {
      for (auto paramOp : realParams) {
        auto paramName = FlatSymbolRefAttr::get(paramOp.getSymNameAttr());
        auto inferredValOpt = inferUnifiedParam(unifyResult, paramName);
        if (!inferredValOpt.has_value()) {
          LLVM_DEBUG(
              llvm::dbgs() << "[InstantiateFuncAtCallOp]  unification for param '" << paramName
                           << "': not found\n"
          );
          continue;
        }
        Attribute inferredVal = *inferredValOpt;
        LLVM_DEBUG(
            llvm::dbgs() << "[InstantiateFuncAtCallOp]  inferredVal: " << inferredVal << '\n'
        );
        if (!isConcreteAttr(inferredVal)) {
          LLVM_DEBUG(
              llvm::dbgs() << "[InstantiateFuncAtCallOp]  unification for param '" << paramName
                           << "': not concrete, " << inferredVal << '\n'
          );
          continue;
        }
        if (failed(recordConcreteParam(paramName, paramOp, inferredVal))) {
          return failure();
        }
      }
      return success();
    }

    // As stated earlier, need to run the verification checks again to ensure the
    // instantiation is valid, except for the size check because that cannot change.
    assert((callParams.size() == llvm::range_size(realParams)) && "per CallOpVerifier");
    if (failed(op.verifyTemplateParamCompatibility(realParams))) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag.append("incompatible with specified param type(s)");
      });
    }
    if (failed(op.verifyTemplateParamsMatchInferred(realParams, unifyResult))) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag.append("incompatible with inferred param value(s)");
      });
    }

    // When template parameters are specified on the CallOp, use them as the source of truth
    // for concrete arguments, then infer wildcard parameters against the full explicit map.
    SmallVector<std::pair<TemplateParamOp, FlatSymbolRefAttr>> wildcardParams;
    for (auto [paramOp, attr] : llvm::zip_equal(realParams, callParams.getValue())) {
      auto paramName = FlatSymbolRefAttr::get(paramOp.getSymNameAttr());
      AttrConcreteness classification = classifyAttrConcreteness(attr);
      if (classification == AttrConcreteness::Concrete) {
        if (failed(recordConcreteParam(paramName, paramOp, attr))) {
          return failure();
        }
        continue;
      }

      if (classification == AttrConcreteness::NonConcrete) {
        LLVM_DEBUG(
            llvm::dbgs() << "[InstantiateFuncAtCallOp]  unification for param '" << paramName
                         << "': not concrete, " << attr << '\n'
        );
        continue;
      }
      wildcardParams.emplace_back(paramOp, paramName);
    }

    WildcardTypeBodyInferer bodyInferer(symTables, paramNameToConcrete);
    for (auto [paramOp, paramName] : wildcardParams) {
      auto inferredValOpt = inferUnifiedParam(unifyResult, paramName);
      if (inferredValOpt.has_value() && isConcreteAttr(*inferredValOpt)) {
        LLVM_DEBUG(
            llvm::dbgs() << "[InstantiateFuncAtCallOp]  inferredVal: " << *inferredValOpt << '\n'
        );
        if (failed(recordConcreteParam(paramName, paramOp, *inferredValOpt))) {
          return failure();
        }
        continue;
      }

      inferredValOpt = bodyInferer.infer(callTgt, paramName);
      if (inferredValOpt.has_value() && isConcreteAttr(*inferredValOpt)) {
        LLVM_DEBUG(
            llvm::dbgs() << "[InstantiateFuncAtCallOp]  body-inferred value for param '"
                         << paramName << "': " << *inferredValOpt << '\n'
        );
        if (failed(recordConcreteParam(paramName, paramOp, *inferredValOpt))) {
          return failure();
        }
      }
    }
    return success();
  }

  /// Create or reuse a fully-instantiated clone in the parent module and return the rewritten
  /// module-level callee reference.
  static FailureOr<SymbolRefAttr> instantiateFully(
      CallOp op, PatternRewriter &rewriter, SymbolTableCollection &symTables, FuncDefOp callTgt,
      TemplateOp parentTemplate, ModuleOp parentModule, StringRef templateNameWithAttrs,
      const DenseMap<Attribute, Attribute> &paramNameToConcrete
  ) {
    MLIRContext *ctx = op.getContext();
    std::string newFuncName =
        (mlir::Twine(templateNameWithAttrs) + "_" + callTgt.getSymName()).str();
    StringRef actualNewFuncName = newFuncName;
    if (!symTables.getSymbolTable(parentModule).lookup(newFuncName)) {
      FuncDefOp newFunc = callTgt.clone();
      newFunc.setSymName(newFuncName);
      convertCalleesInPlace(newFunc, paramNameToConcrete);
      // Insert before the TemplateOp; symbol table may adjust the name to ensure uniqueness.
      symTables.getSymbolTable(parentModule).insert(newFunc, Block::iterator(parentTemplate));
      actualNewFuncName = newFunc.getSymName();
      LLVM_DEBUG(
          llvm::dbgs() << "[InstantiateFuncAtCallOp]  created full instantiation function: "
                       << actualNewFuncName << '\n'
      );
      if (failed(applyBodyConversions(op, newFunc, paramNameToConcrete))) {
        LLVM_DEBUG(
            llvm::dbgs() << "[InstantiateFuncAtCallOp]   body conversion failed for "
                         << actualNewFuncName << '\n'
        );
        // Remove the operation through the table that inserted it so a failed clone leaves no
        // stale symbol entry for a later specialization with the same preferred name.
        symTables.getSymbolTable(parentModule).erase(newFunc);
        return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
          diag.append("failure while creating instantiated function '", actualNewFuncName, '\'');
        });
      }
    } else {
      LLVM_DEBUG(
          llvm::dbgs() << "[InstantiateFuncAtCallOp]  reusing full instantiation function: "
                       << actualNewFuncName << '\n'
      );
    }

    // Callee: drop template & original function names, add the new module-level function name.
    // Original: @[prefix...]::@TemplateName::@funcName
    // New:      @[prefix...]::@newFuncName
    SmallVector<FlatSymbolRefAttr> symPieces = getPieces(op.getCalleeAttr());
    assert(symPieces.size() >= 2 && "callee must include at least template and function names");
    symPieces.pop_back(); // remove original function name
    symPieces.pop_back(); // remove template name
    symPieces.push_back(FlatSymbolRefAttr::get(StringAttr::get(ctx, actualNewFuncName)));
    return asSymbolRefAttr(symPieces);
  }

  /// Create or reuse a partially-instantiated template that preserves the remaining non-concrete
  /// parameters and return the rewritten nested callee reference.
  /// Reuse is keyed by the source function and exact ordered concrete bindings. The rendered name
  /// is only a preferred symbol name and may be changed by SymbolTable insertion.
  static FailureOr<SymbolRefAttr> instantiatePartially(
      CallOp op, PatternRewriter &rewriter, SymbolTableCollection &symTables, FuncDefOp callTgt,
      TemplateOp parentTemplate, ModuleOp parentModule, const InstantiationLayout &layout,
      const DenseMap<Attribute, Attribute> &paramNameToConcrete, ConversionTracker &tracker,
      ArrayRef<TemplateExprOp> deferredExprs, SmallVector<Diagnostic> &deferredExprDiagnostics
  ) {
    if (auto cached = tracker.lookupPartialFuncInstantiation(callTgt, layout.concreteParamKey)) {
      SmallVector<FlatSymbolRefAttr> symPieces = getPieces(op.getCalleeAttr());
      SmallVector<FlatSymbolRefAttr> cachedSuffix = getPieces(*cached);
      assert(symPieces.size() >= 2 && "callee must include at least template and function names");
      assert(cachedSuffix.size() == 2 && "cached callee suffix must contain template and function");
      symPieces.pop_back();
      symPieces.pop_back();
      symPieces.push_back(cachedSuffix[0]);
      symPieces.push_back(cachedSuffix[1]);
      SymbolRefAttr cachedCallee = asSymbolRefAttr(symPieces);
      LLVM_DEBUG(
          llvm::dbgs() << "[InstantiateFuncAtCallOp]  reusing partial instantiation: "
                       << cachedCallee << '\n'
      );
      for (TemplateExprOp exprOp : deferredExprs) {
        exprOp->destroy();
      }
      ::reportDelayedDiagnostics(op, std::move(deferredExprDiagnostics));
      return cachedCallee;
    }
    TemplateOp newTemplate = parentTemplate.cloneWithoutRegions();
    newTemplate.setSymName(layout.templateNameWithAttrs);
    setInstantiationNamePattern(newTemplate, layout.namePattern);
    assert(newTemplate->getNumRegions() > 0 && "region exists");
    newTemplate.getBodyRegion().emplaceBlock();

    Block &newTemplateBody = newTemplate.getBodyRegion().front();
    for (Attribute name : layout.remainingNames) {
      FlatSymbolRefAttr nameSym = llvm::cast<FlatSymbolRefAttr>(name);
      Operation *paramOp = symTables.getSymbolTable(parentTemplate).lookup(nameSym.getAttr());
      assert(paramOp && "symbol must exist");
      newTemplateBody.push_back(paramOp->clone());
    }
    for (TemplateExprOp exprOp : deferredExprs) {
      newTemplateBody.push_back(exprOp.getOperation());
    }

    // Clone and partially convert the function (concretize only the concrete params).
    FuncDefOp newFunc = callTgt.clone();
    convertCalleesInPlace(newFunc, paramNameToConcrete);

    // Insert before body conversion so nested concrete callees verify from the root module. Use
    // SymbolTable::insert() so both physical symbol names are unique if necessary.
    // Use a local table for the detached template so `symTables` cannot retain state for a
    // template that rollback may erase.
    {
      SymbolTable newTemplateSymbols(newTemplate);
      newTemplateSymbols.insert(newFunc);
    }
    symTables.getSymbolTable(parentModule).insert(newTemplate, Block::iterator(parentTemplate));
    if (failed(applyBodyConversions(op, newFunc, paramNameToConcrete))) {
      std::string newFuncName = newFunc.getSymName().str();
      LLVM_DEBUG(
          llvm::dbgs() << "[InstantiateFuncAtCallOp]   body conversion failed for " << newFuncName
                       << '\n'
      );
      // Erase through the parent table so the operation and its published symbol entry roll back
      // together. No table for the erased template is retained in `symTables`.
      symTables.getSymbolTable(parentModule).erase(newTemplate);
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag.append("failure while creating instantiated function '", newFuncName, '\'');
      });
    }

    ::reportDelayedDiagnostics(op, std::move(deferredExprDiagnostics));

    // Use the post-insertion names. The preferred template name may have collided.
    SmallVector<FlatSymbolRefAttr> symPieces = getPieces(op.getCalleeAttr());
    assert(symPieces.size() >= 2 && "callee must include at least template and function names");
    symPieces.pop_back();
    symPieces.pop_back(); // remove original template name
    symPieces.push_back(FlatSymbolRefAttr::get(newTemplate.getSymNameAttr()));
    symPieces.push_back(FlatSymbolRefAttr::get(newFunc.getSymNameAttr()));
    SymbolRefAttr newCallee = asSymbolRefAttr(symPieces);

    LLVM_DEBUG(
        llvm::dbgs() << "[InstantiateFuncAtCallOp]  created partial instantiation: " << newCallee
                     << '\n'
    );
    // Publish only after insertion and body conversion have succeeded.
    tracker.recordPartialFuncInstantiation(callTgt, layout.concreteParamKey, newTemplate, newFunc);
    return newCallee;
  }
};

LogicalResult run(ModuleOp modOp, ConversionTracker &tracker) {
  MLIRContext *ctx = modOp.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<InstantiateFuncAtCallOp>(ctx, tracker);
  MatchFailureListener failureListener;
  walkAndApplyPatterns(modOp, std::move(patterns), &failureListener);
  return failure(failureListener.hadFailure);
}

} // namespace Step2_InstantiateFunctions

namespace Step3_Unroll {

// TODO: not guaranteed to work with WhileOp, can try with our custom attributes though.
template <HasInterface<LoopLikeOpInterface> OpClass>
class LoopUnrollPattern : public OpRewritePattern<OpClass> {
public:
  using OpRewritePattern<OpClass>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpClass loopOp, PatternRewriter &rewriter) const override {
    if (auto maybeConstant = getConstantTripCount(loopOp)) {
      uint64_t tripCount = *maybeConstant;
      if (tripCount == 0) {
        rewriter.eraseOp(loopOp);
        return success();
      } else if (tripCount == 1) {
        return loopOp.promoteIfSingleIteration(rewriter);
      }
      return loopUnrollByFactor(loopOp, tripCount);
    }
    return failure();
  }

private:
  /// Returns the trip count of the loop-like op if its low bound, high bound and step are
  /// constants, `nullopt` otherwise. Trip count is computed as ceilDiv(highBound - lowBound, step).
  static std::optional<int64_t> getConstantTripCount(LoopLikeOpInterface loopOp) {
    std::optional<OpFoldResult> lbVal = loopOp.getSingleLowerBound();
    std::optional<OpFoldResult> ubVal = loopOp.getSingleUpperBound();
    std::optional<OpFoldResult> stepVal = loopOp.getSingleStep();
    if (!lbVal.has_value() || !ubVal.has_value() || !stepVal.has_value()) {
      return std::nullopt;
    }
    return constantTripCount(lbVal.value(), ubVal.value(), stepVal.value());
  }
};

LogicalResult run(ModuleOp modOp, ConversionTracker &tracker) {
  MLIRContext *ctx = modOp.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<LoopUnrollPattern<scf::ForOp>>(ctx);
  patterns.add<LoopUnrollPattern<affine::AffineForOp>>(ctx);

  return applyAndFoldGreedily(modOp, tracker, std::move(patterns));
}
} // namespace Step3_Unroll

namespace Step4_InstantiateAffineMaps {

// Adapted from `mlir::getConstantIntValues()` but that one failed in CI for an unknown reason. This
// version uses a basic loop instead of llvm::map_to_vector().
std::optional<SmallVector<int64_t>> getConstantIntValues(ArrayRef<OpFoldResult> ofrs) {
  SmallVector<int64_t> res;
  for (OpFoldResult ofr : ofrs) {
    std::optional<int64_t> cv = getConstantIntValue(ofr);
    if (!cv.has_value()) {
      return std::nullopt;
    }
    res.push_back(cv.value());
  }
  return res;
}

struct AffineMapFolder {
  struct Input {
    OperandRangeRange mapOpGroups;
    DenseI32ArrayAttr dimsPerGroup;
    ArrayRef<Attribute> paramsOfStructTy;
  };

  struct Output {
    SmallVector<SmallVector<Value>> mapOpGroups;
    SmallVector<int32_t> dimsPerGroup;
    SmallVector<Attribute> paramsOfStructTy;
  };

  static inline SmallVector<ValueRange> getConvertedMapOpGroups(Output out) {
    return llvm::map_to_vector(out.mapOpGroups, [](const SmallVector<Value> &grp) {
      return ValueRange(grp);
    });
  }

  static LogicalResult
  fold(PatternRewriter &rewriter, const Input &in, Output &out, Operation *op, const char *aspect) {
    if (in.mapOpGroups.empty()) {
      // No affine map operands so nothing to do
      return failure();
    }

    assert(in.mapOpGroups.size() <= in.paramsOfStructTy.size());
    assert(std::cmp_equal(in.mapOpGroups.size(), in.dimsPerGroup.size()));

    size_t idx = 0; // index in `mapOpGroups`, i.e., the number of AffineMapAttr encountered
    for (Attribute sizeAttr : in.paramsOfStructTy) {
      if (AffineMapAttr m = dyn_cast<AffineMapAttr>(sizeAttr)) {
        ValueRange currMapOps = in.mapOpGroups[idx++];
        LLVM_DEBUG(
            llvm::dbgs() << "[AffineMapFolder] currMapOps: " << debug::toStringList(currMapOps)
                         << '\n'
        );
        SmallVector<OpFoldResult> currMapOpsCast = getAsOpFoldResult(currMapOps);
        LLVM_DEBUG(
            llvm::dbgs() << "[AffineMapFolder] currMapOps as fold results: "
                         << debug::toStringList(currMapOpsCast) << '\n'
        );
        if (auto constOps = Step4_InstantiateAffineMaps::getConstantIntValues(currMapOpsCast)) {
          SmallVector<Attribute> result;
          bool hasPoison = false; // indicates divide by 0 or mod by <1
          auto constAttrs = llvm::map_to_vector(*constOps, [&rewriter](int64_t v) -> Attribute {
            return rewriter.getIndexAttr(v);
          });
          LogicalResult foldResult = m.getAffineMap().constantFold(constAttrs, result, &hasPoison);
          if (hasPoison) {
            // Diagnostic remark: could be removed for release builds if too noisy
            op->emitRemark()
                .append(
                    "Cannot fold affine_map for ", aspect, ' ', out.paramsOfStructTy.size(),
                    " due to divide by 0 or modulus with negative divisor"
                )
                .report();
            return failure();
          }
          if (failed(foldResult)) {
            // Diagnostic remark: could be removed for release builds if too noisy
            op->emitRemark()
                .append(
                    "Folding affine_map for ", aspect, ' ', out.paramsOfStructTy.size(), " failed"
                )
                .report();
            return failure();
          }
          if (result.size() != 1) {
            // Diagnostic remark: could be removed for release builds if too noisy
            op->emitRemark()
                .append(
                    "Folding affine_map for ", aspect, ' ', out.paramsOfStructTy.size(),
                    " produced ", result.size(), " results but expected 1"
                )
                .report();
            return failure();
          }
          assert(!llvm::isa<AffineMapAttr>(result[0]) && "not converted");
          out.paramsOfStructTy.push_back(result[0]);
          continue;
        }
        // If affine but not foldable, preserve the map ops
        out.mapOpGroups.emplace_back(currMapOps);
        out.dimsPerGroup.push_back(in.dimsPerGroup[idx - 1]); // idx was already incremented
      }
      // If not affine and foldable, preserve the original
      out.paramsOfStructTy.push_back(sizeAttr);
    }
    assert(idx == in.mapOpGroups.size() && "all affine_map not processed");
    assert(
        in.paramsOfStructTy.size() == out.paramsOfStructTy.size() &&
        "produced wrong number of dimensions"
    );

    return success();
  }
};

/// At CreateArrayOp, instantiate ArrayType parameterized with affine_map dimension size(s)
class InstantiateAtCreateArrayOp final : public OpRewritePattern<CreateArrayOp> {
  [[maybe_unused]]
  ConversionTracker &tracker_;

public:
  InstantiateAtCreateArrayOp(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx), tracker_(tracker) {}

  LogicalResult matchAndRewrite(CreateArrayOp op, PatternRewriter &rewriter) const override {
    ArrayType oldResultType = op.getType();

    AffineMapFolder::Output out;
    AffineMapFolder::Input in = {
        op.getMapOperands(),
        op.getNumDimsPerMapAttr(),
        oldResultType.getDimensionSizes(),
    };
    if (failed(AffineMapFolder::fold(rewriter, in, out, op, "array dimension"))) {
      return failure();
    }

    ArrayType newResultType = ArrayType::get(oldResultType.getElementType(), out.paramsOfStructTy);
    if (newResultType == oldResultType) {
      return failure(); // nothing changed
    }
    // ASSERT: folding only preserves the original Attribute or converts affine to integer
    assert(tracker_.isLegalConversion(oldResultType, newResultType, "InstantiateAtCreateArrayOp"));
    LLVM_DEBUG(
        llvm::dbgs() << "[InstantiateAtCreateArrayOp] instantiating " << oldResultType << " as "
                     << newResultType << " in \"" << op << "\"\n"
    );
    replaceOpWithNewOp<CreateArrayOp>(
        rewriter, op, newResultType, AffineMapFolder::getConvertedMapOpGroups(out), out.dimsPerGroup
    );
    return success();
  }
};

/// Instantiate parameterized StructType resulting from CallOp targeting "compute()" functions.
class InstantiateAtCallOpCompute final : public OpRewritePattern<CallOp> {
  ConversionTracker &tracker_;

public:
  InstantiateAtCallOpCompute(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx), tracker_(tracker) {}

  LogicalResult matchAndRewrite(CallOp op, PatternRewriter &rewriter) const override {
    if (!op.calleeIsStructCompute()) {
      // this pattern only applies when the callee is "compute()" within a struct
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "[InstantiateAtCallOpCompute] target: " << op.getCallee() << '\n');
    StructType oldRetTy = op.getSingleResultTypeOfCompute();
    LLVM_DEBUG(llvm::dbgs() << "[InstantiateAtCallOpCompute]   oldRetTy: " << oldRetTy << '\n');
    ArrayAttr params = oldRetTy.getParams();
    if (isNullOrEmpty(params)) {
      // nothing to do if the StructType is not parameterized
      return failure();
    }

    AffineMapFolder::Output out;
    AffineMapFolder::Input in = {
        op.getMapOperands(),
        op.getNumDimsPerMapAttr(),
        params.getValue(),
    };
    if (!in.mapOpGroups.empty()) {
      // If there are affine map operands, attempt to fold them to a constant.
      if (failed(AffineMapFolder::fold(rewriter, in, out, op, "struct parameter"))) {
        return failure();
      }
      LLVM_DEBUG({
        llvm::dbgs() << "[InstantiateAtCallOpCompute]   folded affine_map in result type params\n";
      });
    } else {
      // If there are no affine map operands, attempt to refine the result type of the CallOp using
      // the function argument types and the type of the target function.
      auto callArgTypes = op.getArgOperands().getTypes();
      if (callArgTypes.empty()) {
        // no refinement possible if no function arguments
        return failure();
      }
      if (calleeReferencesTemplateParam(op)) {
        return failure();
      }
      SymbolTableCollection tables;
      auto lookupRes = lookupTopLevelSymbol<FuncDefOp>(tables, op.getCalleeAttr(), op);
      if (failed(lookupRes)) {
        return failure();
      }
      if (failed(instantiateViaTargetType(in, out, callArgTypes, lookupRes->get()))) {
        return failure();
      }
      LLVM_DEBUG({
        llvm::dbgs() << "[InstantiateAtCallOpCompute]   propagated instantiations via symrefs in "
                        "result type params: "
                     << debug::toStringList(out.paramsOfStructTy) << '\n';
      });
    }

    StructType newRetTy = StructType::get(oldRetTy.getNameRef(), out.paramsOfStructTy);
    LLVM_DEBUG(llvm::dbgs() << "[InstantiateAtCallOpCompute]   newRetTy: " << newRetTy << '\n');
    if (newRetTy == oldRetTy) {
      return failure(); // nothing changed
    }
    // The `newRetTy` is computed via instantiateViaTargetType() which can only preserve the
    // original Attribute or convert to a concrete attribute via the unification process. Thus, if
    // the conversion here is illegal it means there is a type conflict within the LLZK code that
    // prevents instantiation of the struct with the requested type.
    if (!tracker_.isLegalConversion(oldRetTy, newRetTy, "InstantiateAtCallOpCompute")) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag.append(
            "result type mismatch: due to struct instantiation, expected type ", newRetTy,
            ", but found ", oldRetTy
        );
      });
    }
    LLVM_DEBUG(llvm::dbgs() << "[InstantiateAtCallOpCompute] replaced " << op);
    CallOp newOp = replaceOpWithNewOp<CallOp>(
        rewriter, op, TypeRange {newRetTy}, op.getCallee(),
        AffineMapFolder::getConvertedMapOpGroups(out), out.dimsPerGroup, op.getArgOperands()
    );
    (void)newOp; // tell compiler it's intentionally unused in release builds
    LLVM_DEBUG(llvm::dbgs() << " with " << newOp << '\n');
    return success();
  }

private:
  /// Use the type of the target function to propagate instantiation knowledge from the function
  /// argument types to the function return type in the CallOp.
  inline LogicalResult instantiateViaTargetType(
      const AffineMapFolder::Input &in, AffineMapFolder::Output &out,
      OperandRange::type_range callArgTypes, FuncDefOp targetFunc
  ) const {
    assert(targetFunc.isStructCompute()); // since `op.calleeIsStructCompute()`
    ArrayAttr targetResTyParams = targetFunc.getSingleResultTypeOfCompute().getParams();
    assert(!isNullOrEmpty(targetResTyParams)); // same cardinality as `in.paramsOfStructTy`
    assert(in.paramsOfStructTy.size() == targetResTyParams.size()); // verifier ensures this

    if (llvm::all_of(in.paramsOfStructTy, isConcreteAttr<>)) {
      // Nothing can change if everything is already concrete
      return failure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << '[' << __FUNCTION__ << ']'
                   << " call arg types: " << debug::toStringList(callArgTypes) << '\n';
      llvm::dbgs() << '[' << __FUNCTION__ << ']' << " target func arg types: "
                   << debug::toStringList(targetFunc.getArgumentTypes()) << '\n';
      llvm::dbgs() << '[' << __FUNCTION__ << ']'
                   << " struct params @ call: " << debug::toStringList(in.paramsOfStructTy) << '\n';
      llvm::dbgs() << '[' << __FUNCTION__ << ']'
                   << " target struct params: " << debug::toStringList(targetResTyParams) << '\n';
    });

    UnificationMap unifications;
    bool unifies = typeListsUnify(targetFunc.getArgumentTypes(), callArgTypes, {}, &unifications);
    (void)unifies; // tell compiler it's intentionally unused in builds without assertions
    assert(unifies && "should have been checked by verifiers");

    LLVM_DEBUG({
      llvm::dbgs() << '[' << __FUNCTION__ << ']'
                   << " unifications of arg types: " << debug::toStringList(unifications) << '\n';
    });

    // Check for LHS SymRef (i.e., from the target function) that have RHS concrete Attributes (i.e.
    // from the call argument types) without any struct parameters (because the type with concrete
    // struct parameters will be used to instantiate the target struct rather than the fully
    // flattened struct type resulting in type mismatch of the callee to target) and perform those
    // replacements in the `targetFunc` return type to produce the new result type for the CallOp.
    SmallVector<Attribute> newReturnStructParams = llvm::map_to_vector(
        llvm::zip_equal(targetResTyParams.getValue(), in.paramsOfStructTy),
        [&unifications](std::tuple<Attribute, Attribute> p) {
      Attribute fromCall = std::get<1>(p);
      // Preserve attributes that are already concrete at the call site. Otherwise attempt to lookup
      // non-parameterized concrete unification for the target struct parameter symbol.
      if (!isConcreteAttr(fromCall)) {
        Attribute fromTgt = std::get<0>(p);
        LLVM_DEBUG({
          llvm::dbgs() << "[instantiateViaTargetType]   fromCall = " << fromCall << '\n';
          llvm::dbgs() << "[instantiateViaTargetType]   fromTgt = " << fromTgt << '\n';
        });
        assert(llvm::isa<SymbolRefAttr>(fromTgt));
        auto it = unifications.find(std::make_pair(llvm::cast<SymbolRefAttr>(fromTgt), Side::LHS));
        if (it != unifications.end()) {
          Attribute unifiedAttr = it->second;
          LLVM_DEBUG({
            llvm::dbgs() << "[instantiateViaTargetType]   unifiedAttr = " << unifiedAttr << '\n';
          });
          if (unifiedAttr && isConcreteAttr<false>(unifiedAttr)) {
            return unifiedAttr;
          }
        }
      }
      return fromCall;
    }
    );

    out.paramsOfStructTy = newReturnStructParams;
    assert(out.paramsOfStructTy.size() == in.paramsOfStructTy.size() && "post-condition");
    assert(out.mapOpGroups.empty() && "post-condition");
    assert(out.dimsPerGroup.empty() && "post-condition");
    return success();
  }
};

LogicalResult run(ModuleOp modOp, ConversionTracker &tracker) {
  MLIRContext *ctx = modOp.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<
      InstantiateAtCreateArrayOp, // CreateArrayOp
      InstantiateAtCallOpCompute  // CallOp, targeting struct "compute()"
      >(ctx, tracker);

  return applyAndFoldGreedily(modOp, tracker, std::move(patterns));
}

} // namespace Step4_InstantiateAffineMaps

namespace Step5_PropagateTypes {

/// Update the array element type by looking at the values stored into it from uses.
class UpdateNewArrayElemFromWrite final : public OpRewritePattern<CreateArrayOp> {
  ConversionTracker &tracker_;

public:
  UpdateNewArrayElemFromWrite(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  LogicalResult matchAndRewrite(CreateArrayOp op, PatternRewriter &rewriter) const override {
    Value createResult = op.getResult();
    ArrayType createResultType = dyn_cast<ArrayType>(createResult.getType());
    assert(createResultType && "CreateArrayOp must produce ArrayType");
    Type oldResultElemType = createResultType.getElementType();

    // Look for WriteArrayOp where the array reference is the result of the CreateArrayOp and the
    // element type is different.
    Type newResultElemType = nullptr;
    for (Operation *user : createResult.getUsers()) {
      if (WriteArrayOp writeOp = dyn_cast<WriteArrayOp>(user)) {
        if (writeOp.getArrRef() != createResult) {
          continue;
        }
        Type writeRValueType = writeOp.getRvalue().getType();
        if (writeRValueType == oldResultElemType) {
          continue;
        }
        if (newResultElemType && newResultElemType != writeRValueType) {
          LLVM_DEBUG(
              llvm::dbgs()
              << "[UpdateNewArrayElemFromWrite] multiple possible element types for CreateArrayOp "
              << newResultElemType << " vs " << writeRValueType << '\n'
          );
          return failure();
        }
        newResultElemType = writeRValueType;
      }
    }
    if (!newResultElemType) {
      // no replacement type found
      return failure();
    }
    if (!tracker_.isLegalConversion(
            oldResultElemType, newResultElemType, "UpdateNewArrayElemFromWrite"
        )) {
      return failure();
    }
    ArrayType newType = createResultType.cloneWith(newResultElemType);
    rewriter.modifyOpInPlace(op, [&createResult, &newType]() { createResult.setType(newType); });
    LLVM_DEBUG(
        llvm::dbgs() << "[UpdateNewArrayElemFromWrite] updated result type of " << op << '\n'
    );
    return success();
  }
};

namespace {

LogicalResult updateArrayElemFromArrAccessOp(
    ArrayAccessOpInterface op, Type scalarElemTy, ConversionTracker &tracker,
    PatternRewriter &rewriter
) {
  ArrayType oldArrType = op.getArrRefType();
  if (oldArrType.getElementType() == scalarElemTy) {
    return failure(); // no change needed
  }
  ArrayType newArrType = oldArrType.cloneWith(scalarElemTy);
  if (oldArrType == newArrType ||
      !tracker.isLegalConversion(oldArrType, newArrType, "updateArrayElemFromArrAccessOp")) {
    return failure();
  }
  rewriter.modifyOpInPlace(op, [&op, &newArrType]() { op.getArrRef().setType(newArrType); });
  LLVM_DEBUG(
      llvm::dbgs() << "[updateArrayElemFromArrAccessOp] updated base array type in " << op << '\n'
  );
  return success();
}

} // namespace

class UpdateArrayElemFromArrWrite final : public OpRewritePattern<WriteArrayOp> {
  ConversionTracker &tracker_;

public:
  UpdateArrayElemFromArrWrite(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  LogicalResult matchAndRewrite(WriteArrayOp op, PatternRewriter &rewriter) const override {
    return updateArrayElemFromArrAccessOp(op, op.getRvalue().getType(), tracker_, rewriter);
  }
};

class UpdateArrayElemFromArrRead final : public OpRewritePattern<ReadArrayOp> {
  ConversionTracker &tracker_;

public:
  UpdateArrayElemFromArrRead(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  LogicalResult matchAndRewrite(ReadArrayOp op, PatternRewriter &rewriter) const override {
    return updateArrayElemFromArrAccessOp(op, op.getResult().getType(), tracker_, rewriter);
  }
};

/// Update the type of MemberDefOp instances by checking the updated types from MemberWriteOp.
class UpdateMemberDefTypeFromWrite final : public OpRewritePattern<MemberDefOp> {
  ConversionTracker &tracker_;

public:
  UpdateMemberDefTypeFromWrite(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  LogicalResult matchAndRewrite(MemberDefOp op, PatternRewriter &rewriter) const override {
    // Find all uses of the member symbol name within its parent struct.
    StructDefOp parentRes = getParentOfType<StructDefOp>(op);
    assert(parentRes && "MemberDefOp parent is always StructDefOp"); // per ODS def

    // If the symbol is used by a MemberWriteOp with a different result type then change
    // the type of the MemberDefOp to match the MemberWriteOp result type.
    Type newType = nullptr;
    if (auto memberUsers = llzk::getSymbolUses(op, parentRes)) {
      std::optional<Location> newTypeLoc = std::nullopt;
      for (SymbolTable::SymbolUse symUse : memberUsers.value()) {
        if (MemberWriteOp writeOp = llvm::dyn_cast<MemberWriteOp>(symUse.getUser())) {
          Type writeToType = writeOp.getVal().getType();
          LLVM_DEBUG(llvm::dbgs() << "[UpdateMemberDefTypeFromWrite] checking " << writeOp << '\n');
          if (!newType) {
            // If a new type has not yet been discovered, store the new type.
            newType = writeToType;
            newTypeLoc = writeOp.getLoc();
          } else if (writeToType != newType) {
            // Typically, there will only be one write for each member of a struct but do not rely
            // on that assumption. If multiple writes with a different types A and B are found where
            // A->B is a legal conversion (i.e., more concrete unification), then it is safe to use
            // type B with the assumption that the write with type A will be updated by another
            // pattern to also use type B.
            if (!tracker_.isLegalConversion(writeToType, newType, "UpdateMemberDefTypeFromWrite")) {
              if (tracker_.isLegalConversion(
                      newType, writeToType, "UpdateMemberDefTypeFromWrite"
                  )) {
                // 'writeToType' is the more concrete type
                newType = writeToType;
                newTypeLoc = writeOp.getLoc();
              } else {
                // Give an error if the types are incompatible.
                return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
                  diag.append(
                      "Cannot update type of '", MemberDefOp::getOperationName(),
                      "' because there are multiple '", MemberWriteOp::getOperationName(),
                      "' with different value types"
                  );
                  if (newTypeLoc) {
                    diag.attachNote(newTypeLoc).append("type written here is ", newType);
                  }
                  diag.attachNote(writeOp.getLoc()).append("type written here is ", writeToType);
                });
              }
            }
          }
        }
      }
    }
    if (!newType || newType == op.getType()) {
      return failure(); // nothing changed
    }
    if (!tracker_.isLegalConversion(op.getType(), newType, "UpdateMemberDefTypeFromWrite")) {
      return failure();
    }
    rewriter.modifyOpInPlace(op, [&op, &newType]() { op.setType(newType); });
    LLVM_DEBUG(llvm::dbgs() << "[UpdateMemberDefTypeFromWrite] updated type of " << op << '\n');
    return success();
  }
};

namespace {

SmallVector<std::unique_ptr<Region>> moveRegions(Operation *op) {
  SmallVector<std::unique_ptr<Region>> newRegions;
  for (Region &region : op->getRegions()) {
    auto newRegion = std::make_unique<Region>();
    newRegion->takeBody(region);
    newRegions.push_back(std::move(newRegion));
  }
  return newRegions;
}

} // namespace

/// Updates the result type in Ops with the InferTypeOpAdaptor trait including ReadArrayOp,
/// ExtractArrayOp, etc.
class UpdateInferredResultTypes final : public OpTraitRewritePattern<OpTrait::InferTypeOpAdaptor> {
  ConversionTracker &tracker_;

public:
  UpdateInferredResultTypes(MLIRContext *ctx, ConversionTracker &tracker)
      : OpTraitRewritePattern(ctx, 6), tracker_(tracker) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    SmallVector<Type, 1> inferredResultTypes;
    InferTypeOpInterface retTypeFn = llvm::cast<InferTypeOpInterface>(op);
    LogicalResult result = retTypeFn.inferReturnTypes(
        op->getContext(), op->getLoc(), op->getOperands(), op->getRawDictionaryAttrs(),
        op->getPropertiesStorage(), op->getRegions(), inferredResultTypes
    );
    if (failed(result)) {
      return failure();
    }
    if (op->getResultTypes() == inferredResultTypes) {
      return failure(); // nothing changed
    }
    if (!tracker_.areLegalConversions(
            op->getResultTypes(), inferredResultTypes, "UpdateInferredResultTypes"
        )) {
      return failure();
    }

    // Move nested region bodies and replace the original op with the updated types list.
    LLVM_DEBUG(llvm::dbgs() << "[UpdateInferredResultTypes] replaced " << *op);
    SmallVector<std::unique_ptr<Region>> newRegions = moveRegions(op);
    Operation *newOp = rewriter.create(
        op->getLoc(), op->getName().getIdentifier(), op->getOperands(), inferredResultTypes,
        op->getAttrs(), op->getSuccessors(), newRegions
    );
    rewriter.replaceOp(op, newOp);
    LLVM_DEBUG(llvm::dbgs() << " with " << *newOp << '\n');
    return success();
  }
};

/// Update FuncDefOp return type by checking the updated types from ReturnOp.
class UpdateFuncTypeFromReturn final : public OpRewritePattern<FuncDefOp> {
  ConversionTracker &tracker_;

public:
  UpdateFuncTypeFromReturn(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  LogicalResult matchAndRewrite(FuncDefOp op, PatternRewriter &rewriter) const override {
    Region &body = op.getFunctionBody();
    if (body.empty()) {
      return failure();
    }
    ReturnOp retOp = llvm::dyn_cast<ReturnOp>(body.back().getTerminator());
    assert(retOp && "final op in body region must be return");
    OperandRange::type_range tyFromReturnOp = retOp.getOperands().getTypes();

    FunctionType oldFuncTy = op.getFunctionType();
    if (oldFuncTy.getResults() == tyFromReturnOp) {
      return failure(); // nothing changed
    }
    if (!tracker_.areLegalConversions(
            oldFuncTy.getResults(), tyFromReturnOp, "UpdateFuncTypeFromReturn"
        )) {
      return failure();
    }

    rewriter.modifyOpInPlace(op, [&]() {
      op.setFunctionType(rewriter.getFunctionType(oldFuncTy.getInputs(), tyFromReturnOp));
    });
    LLVM_DEBUG(
        llvm::dbgs() << "[UpdateFuncTypeFromReturn] changed " << op.getSymName() << " from "
                     << oldFuncTy << " to " << op.getFunctionType() << '\n'
    );
    return success();
  }
};

/// Update a free-function call's result types from its target definition while preserving ordered
/// explicit template arguments. Struct methods are excluded: they return a `StructType` or no
/// value, and copying a method's declaration type back to its call could reintroduce a
/// still-parameterized type after caller-side instantiation.
class UpdateFreeFuncCallOpTypes final : public OpRewritePattern<CallOp> {
  ConversionTracker &tracker_;

public:
  UpdateFreeFuncCallOpTypes(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  LogicalResult matchAndRewrite(CallOp op, PatternRewriter &rewriter) const override {
    if (calleeReferencesTemplateParam(op)) {
      return failure();
    }
    SymbolTableCollection tables;
    auto lookupRes = lookupTopLevelSymbol<FuncDefOp>(tables, op.getCalleeAttr(), op);
    if (failed(lookupRes)) {
      return failure();
    }
    FuncDefOp targetFunc = lookupRes->get();
    if (targetFunc.isInStruct()) {
      // this pattern only applies when the callee is NOT in a struct
      return failure();
    }
    if (op.getResultTypes() == targetFunc.getFunctionType().getResults()) {
      return failure(); // nothing changed
    }
    if (!tracker_.areLegalConversions(
            op.getResultTypes(), targetFunc.getFunctionType().getResults(),
            "UpdateFreeFuncCallOpTypes"
        )) {
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "[UpdateFreeFuncCallOpTypes] replaced " << op);
    ArrayAttr templateParamsAttr = op.getTemplateParamsAttr();
    ArrayRef<Attribute> templateParams =
        templateParamsAttr ? templateParamsAttr.getValue() : ArrayRef<Attribute>();
    CallOp newOp =
        replaceOpWithNewOp<CallOp>(rewriter, op, targetFunc, op.getArgOperands(), templateParams);
    (void)newOp; // tell compiler it's intentionally unused in release builds
    LLVM_DEBUG(llvm::dbgs() << " with " << newOp << '\n');
    return success();
  }
};

namespace {

LogicalResult updateMemberRefValFromMemberDef(
    MemberRefOpInterface op, ConversionTracker &tracker, PatternRewriter &rewriter
) {
  SymbolTableCollection tables;
  auto def = op.getMemberDefOp(tables);
  if (failed(def)) {
    return failure();
  }
  Type oldResultType = op.getVal().getType();
  Type newResultType = def->get().getType();
  if (oldResultType == newResultType ||
      !tracker.isLegalConversion(oldResultType, newResultType, "updateMemberRefValFromMemberDef")) {
    return failure();
  }
  rewriter.modifyOpInPlace(op, [&op, &newResultType]() { op.getVal().setType(newResultType); });
  LLVM_DEBUG(
      llvm::dbgs() << "[updateMemberRefValFromMemberDef] updated value type in " << op << '\n'
  );
  return success();
}

} // namespace

/// Update the type of MemberReadOp result based on updated types from MemberDefOp.
class UpdateMemberReadValFromDef final : public OpRewritePattern<MemberReadOp> {
  ConversionTracker &tracker_;

public:
  UpdateMemberReadValFromDef(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  LogicalResult matchAndRewrite(MemberReadOp op, PatternRewriter &rewriter) const override {
    return updateMemberRefValFromMemberDef(op, tracker_, rewriter);
  }
};

/// Update the type of MemberWriteOp value based on updated types from MemberDefOp.
class UpdateMemberWriteValFromDef final : public OpRewritePattern<MemberWriteOp> {
  ConversionTracker &tracker_;

public:
  UpdateMemberWriteValFromDef(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  LogicalResult matchAndRewrite(MemberWriteOp op, PatternRewriter &rewriter) const override {
    return updateMemberRefValFromMemberDef(op, tracker_, rewriter);
  }
};

LogicalResult run(ModuleOp modOp, ConversionTracker &tracker) {
  MLIRContext *ctx = modOp.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<
      // Benefit of this one must be higher than rules that would propagate the type in the opposite
      // direction (ex: `UpdateArrayElemFromArrRead`) else the greedy conversion would not converge.
      //  benefit = 6
      UpdateInferredResultTypes, // OpTrait::InferTypeOpAdaptor (ReadArrayOp, ExtractArrayOp)
      //  benefit = 3
      UpdateFreeFuncCallOpTypes,    // CallOp, targeting non-struct functions
      UpdateFuncTypeFromReturn,     // FuncDefOp
      UpdateNewArrayElemFromWrite,  // CreateArrayOp
      UpdateArrayElemFromArrRead,   // ReadArrayOp
      UpdateArrayElemFromArrWrite,  // WriteArrayOp
      UpdateMemberDefTypeFromWrite, // MemberDefOp
      UpdateMemberReadValFromDef,   // MemberReadOp
      UpdateMemberWriteValFromDef   // MemberWriteOp
      >(ctx, tracker);

  return applyAndFoldGreedily(modOp, tracker, std::move(patterns));
}
} // namespace Step5_PropagateTypes

namespace Step6_Cleanup {

struct FromKeepSet : public CleanupBase {
  using CleanupBase::CleanupBase;

  /// Return `true` iff the given free function or struct definition still has unresolved template
  /// symbol bindings.
  static bool hasTemplateSymbolBindings(Operation *op) {
    if (StructDefOp sdef = llvm::dyn_cast<StructDefOp>(op)) {
      return sdef.hasTemplateSymbolBindings();
    }
    if (llvm::isa<function::FuncDefOp>(op)) {
      if (TemplateOp parent = getParentOfType<TemplateOp>(op)) {
        return parent.hasConstOps<TemplateSymbolBindingOpInterface>();
      }
    }
    return false;
  }

  /// Erase all cleanup-candidate definitions that are not reachable (via calls, types, or symbol
  /// usage) from one of the given roots or from some global def (since this pass does not remove
  /// global definitions, any symbols reachable from them must not be removed).
  LogicalResult eraseUnreachableFrom(ArrayRef<SymbolOpInterface> keep) {
    // Initialize roots from the given symbol definitions.
    SetVector<SymbolOpInterface> roots(keep.begin(), keep.end());
    // Add GlobalDefOp to the set of roots.
    rootMod.walk([&roots](global::GlobalDefOp gdef) { roots.insert(gdef); });

    // Use a SymbolDefTree to find all Symbol defs reachable from one of the root nodes. Then
    // collect all Symbol uses reachable from those def nodes. These are the symbols that should
    // be preserved. All other symbol defs should be removed.
    DenseSet<Operation *> defsToKeep;
    llvm::df_iterator_default_set<const SymbolUseGraphNode *> symbolsToKeep;
    for (size_t i = 0; i < roots.size(); ++i) { // iterate for safe insertion
      SymbolOpInterface keepRoot = roots[i];
      LLVM_DEBUG({ llvm::dbgs() << "[EraseUnreachable] root: " << keepRoot << '\n'; });
      const SymbolDefTreeNode *keepRootNode = defTree.lookupNode(keepRoot);
      assert(keepRootNode && "every symbol def must be in the def tree");
      for (const SymbolDefTreeNode *reachableDefNode : llvm::depth_first(keepRootNode)) {
        LLVM_DEBUG({
          llvm::dbgs() << "[EraseUnreachable] can reach: " << reachableDefNode->getOp() << '\n';
        });
        if (SymbolOpInterface reachableDef = reachableDefNode->getOp()) {
          if (isErasableDefinition(reachableDef.getOperation())) {
            defsToKeep.insert(reachableDef.getOperation());
          }
          // Use 'depth_first_ext()' to get all symbol uses reachable from the current Symbol def
          // node. There are no uses if the node is not in the graph. Within the loop that populates
          // 'depth_first_ext()', also check if the symbol is an erasable definition and ensure it
          // is in 'roots' so the outer loop preserves all symbols reachable from it.
          if (const SymbolUseGraphNode *useGraphNodeForDef = useGraph.lookupNode(reachableDef)) {
            for (const SymbolUseGraphNode *usedSymbolNode :
                 depth_first_ext(useGraphNodeForDef, symbolsToKeep)) {
              LLVM_DEBUG({
                llvm::dbgs() << "[EraseUnreachable]   uses symbol: "
                             << usedSymbolNode->getSymbolPath() << '\n';
              });
              // Ignore struct/template parameter symbols (before doing the lookup below because it
              // would fail anyway and then cause the "failed" case to be triggered unnecessarily).
              if (usedSymbolNode->isTemplateSymbolBinding()) {
                continue;
              }
              // If `usedSymbolNode` references an erasable definition, ensure it's considered in
              // the roots so symbols reachable from its body are preserved too.
              auto lookupRes = usedSymbolNode->lookupSymbol(tables);
              if (failed(lookupRes)) {
                LLVM_DEBUG(useGraph.dumpToDotFile());
                return failure();
              }
              //  If loaded via an IncludeOp it's not in the current AST anyway so ignore.
              if (lookupRes->viaInclude()) {
                continue;
              }
              Operation *usedOp = lookupRes->get();
              if (isErasableDefinition(usedOp)) {
                SymbolOpInterface asSymbol = llvm::cast<SymbolOpInterface>(usedOp);
                bool insertRes = roots.insert(asSymbol);
                (void)insertRes; // tell compiler it's intentionally unused in release builds
                LLVM_DEBUG({
                  if (insertRes) {
                    llvm::dbgs() << "[EraseUnreachable]  found another root: " << asSymbol << '\n';
                  }
                });
              }
            }
          }
        }
      }
    }

    SmallVector<SymbolOpInterface> toErase;
    rootMod.walk([this, &defsToKeep, &symbolsToKeep, &toErase](Operation *op) {
      if (!isErasableDefinition(op) || defsToKeep.contains(op)) {
        return;
      }
      SymbolOpInterface symOp = llvm::cast<SymbolOpInterface>(op);
      const SymbolUseGraphNode *n = this->useGraph.lookupNode(symOp);
      if (!n || !symbolsToKeep.contains(n)) {
        LLVM_DEBUG(llvm::dbgs() << "[EraseUnreachable] removing: " << symOp.getNameAttr() << '\n');
        toErase.push_back(symOp);
      }
    });
    for (SymbolOpInterface symOp : toErase) {
      symOp.erase();
    }

    return success();
  }
};

} // namespace Step6_Cleanup

class PassImpl : public llzk::polymorphic::impl::FlatteningPassBase<PassImpl> {
  using Base = FlatteningPassBase<PassImpl>;
  using Base::Base;

  /// If the cleanup mode is unspecified, default to `Preimage`.
  FlatteningCleanupMode getEffectiveCleanupMode() const {
    FlatteningCleanupMode m = cleanupMode.getValue();
    return m == FlatteningCleanupMode::Unspecified ? FlatteningCleanupMode::Preimage : m;
  }

  void runOnOperation() override {
    ModuleOp modOp = getOperation();
    if (failed(runOn(modOp))) {
      LLVM_DEBUG({
        // If the pass failed, dump the current IR.
        llvm::dbgs() << "=====================================================================\n";
        llvm::dbgs() << " Dumping module after failure of pass " << DEBUG_TYPE << '\n';
        modOp.print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
        llvm::dbgs() << "=====================================================================\n";
      });
      signalPassFailure();
    }
  }

  inline LogicalResult runOn(ModuleOp modOp) {
    FlatteningCleanupMode effectiveCleanupMode = getEffectiveCleanupMode();
    // If the cleanup mode is set to remove anything not reachable from the main struct, do an
    // initial pass to remove things that are not reachable (as an optimization) because creating
    // an instantiated version of a struct will not cause something to become reachable that was
    // not already reachable in parameterized form.
    if (effectiveCleanupMode == FlatteningCleanupMode::MainAsRoot) {
      if (failed(eraseUnreachableFromMainStruct(modOp))) {
        return failure();
      }
    }

    // Pass Manager to run some standard cleanup passes that are always beneficial:
    // - Remove templates that contain no struct or function definitions
    // - Convert templates with no constant parameters or expressions into modules
    OpPassManager universalCleanup(ModuleOp::getOperationName());
    universalCleanup.addPass(createEmptyTemplateRemovalPass());

    // Run universal cleanup first so templates without `poly.param` or `poly.expr` bindings are
    // converted to modules before specialization; templates with expressions and no parameters
    // must remain eligible for cloning.
    if (failed(runPipeline(universalCleanup, modOp))) {
      return failure();
    }

    ConversionTracker tracker(modOp);
    if (failed(Step1_InstantiateStructs::instantiateMainStruct(modOp, tracker))) {
      llvm::errs() << DEBUG_TYPE << " failed while instantiating the main struct\n";
      return failure();
    }

    unsigned loopCount = 0;
    do {
      ++loopCount;
      if (loopCount > iterationLimit) {
        llvm::errs() << DEBUG_TYPE << " exceeded the limit of " << iterationLimit
                     << " iterations!\n";
        return failure();
      }
      tracker.resetModifiedFlag();

      LLVM_DEBUG({
        llvm::dbgs() << "[FlatteningPass(count=" << loopCount
                     << ")] Running step 1: struct instantiation\n";
      });
      // Find calls to "compute()" that return a parameterized struct type and replace it to call an
      // instantiated version of the struct that has parameters replaced with the constant values.
      // Create the necessary instantiated/flattened struct in the same location as the original.
      if (failed(Step1_InstantiateStructs::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while instantiating structs in templates\n";
        return failure();
      }
      // Instantiate calls to templated functions.
      if (failed(Step2_InstantiateFunctions::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while instantiating functions in templates\n";
        return failure();
      }

      LLVM_DEBUG({
        llvm::dbgs() << "[FlatteningPass(count=" << loopCount
                     << ")] Running step 2: loop unrolling\n";
      });
      // Unroll loops with known iterations.
      if (failed(Step3_Unroll::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while unrolling loops\n";
        return failure();
      }

      LLVM_DEBUG({
        llvm::dbgs() << "[FlatteningPass(count=" << loopCount
                     << ")] Running step 3: affine maps instantiation\n";
      });
      // Instantiate affine_map parameters of StructType and ArrayType.
      if (failed(Step4_InstantiateAffineMaps::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while instantiating `affine_map` parameters\n";
        return failure();
      }

      LLVM_DEBUG({
        llvm::dbgs() << "[FlatteningPass(count=" << loopCount
                     << ")] Running step 4: type propagation\n";
      });
      // Propagate updated types using the semantics of various ops.
      if (failed(Step5_PropagateTypes::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while propagating instantiated types\n";
        return failure();
      }

      LLVM_DEBUG(if (tracker.isModified()) {
        llvm::dbgs() << "=====================================================================\n";
        llvm::dbgs() << " Dumping module between iterations of " << DEBUG_TYPE << '\n';
        modOp.print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
        llvm::dbgs() << "=====================================================================\n";
      });
    } while (tracker.isModified());

    tracker.clearPartialFuncInstantiations();

    // Run user-selected cleanup first.
    if (failed(cleanupSwitch(modOp, tracker))) {
      return failure();
    }
    // Run universal cleanup again since no-param or param-only structs may exist now.
    if (failed(runPipeline(universalCleanup, modOp))) {
      return failure();
    }

    OpPassManager allocationCleanup(ModuleOp::getOperationName());
    allocationCleanup.addPass(createRemoveUnusedDiscardableAllocationsPass(
        RemoveUnusedDiscardableAllocationsPassOptions {
            .allocatorOpName = CreateArrayOp::getOperationName().str()
        }
    ));
    return runPipeline(allocationCleanup, modOp);
  }

  // Perform cleanup according to the 'cleanupMode' option.
  LogicalResult cleanupSwitch(ModuleOp modOp, const ConversionTracker &tracker) {
    FlatteningCleanupMode effectiveCleanupMode = getEffectiveCleanupMode();
    LLVM_DEBUG({ llvm::dbgs() << "[FlatteningPass] Running step 5: cleanup "; });
    switch (effectiveCleanupMode) {
    case FlatteningCleanupMode::MainAsRoot:
      LLVM_DEBUG(llvm::dbgs() << "(main as root mode)\n");
      return eraseUnreachableFromMainStruct(modOp, false);
    case FlatteningCleanupMode::ConcreteAsRoot:
      LLVM_DEBUG(llvm::dbgs() << "(concrete definitions mode)\n");
      return eraseUnreachableFromConcreteDefinitions(modOp);
    case FlatteningCleanupMode::Preimage:
      LLVM_DEBUG(llvm::dbgs() << "(preimage mode)\n");
      return erasePreimageOfInstantiations(modOp, tracker);
    case FlatteningCleanupMode::Unspecified:
    default:
      LLVM_DEBUG(llvm::dbgs() << "(disabled)\n");
      return success();
    }
  }

  // Erase parameterized definitions that were replaced with concrete instantiations.
  LogicalResult erasePreimageOfInstantiations(ModuleOp rootMod, const ConversionTracker &tracker) {
    // TODO: The names from getInstantiatedDefinitionNames() are NOT guaranteed to be paths from the
    // "top root" and they also do not indicate a root module so there could be ambiguity. This is a
    // broader problem in the FlatteningPass itself so let's just assume, for now, that these are
    // paths from the "top root". See [LLZK-286].
    FromEraseSet cleaner(
        rootMod, getAnalysis<SymbolDefTree>(), getAnalysis<SymbolUseGraph>(),
        tracker.getInstantiatedDefinitionNames()
    );
    LogicalResult res = cleaner.eraseUnusedDefinitions();
    if (succeeded(res)) {
      LLVM_DEBUG(llvm::dbgs() << "[Cleanup(preimage)] success\n";);
      // Warn about any definitions that were instantiated but still have uses elsewhere.
      const SymbolUseGraph *useGraph = nullptr;
      rootMod->walk([this, &cleaner, &useGraph](Operation *walkedOp) {
        SymbolOpInterface op = llvm::dyn_cast<SymbolOpInterface>(walkedOp);
        if (!op || !cleaner.getTryToEraseSet().contains(op)) {
          return;
        }
        // If needed, rebuild use graph to reflect deletions.
        if (!useGraph) {
          useGraph = &getAnalysis<SymbolUseGraph>();
        }
        // If the op has any users, report the warning.
        if (useGraph->lookupNode(op)->hasPredecessor()) {
          op.emitWarning("Parameterized definition still has uses!").report();
        }
      });
    } else {
      LLVM_DEBUG(llvm::dbgs() << "[Cleanup(preimage)] failed\n";);
    }
    return res;
  }

  LogicalResult eraseUnreachableFromConcreteDefinitions(ModuleOp rootMod) {
    SmallVector<SymbolOpInterface> roots;
    rootMod.walk([&roots](Operation *op) {
      if (isErasableDefinition(op) && !Step6_Cleanup::FromKeepSet::hasTemplateSymbolBindings(op)) {
        roots.push_back(llvm::cast<SymbolOpInterface>(op));
      }
    });

    Step6_Cleanup::FromKeepSet cleaner(
        rootMod, getAnalysis<SymbolDefTree>(), getAnalysis<SymbolUseGraph>()
    );
    return cleaner.eraseUnreachableFrom(roots);
  }

  LogicalResult eraseUnreachableFromMainStruct(ModuleOp rootMod, bool emitWarning = true) {
    Step6_Cleanup::FromKeepSet cleaner(
        rootMod, getAnalysis<SymbolDefTree>(), getAnalysis<SymbolUseGraph>()
    );
    FailureOr<SymbolLookupResult<StructDefOp>> mainOpt =
        getMainInstanceDef(cleaner.tables, rootMod.getOperation());
    if (failed(mainOpt)) {
      return failure();
    }
    SymbolLookupResult<StructDefOp> main = mainOpt.value();
    if (emitWarning && !main) {
      // Emit warning if there is no main specified because all cleanup-candidate definitions not
      // reachable from global defs may be removed.
      rootMod.emitWarning()
          .append(
              "using option '", cleanupMode.getArgStr(), '=',
              stringifyFlatteningCleanupMode(FlatteningCleanupMode::MainAsRoot), "' with no \"",
              MAIN_ATTR_NAME,
              "\" attribute on the top-level module may remove all cleanup-candidate definitions!"
          )
          .report();
    }
    SmallVector<SymbolOpInterface> roots;
    if (main) {
      roots.push_back(*main);
    }
    return cleaner.eraseUnreachableFrom(roots);
  }
};

} // namespace
