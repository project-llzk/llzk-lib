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
#include "llzk/Dialect/LLZK/IR/Ops.h"
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

/// Emit diagnostics that were collected while converting a cloned body, rebasing placeholder notes
/// onto the call site that triggered the clone.
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
  /// Tracks if some step performed a modification of the code such that another pass should be run.
  bool modified;
  /// Maps original remote (i.e., use site) type to new remote type.
  /// Note: The keys are always parameterized StructType and the values are no-parameter StructType.
  DenseMap<StructType, StructType> structInstantiations;
  /// Contains the reverse of mappings in `structInstantiations` for use in legal conversion check.
  DenseMap<StructType, StructType> reverseInstantiations;
  /// Tracks original free function definitions for which instantiated clones were created.
  DenseSet<SymbolRefAttr> funcInstantiations;
  /// Maps new remote type (i.e., the values in 'structInstantiations') to a list of Diagnostic
  /// to report at the location(s) of the compute() that causes the instantiation to the StructType.
  DenseMap<StructType, SmallVector<Diagnostic>> delayedDiagnostics;

public:
  /// Return whether the current flattening iteration has changed the IR.
  bool isModified() const { return modified; }

  /// Clear the per-iteration modification flag before starting the next iteration.
  void resetModifiedFlag() { modified = false; }

  /// Merge the modification status from one rewrite step into the iteration state.
  void updateModifiedFlag(bool currStepModified) { modified |= currStepModified; }

  /// Record a struct instantiation from the original use-site type to its cloned replacement type.
  void recordInstantiation(StructType oldType, StructType newType) {
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

  /// Return the instantiated type of the given StructType, if any.
  std::optional<StructType> getInstantiation(StructType oldType) const {
    auto cachedResult = structInstantiations.find(oldType);
    if (cachedResult != structInstantiations.end()) {
      return cachedResult->second;
    }
    return std::nullopt;
  }

  /// Record that the given free function was instantiated.
  void recordInstantiation(SymbolRefAttr funcName) {
    funcInstantiations.insert(funcName);
    modified = true;
  }

  /// Collect the fully-qualified names of all structs and free functions that were instantiated.
  DenseSet<SymbolRefAttr> getInstantiatedDefinitionNames() const {
    DenseSet<SymbolRefAttr> instantiatedNames = funcInstantiations;
    for (const auto &[origRemoteTy, _] : structInstantiations) {
      instantiatedNames.insert(origRemoteTy.getNameRef());
    }
    return instantiatedNames;
  }

  /// Emit diagnostics delayed until a compute call has been rewritten to the instantiated type.
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

  /// Return the mutable diagnostic queue associated with `newType`.
  SmallVector<Diagnostic> &delayedDiagnosticSet(StructType newType) {
    return delayedDiagnostics[newType];
  }

  /// Check if the type conversion is legal, i.e., the new type unifies with and is more concrete
  /// than the old type with additional allowance for the results of struct flattening conversions.
  bool isLegalConversion(Type oldType, Type newType, const char *patName) const {
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
        llvm::dbgs() << "[" << patName << "] Cannot replace old type " << oldType
                     << " with new type " << newType
                     << " because it does not define a compatible and more concrete type.\n";
    );
    return false;
  }

  /// Check whether every corresponding pair in `oldTypes` and `newTypes` is a legal conversion.
  template <typename T, typename U>
  inline bool areLegalConversions(T oldTypes, U newTypes, const char *patName) const {
    return llvm::all_of(
        llvm::zip_equal(oldTypes, newTypes), [this, &patName](std::tuple<Type, Type> oldThenNew) {
      return this->isLegalConversion(std::get<0>(oldThenNew), std::get<1>(oldThenNew), patName);
    }
    );
  }
};

/// Base conversion pattern for ops that reference template symbols by attribute and rewrite only
/// when that symbol has a concrete instantiation value of one of `HandledAttrs`.
template <typename Impl, typename Op, typename... HandledAttrs>
class SymbolUserHelper : public OpConversionPattern<Op> {
private:
  const DenseMap<Attribute, Attribute> &paramNameToValue;

  /// Construct the CRTP helper with the template binding map used for symbol lookups.
  SymbolUserHelper(
      TypeConverter &converter, MLIRContext *ctx, unsigned patternBenefit,
      const DenseMap<Attribute, Attribute> &paramNameToInstantiatedValue
  )
      : OpConversionPattern<Op>(converter, ctx, patternBenefit),
        paramNameToValue(paramNameToInstantiatedValue) {}

public:
  using OpAdaptor = typename mlir::OpConversionPattern<Op>::OpAdaptor;

  /// Return the attribute on `op` that should be looked up in the instantiation map.
  virtual Attribute getNameAttr(Op) const = 0;

  /// Report a type-specific fallback diagnostic for instantiated values not handled by `Impl`.
  virtual LogicalResult handleDefaultRewrite(
      Attribute, Op op, OpAdaptor, ConversionPatternRewriter &, Attribute a
  ) const {
    return op->emitOpError().append("expected value with type ", op.getType(), " but found ", a);
  }

  /// Dispatch an instantiated symbol value to the concrete `Impl::handleRewrite` overload.
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

/// Rewrite `poly.const_read` uses in cloned bodies to concrete constants when their referenced
/// template parameter has been instantiated.
class ClonedBodyConstReadOpPattern
    : public SymbolUserHelper<
          ClonedBodyConstReadOpPattern, ConstReadOp, IntegerAttr, FeltConstAttr> {
  SmallVector<Diagnostic> &diagnostics;

  using super =
      SymbolUserHelper<ClonedBodyConstReadOpPattern, ConstReadOp, IntegerAttr, FeltConstAttr>;

public:
  /// Construct the const-read conversion pattern and collect delayed diagnostics in
  /// `instantiationDiagnostics`.
  ClonedBodyConstReadOpPattern(
      TypeConverter &converter, MLIRContext *ctx,
      const DenseMap<Attribute, Attribute> &paramNameToInstantiatedValue,
      SmallVector<Diagnostic> &instantiationDiagnostics
  )
      // benefit>0 so this applies instead of GeneralTypeReplacePattern<ConstReadOp>
      : super(converter, ctx, /*patternBenefit=*/1, paramNameToInstantiatedValue),
        diagnostics(instantiationDiagnostics) {}

  /// Use the referenced constant symbol as the lookup key.
  Attribute getNameAttr(ConstReadOp op) const override { return op.getConstNameAttr(); }

  /// Replace an integer-backed template value with the constant op matching the converted result
  /// type.
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

  /// Replace an already-felt template value with a felt constant.
  LogicalResult handleRewrite(
      Attribute, ConstReadOp op, OpAdaptor, ConversionPatternRewriter &rewriter, FeltConstAttr a
  ) const {
    replaceOpWithNewOp<FeltConstantOp>(rewriter, op, a);
    return success();
  }
};

/// Patterns can use this listener and call notifyMatchFailure(..) for failures where the entire
/// pass must fail, i.e., where instantiation would introduce an illegal type conversion.
struct MatchFailureListener : public RewriterBase::Listener {
  bool hadFailure = false;

  /// Destroy the listener through the MLIR listener base class.
  ~MatchFailureListener() override {}

  /// Convert match failures into reported diagnostics and remember that the pass must fail.
  void notifyMatchFailure(Location loc, function_ref<void(Diagnostic &)> reasonCallback) override {
    hadFailure = true;

    InFlightDiagnostic diag = emitError(loc);
    reasonCallback(*diag.getUnderlyingDiagnostic());
    diag.report();
  }
};

/// Apply a greedy rewrite set, record whether it changed the module, and fail if any pattern
/// reported a hard match failure through `MatchFailureListener`.
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

/// Helper for applying template-parameter substitutions to attributes embedded in types.
class TemplateParamSubstitutions {
  const DenseMap<Attribute, Attribute> &paramNameToValue;

public:
  /// Store the template-parameter binding map used by all substitution helpers.
  explicit TemplateParamSubstitutions(const DenseMap<Attribute, Attribute> &bindings)
      : paramNameToValue(bindings) {}

  /// Return the bound value for `a` when present, otherwise return `a` unchanged.
  Attribute lookupOrSelf(Attribute a) const {
    auto res = paramNameToValue.find(a);
    return (res != paramNameToValue.end()) ? res->second : a;
  }

  /// Return true iff `nameAttr` has a concrete binding.
  bool contains(Attribute nameAttr) const { return paramNameToValue.contains(nameAttr); }

  /// Replace a type variable with a concrete type binding when the binding is usable here.
  Type convertTypeVarBinding(TypeVarType inputTy) const {
    if (TypeAttr tyAttr = llvm::dyn_cast<TypeAttr>(lookupOrSelf(inputTy.getNameRef()))) {
      Type convertedType = tyAttr.getValue();
      if (isConcreteType(convertedType)) {
        return convertedType;
      }
    }
    return inputTy;
  }

  /// Substitute attributes, recursively converting nested `TypeAttr` payloads through `converter`.
  SmallVector<Attribute> convertAttrs(
      const TypeConverter &converter, ArrayRef<Attribute> attrs, bool *changed = nullptr
  ) const {
    SmallVector<Attribute> updated;
    bool anyChanged = false;
    for (Attribute attr : attrs) {
      Attribute converted = attr;
      if (TypeAttr tyAttr = dyn_cast<TypeAttr>(attr)) {
        Type newTy = converter.convertType(tyAttr.getValue());
        if (newTy != tyAttr.getValue()) {
          converted = TypeAttr::get(newTy);
        }
      } else {
        converted = lookupOrSelf(attr);
      }
      anyChanged |= (converted != attr);
      updated.push_back(converted);
    }
    if (changed) {
      *changed = anyChanged;
    }
    return updated;
  }
};

/// Replace a callee rooted at a template parameter with the concrete struct callee named by that
/// parameter's instantiated type.
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

/// Rewrite all nested calls in `op` whose callee root names a concretized template parameter.
static void
convertCalleesInPlace(Operation *op, const DenseMap<Attribute, Attribute> &paramNameToValue) {
  op->walk([&paramNameToValue](CallOp callOp) {
    callOp.setCalleeAttr(convertCalleeSymRefs(callOp.getCalleeAttr(), paramNameToValue));
  });
}

/// Return true iff `op` calls a single-nested symbol rooted at a parameter of its parent template.
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

/// Attempt to evaluate the concrete result of a single `TemplateExprOp` expression given
/// the currently-known concrete param values in `paramNameToConcrete`. Returns the result
/// attribute if all referenced params are concrete and all operations in the body can be
/// constant-folded; otherwise returns `std::nullopt`.
static std::optional<Attribute>
evaluateExpr(TemplateExprOp exprOp, const DenseMap<Attribute, Attribute> &paramNameToConcrete) {
  // Map from SSA value in the expr body to its concrete Attribute.
  DenseMap<Value, Attribute> valueMap;
  for (Operation &bodyOp : exprOp.getInitializerRegion().front()) {
    if (auto yieldOp = llvm::dyn_cast<YieldOp>(bodyOp)) {
      auto it = valueMap.find(yieldOp.getVal());
      return it != valueMap.end() ? std::make_optional(it->second) : std::nullopt;
    }

    if (auto constReadOp = llvm::dyn_cast<ConstReadOp>(bodyOp)) {
      auto it = paramNameToConcrete.find(constReadOp.getConstNameAttr());
      if (it == paramNameToConcrete.end()) {
        return std::nullopt; // a referenced param is not concrete
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
        return std::nullopt; // operand not known as a constant
      }
      operandAttrs.push_back(it->second);
    }

    // Try constant folding.
    SmallVector<OpFoldResult> foldResults;
    if (succeeded(bodyOp.fold(operandAttrs, foldResults)) &&
        foldResults.size() == bodyOp.getNumResults()) {
      for (auto [result, fr] : llvm::zip_equal(bodyOp.getResults(), foldResults)) {
        if (Attribute a = llvm::dyn_cast<Attribute>(fr)) {
          valueMap[result] = a;
        } else {
          return std::nullopt;
        }
      }
    }
  }
  return std::nullopt; // no YieldOp found (shouldn't happen in a valid expr)
}

/// Evaluate all `TemplateExprOp`s in `templateOp` that can be computed from the currently-known
/// concrete param values in `paramNameToConcrete`, and add their results to the map.
/// Exprs whose operands are not all concrete are silently skipped (partial instantiation).
static void
evaluateTemplateExprs(TemplateOp templateOp, DenseMap<Attribute, Attribute> &paramNameToConcrete) {
  LLVM_DEBUG(
      llvm::dbgs() << "[evaluateTemplateExprs] before: " << debug::toStringList(paramNameToConcrete)
                   << '\n'
  );
  for (TemplateExprOp exprOp : templateOp.getConstOps<TemplateExprOp>()) {
    std::optional<Attribute> result = evaluateExpr(exprOp, paramNameToConcrete);
    if (result.has_value()) {
      auto exprNameAttr = FlatSymbolRefAttr::get(exprOp.getSymNameAttr());
      paramNameToConcrete.try_emplace(exprNameAttr, *result);
      LLVM_DEBUG(
          llvm::dbgs() << "[evaluateTemplateExprs] expr @" << exprOp.getSymName()
                       << " evaluated to " << *result << '\n'
      );
    }
  }
  LLVM_DEBUG(
      llvm::dbgs() << "[evaluateTemplateExprs] after: " << debug::toStringList(paramNameToConcrete)
                   << '\n'
  );
}

/// Return true iff `op` no longer has a symbolic member table offset.
static inline bool tableOffsetIsntSymbol(MemberReadOp op) {
  return !llvm::isa_and_present<SymbolRefAttr>(op.getTableOffset().value_or(nullptr));
}

/// Materialize symbolic member table offsets only from integer template bindings. Member tables are
/// index-addressed, so other concrete attribute kinds emit diagnostics instead of being coerced.
class ClonedMemberReadOpPattern
    : public SymbolUserHelper<ClonedMemberReadOpPattern, MemberReadOp, IntegerAttr> {
  using super = SymbolUserHelper<ClonedMemberReadOpPattern, MemberReadOp, IntegerAttr>;

public:
  /// Construct the member-read conversion pattern for the active instantiation map.
  ClonedMemberReadOpPattern(
      TypeConverter &converter, MLIRContext *ctx,
      const DenseMap<Attribute, Attribute> &paramNameToInstantiatedValue
  )
      // benefit>0 so this applies instead of GeneralTypeReplacePattern<MemberReadOp>
      : super(converter, ctx, /*patternBenefit=*/1, paramNameToInstantiatedValue) {}

  /// Use the table-offset attribute as the lookup key.
  Attribute getNameAttr(MemberReadOp op) const override {
    return op.getTableOffset().value_or(nullptr);
  }

  /// Replace a symbolic table offset with the concrete index value.
  LogicalResult handleRewrite(
      Attribute, MemberReadOp op, OpAdaptor, ConversionPatternRewriter &rewriter, IntegerAttr a
  ) const {
    rewriter.modifyOpInPlace(op, [&]() {
      op.setTableOffsetAttr(rewriter.getIndexAttr(fromAPInt(a.getValue())));
    });

    return success();
  }

  /// Emit a diagnostic for concrete template bindings that cannot index member tables.
  LogicalResult handleDefaultRewrite(
      Attribute, MemberReadOp op, OpAdaptor, ConversionPatternRewriter &, Attribute a
  ) const override {
    return op->emitOpError().append(
        "table offset requires an integer template value, but found ", a
    );
  }

  /// Rewrite only member reads whose table offset is still a symbol.
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

/// Implements cloning a `StructDefOp` for a specific instantiation site, using the concrete
/// parameters from the instantiation to replace parameters from the original `StructDefOp`.
class StructCloner {
  ConversionTracker &tracker_;
  ModuleOp rootMod;
  SymbolTableCollection symTables;
  bool reportMissing = true;

  class MappedTypeConverter : public TypeConverter {
    StructType origTy;
    StructType newTy;
    TemplateParamSubstitutions substitutions;

  public:
    /// Build a converter for a cloned struct body, replacing `originalType` with `newType` and
    /// substituting any concretized template parameters.
    MappedTypeConverter(
        StructType originalType, StructType newType,
        /// Instantiated values for the parameter names in `originalType`
        const DenseMap<Attribute, Attribute> &paramNameToInstantiatedValue
    )
        : TypeConverter(), origTy(originalType), newTy(newType),
          substitutions(paramNameToInstantiatedValue) {

      addConversion([](Type inputTy) { return inputTy; });

      addConversion([this](StructType inputTy) {
        LLVM_DEBUG(llvm::dbgs() << "[MappedTypeConverter] convert " << inputTy << '\n');

        // Check for replacement of the full type
        if (inputTy == this->origTy) {
          return this->newTy;
        }
        // Check for replacement of parameter symbol names with concrete values
        if (ArrayAttr inputTyParams = inputTy.getParams()) {
          SmallVector<Attribute> updated =
              substitutions.convertAttrs(*this, inputTyParams.getValue());
          return getStructTypeWithParams(inputTy.getNameRef(), inputTy.getContext(), updated);
        }
        // Otherwise, return the type unchanged
        return inputTy;
      });

      addConversion([this](ArrayType inputTy) {
        // Check for replacement of parameter symbol names with concrete values
        ArrayRef<Attribute> dimSizes = inputTy.getDimensionSizes();
        if (!dimSizes.empty()) {
          SmallVector<Attribute> updated = substitutions.convertAttrs(*this, dimSizes);
          return ArrayType::get(this->convertType(inputTy.getElementType()), updated);
        }
        // Otherwise, return the type unchanged
        return inputTy;
      });

      addConversion([this](TypeVarType inputTy) -> Type {
        // Keep unresolved type variables from other templates because they reference names that
        // are not valid in the current struct.
        return substitutions.convertTypeVarBinding(inputTy);
      });
    }
  };

  /// Clone `typeAtCaller` if at least one of its parameters is concrete at the current use site.
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

    // Map of StructDefOp parameter name to concrete Attribute at the current instantiation site.
    DenseMap<Attribute, Attribute> paramNameToConcrete;
    // List of concrete Attributes from the struct instantiation with `nullptr` at any positions
    // where the original attribute from the current instantiation site was not concrete. This is
    // used for generating the new struct name. See `BuildShortTypeString::from()`.
    SmallVector<Attribute> attrsForInstantiatedNameSuffix;
    // List of template const param names that must be preserved because they
    // were not assigned concrete values at the current instantiation site.
    SmallVector<Attribute> remainingNames;
    // Reduced from `typeAtCallerParams` to contain only the non-concrete Attributes.
    ArrayAttr reducedCallerParams = nullptr;
    {
      ArrayAttr paramNames = typeAtDef.getParams();

      // pre-conditions
      assert(!isNullOrEmpty(paramNames));
      assert(paramNames.size() == typeAtCallerParams.size());

      SmallVector<Attribute> nonConcreteParams;
      for (size_t i = 0, e = paramNames.size(); i < e; ++i) {
        Attribute next = typeAtCallerParams[i];
        if (isConcreteAttr<false>(next)) {
          paramNameToConcrete[paramNames[i]] = next;
          attrsForInstantiatedNameSuffix.push_back(next);
        } else {
          remainingNames.push_back(paramNames[i]);
          nonConcreteParams.push_back(next);
          attrsForInstantiatedNameSuffix.push_back(nullptr);
        }
      }
      // post-conditions
      assert(remainingNames.size() == nonConcreteParams.size());
      assert(attrsForInstantiatedNameSuffix.size() == paramNames.size());
      assert(remainingNames.size() + paramNameToConcrete.size() == paramNames.size());

      if (paramNameToConcrete.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   skip: no concrete params \n");
        return failure();
      }
      if (!remainingNames.empty()) {
        reducedCallerParams = ArrayAttr::get(ctx, nonConcreteParams);
      }
    }

    // This list will be used to build the new remote/external type.
    SmallVector<FlatSymbolRefAttr> typeAtCallerSymPieces = getPieces(typeAtCaller.getNameRef());
    typeAtCallerSymPieces.pop_back(); // drop struct name
    // Name of template with instantiated parameter values.
    std::string templateNameWithAttrs = BuildShortTypeString::from(
        typeAtCallerSymPieces.back().getValue().str(), attrsForInstantiatedNameSuffix
    );

    // Get parent refs
    TemplateOp parentTemplate = getParentOfType<TemplateOp>(origStruct);
    assert(parentTemplate && "parameterized struct must be nested in a TemplateOp");
    ModuleOp parentModule = getParentOfType<ModuleOp>(parentTemplate);
    assert(parentModule && "TemplateOp must be nested in a ModuleOp");

    // Evaluate any poly.expr symbols whose param dependencies are now concrete; add them to the
    // map so ClonedBodyConstReadOpPattern can replace uses of those symbols too.
    evaluateTemplateExprs(parentTemplate, paramNameToConcrete);

    // Clone the original struct.
    StructDefOp newStruct = origStruct.clone();
    convertCalleesInPlace(newStruct, paramNameToConcrete);
    if (remainingNames.empty()) { // FULL INSTANTIATION CASE
      // Set name of the new struct by prepending its name with instantiated template name.
      newStruct.setSymName(
          (templateNameWithAttrs + mlir::Twine('_') + newStruct.getSymName()).str()
      );
      // Insert 'newStruct' into the parent ModuleOp of the original TemplateOp. Use the
      // `SymbolTable::insert()` function so that the name will be made unique if necessary.
      symTables.getSymbolTable(parentModule).insert(newStruct, Block::iterator(parentTemplate));
      // Drop the old template name from the list.
      typeAtCallerSymPieces.pop_back();
    } else { // PARTIAL INSTANTIATION CASE
      // Clone the template and set instantiated name.
      TemplateOp newTemplate = parentTemplate.cloneWithoutRegions();
      newTemplate.setSymName(templateNameWithAttrs);
      assert(newTemplate->getNumRegions() > 0 && "region exists"); // it just doesn't have a block
      newTemplate.getBodyRegion().emplaceBlock();

      // Clone preserved const param/expr ops.
      for (Attribute name : remainingNames) {
        FlatSymbolRefAttr nameSym = llvm::dyn_cast<FlatSymbolRefAttr>(name);
        assert(nameSym && "expected FlatSymbolRefAttr");

        Operation *symOp = symTables.getSymbolTable(parentTemplate).lookup(nameSym.getAttr());
        assert(symOp && "symbol must exist");
        newTemplate.insert(newTemplate.begin(), symOp->clone());
      }

      // Insert the struct into the template and the template into the module. Use the
      // `SymbolTable::insert()` function so that the name will be made unique if necessary.
      symTables.getSymbolTable(newTemplate).insert(newStruct);
      symTables.getSymbolTable(parentModule).insert(newTemplate, Block::iterator(parentTemplate));

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
        tyConv, ctx, paramNameToConcrete, tracker_.delayedDiagnosticSet(newLocalType)
    );
    patterns.add<ClonedMemberReadOpPattern>(tyConv, ctx, paramNameToConcrete);
    if (failed(applyFullConversion(newStruct, target, std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   instantiating body of struct failed \n");
      return failure();
    }
    return newRemoteType;
  }

public:
  /// Construct a cloner rooted at `root` and reporting modifications through `tracker`.
  StructCloner(ConversionTracker &tracker, ModuleOp root)
      : tracker_(tracker), rootMod(root), symTables() {}

  /// Create a full or partial instantiated clone for `orig`, if `orig` has concrete parameters.
  FailureOr<StructType> createInstantiatedClone(StructType orig) {
    LLVM_DEBUG(llvm::dbgs() << "[StructCloner] orig: " << orig << '\n');
    if (ArrayAttr params = orig.getParams()) {
      return genClone(orig, params.getValue());
    }
    LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   skip: nullptr for params \n");
    return failure();
  }

  /// Re-enable diagnostics when a referenced struct definition cannot be found.
  void enableReportMissing() { reportMissing = true; }

  /// Temporarily suppress missing-symbol diagnostics during speculative legality checks.
  void disableReportMissing() { reportMissing = false; }
};

class DisableReportMissing;

class ParameterizedStructUseTypeConverter : public TypeConverter {
  ConversionTracker &tracker_;
  StructCloner cloner;

  friend DisableReportMissing;

public:
  /// Build a type converter that instantiates parameterized struct uses on demand.
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
      tracker_.recordInstantiation(inputTy, newTy);
      return newTy;
    });

    addConversion([this](ArrayType inputTy) {
      return inputTy.cloneWith(convertType(inputTy.getElementType()));
    });
  }
};

/// Rewrite calls to struct `compute`/`constrain` functions after their struct types have been
/// instantiated.
class CallStructFuncPattern : public OpConversionPattern<CallOp> {
  ConversionTracker &tracker_;

public:
  /// Construct the call rewrite pattern using the active type converter and tracker.
  CallStructFuncPattern(TypeConverter &converter, MLIRContext *ctx, ConversionTracker &tracker)
      // benefit>0 so this applies instead of CallOpClassReplacePattern
      : OpConversionPattern<CallOp>(converter, ctx, /*benefit=*/1), tracker_(tracker) {}

  /// Replace a call with converted result types and, when needed, a callee rooted at the
  /// instantiated struct type.
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
    CallOp newOp = replaceOpWithNewOp<CallOp>(
        rewriter, op, newResultTypes, calleeAttr, adapter.getMapOperands(),
        op.getNumDimsPerMapAttr(), adapter.getArgOperands()
    );
    (void)newOp; // tell compiler it's intentionally unused in release builds
    LLVM_DEBUG(llvm::dbgs() << " with " << newOp << '\n');
    return success();
  }
};

/// Ensure `struct.member` types are converted even if no read/write pattern visits them.
class MemberDefOpPattern : public OpConversionPattern<MemberDefOp> {
public:
  /// Construct the member definition conversion pattern.
  MemberDefOpPattern(TypeConverter &converter, MLIRContext *ctx, ConversionTracker &)
      // benefit>0 so this applies instead of GeneralTypeReplacePattern<MemberDefOp>
      : OpConversionPattern<MemberDefOp>(converter, ctx, /*benefit=*/1) {}

  /// Update the member definition type when the active type converter changes it.
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
  /// Tie the callback to the converter whose cloner should suppress lookup diagnostics.
  explicit DisableReportMissing(ParameterizedStructUseTypeConverter &tc) : tyConv(tc) {}

  /// Suppress missing-symbol diagnostics before a speculative legality check begins.
  void checkStarted() override { tyConv.cloner.disableReportMissing(); }

  /// Re-enable missing-symbol diagnostics after the speculative legality check finishes.
  void checkEnded(bool) override { tyConv.cloner.enableReportMissing(); }
};

/// Run struct instantiation and call/member rewrites for the current module.
LogicalResult run(ModuleOp modOp, ConversionTracker &tracker) {
  MLIRContext *ctx = modOp.getContext();
  ParameterizedStructUseTypeConverter tyConv(tracker, modOp);
  DisableReportMissing drm(tyConv);
  ConversionTarget target = newConverterDefinedTargetWithCallback<>(tyConv, ctx, drm);
  RewritePatternSet patterns = newGeneralRewritePatternSet(tyConv, ctx, target);
  patterns.add<CallStructFuncPattern, MemberDefOpPattern>(tyConv, ctx, tracker);
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
  if (!mainType || isNullOrEmpty(mainType.getParams()) || tracker.getInstantiation(mainType)) {
    return success();
  }

  StructCloner cloner(tracker, modOp);
  FailureOr<StructType> cloneRes = cloner.createInstantiatedClone(mainType);
  if (failed(cloneRes)) {
    return failure();
  }

  StructType instantiatedMainType = cloneRes.value();
  tracker.recordInstantiation(mainType, instantiatedMainType);
  modOp->setAttr(MAIN_ATTR_NAME, TypeAttr::get(instantiatedMainType));
  return success();
}

} // namespace Step1_InstantiateStructs

namespace Step2_InstantiateFunctions {

/// TypeConverter for function instantiation that replaces TypeVarType and symbolic
/// ArrayType/StructType parameters with their concrete values determined by unification.
class FuncInstTypeConverter : public TypeConverter {
  DenseMap<Attribute, Attribute> paramNameToValue;
  TemplateParamSubstitutions substitutions;

public:
  /// Build the function-instantiation type converter from concrete template bindings.
  explicit FuncInstTypeConverter(DenseMap<Attribute, Attribute> paramNameToConcrete)
      : TypeConverter(), paramNameToValue(std::move(paramNameToConcrete)),
        substitutions(paramNameToValue) {
    addConversion([](Type t) { return t; });

    addConversion([this](TypeVarType inputTy) -> Type {
      return substitutions.convertTypeVarBinding(inputTy);
    });

    addConversion([this](ArrayType inputTy) {
      bool changed = false;
      SmallVector<Attribute> updated =
          substitutions.convertAttrs(*this, inputTy.getDimensionSizes(), &changed);
      Type newElemTy = this->convertType(inputTy.getElementType());
      if (!changed && newElemTy == inputTy.getElementType()) {
        return inputTy;
      }
      return flattenArrayElementType(
          inputTy.cloneWith(inputTy.getElementType(), updated), newElemTy
      );
    });

    addConversion([this](StructType inputTy) -> StructType {
      if (ArrayAttr params = inputTy.getParams()) {
        bool changed = false;
        SmallVector<Attribute> updated =
            substitutions.convertAttrs(*this, params.getValue(), &changed);
        if (changed) {
          return getStructTypeWithParams(inputTy.getNameRef(), inputTy.getContext(), updated);
        }
      }
      return inputTy;
    });
  }

  /// Convert an attribute that may contain a type or a direct template-parameter reference.
  Attribute convertAttr(Attribute attr) const {
    if (TypeAttr tyAttr = llvm::dyn_cast<TypeAttr>(attr)) {
      Type convertedTy = convertType(tyAttr.getValue());
      if (convertedTy != tyAttr.getValue()) {
        return TypeAttr::get(convertedTy);
      }
    }
    return substitutions.lookupOrSelf(attr);
  }

  /// Return true iff the given template parameter has a concrete binding in this converter.
  bool containsParam(Attribute nameAttr) const { return substitutions.contains(nameAttr); }

  /// Return the underlying template-parameter binding map.
  const DenseMap<Attribute, Attribute> &getParamMap() const { return paramNameToValue; }
};

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
  /// Construct a body inferer over the current symbol tables and known concrete bindings.
  WildcardTypeBodyInferer(
      SymbolTableCollection &symTables, const DenseMap<Attribute, Attribute> &paramNameToConcrete
  )
      : symTables_(symTables), paramNameToConcrete_(paramNameToConcrete) {}

  /// Search `func` for a concrete value that can resolve `paramName`.
  std::optional<Attribute> infer(FuncDefOp func, FlatSymbolRefAttr paramName) {
    if (llvm::any_of(activeInferences_, [&](const auto &e) {
      return e.first == func.getOperation() && e.second == paramName;
    })) {
      return std::nullopt;
    }
    activeInferences_.emplace_back(func.getOperation(), paramName);

    FuncInstTypeConverter tyConv((paramNameToConcrete_));
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
  /// Infer a nested callee parameter value from the nested call's explicit template arguments.
  std::optional<Attribute> inferFromExplicitNestedCallParams(
      CallOp nestedCall, TemplateOp nestedTemplate, FlatSymbolRefAttr nestedParamName,
      const FuncInstTypeConverter &tyConv
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

  /// Replace a scalar element read with an array extract when conversion makes the result an array.
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

  /// Replace a scalar element write with an array insert when conversion makes the value an array.
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

/// Use `FuncInstTypeConverter` to apply the given substitutions from instantiation and verify
/// that `CallOp`s in the converted function are valid for their respective targets (we can emit a
/// more helpful error at this point rather than discovering it later when verifying the module).
static LogicalResult applyBodyConversions(
    CallOp op, FuncDefOp newFunc, const DenseMap<Attribute, Attribute> &paramNameToConcrete
) {
  MLIRContext *ctx = op.getContext();
  FuncInstTypeConverter tyConv(paramNameToConcrete);
  ConversionTarget target = newConverterDefinedTarget<>(tyConv, ctx, tableOffsetIsntSymbol);
  target.addDynamicallyLegalOp<ConstReadOp>([&tyConv](ConstReadOp p) {
    // Legal if it's not in the map of concrete attribute instantiations
    return !tyConv.containsParam(p.getConstNameAttr());
  });
  SmallVector<Diagnostic> delayedDiagnostics;
  RewritePatternSet bodyPatterns = newGeneralRewritePatternSet(tyConv, ctx, target);
  bodyPatterns.add<ClonedBodyConstReadOpPattern>(
      tyConv, ctx, tyConv.getParamMap(), delayedDiagnostics
  );
  bodyPatterns.add<ClonedBodyArrayReadOpPattern, ClonedBodyArrayWriteOpPattern>(tyConv, ctx);
  bodyPatterns.add<ClonedMemberReadOpPattern>(tyConv, ctx, paramNameToConcrete);
  if (failed(applyFullConversion(newFunc, target, std::move(bodyPatterns)))) {
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "[InstantiateFuncAtCallOp]   instantiated clone: " << newFunc << '\n');
  ::reportDelayedDiagnostics(op, std::move(delayedDiagnostics));

  SymbolTableCollection tables;
  WalkResult res = newFunc.walk([&tables](CallOp nestedCall) {
    return WalkResult(nestedCall.verifySymbolUses(tables));
  });
  return failure(res.wasInterrupted());
}

class InstantiateFuncAtCallOp final : public OpRewritePattern<CallOp> {
  ConversionTracker &tracker_;

public:
  /// Construct the function-instantiation pattern.
  InstantiateFuncAtCallOp(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern<CallOp>(ctx), tracker_(tracker) {}

  /// Instantiate the target function or template at a call site and rewrite the callee reference.
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

    // Maps template parameter symbols to the instantiation value at the call site.
    DenseMap<Attribute, Attribute> paramNameToConcrete;
    if (failed(collectConcreteTemplateParams(
            op, rewriter, symTables, callTgt, parentTemplate, unifyResult.value(),
            paramNameToConcrete
        ))) {
      return failure();
    }

    if (paramNameToConcrete.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "[InstantiateFuncAtCallOp]  skip: no concrete params\n");
      return failure();
    }

    evaluateTemplateExprs(parentTemplate, paramNameToConcrete);

    InstantiationLayout layout =
        buildInstantiationLayout(parentTemplate, op.getTemplateParamsAttr(), paramNameToConcrete);
    ModuleOp parentModule = getParentOfType<ModuleOp>(parentTemplate);
    assert(parentModule && "TemplateOp must be nested in a ModuleOp");

    SymbolRefAttr originalCalleeAttr = op.getCalleeAttr();
    FailureOr<SymbolRefAttr> newCalleeAttr =
        layout.remainingNames.empty()
            ? instantiateFully(
                  op, rewriter, symTables, callTgt, parentTemplate, parentModule,
                  layout.templateNameWithAttrs, paramNameToConcrete
              )
            : instantiatePartially(
                  op, rewriter, symTables, callTgt, parentTemplate, parentModule, layout,
                  paramNameToConcrete
              );
    if (failed(newCalleeAttr)) {
      return failure();
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

  /// Populate the concrete subset of template parameters chosen for this instantiation, using
  /// explicit call-site arguments when present and otherwise relying on unification.
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
                                   Attribute concreteValue) {
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
        paramNameToConcrete[paramName] = attr;
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
        newFunc->erase();
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
  /// New template name encodes the concrete values and uses placeholder chars for the rest,
  /// e.g., "TemplateName_8_\x1A" where \x1A marks the position of a non-concrete param.
  static FailureOr<SymbolRefAttr> instantiatePartially(
      CallOp op, PatternRewriter &rewriter, SymbolTableCollection &symTables, FuncDefOp callTgt,
      TemplateOp parentTemplate, ModuleOp parentModule, const InstantiationLayout &layout,
      const DenseMap<Attribute, Attribute> &paramNameToConcrete
  ) {
    TemplateOp newTemplate;
    if (Operation *existing =
            symTables.getSymbolTable(parentModule).lookup(layout.templateNameWithAttrs)) {
      newTemplate = llvm::dyn_cast<TemplateOp>(existing);
    }
    if (!newTemplate) {
      newTemplate = parentTemplate.cloneWithoutRegions();
      newTemplate.setSymName(layout.templateNameWithAttrs);
      assert(newTemplate->getNumRegions() > 0 && "region exists");
      newTemplate.getBodyRegion().emplaceBlock();

      Block &newTemplateBody = newTemplate.getBodyRegion().front();
      for (Attribute name : layout.remainingNames) {
        FlatSymbolRefAttr nameSym = llvm::cast<FlatSymbolRefAttr>(name);
        Operation *paramOp = symTables.getSymbolTable(parentTemplate).lookup(nameSym.getAttr());
        assert(paramOp && "symbol must exist");
        newTemplateBody.push_back(paramOp->clone());
      }

      // Clone and partially convert the function (concretize only the concrete params).
      FuncDefOp newFunc = callTgt.clone();
      convertCalleesInPlace(newFunc, paramNameToConcrete);

      // Insert before body conversion so nested concrete callees verify from the root module. Use
      // the `SymbolTable::insert()` function so that the name will be made unique if necessary.
      symTables.getSymbolTable(newTemplate).insert(newFunc);
      symTables.getSymbolTable(parentModule).insert(newTemplate, Block::iterator(parentTemplate));
      if (failed(applyBodyConversions(op, newFunc, paramNameToConcrete))) {
        StringRef newFuncName = newFunc.getSymName();
        LLVM_DEBUG(
            llvm::dbgs() << "[InstantiateFuncAtCallOp]   body conversion failed for " << newFuncName
                         << '\n'
        );
        newTemplate->erase();
        return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
          diag.append("failure while creating instantiated function '", newFuncName, '\'');
        });
      }

      LLVM_DEBUG(
          llvm::dbgs() << "[InstantiateFuncAtCallOp]  created partial instantiation template: "
                       << newTemplate.getSymName() << '\n'
      );
    } else {
      LLVM_DEBUG(
          llvm::dbgs() << "[InstantiateFuncAtCallOp]  reusing partial instantiation template: "
                       << newTemplate.getSymName() << '\n'
      );
    }

    // Callee: replace old template name with new template name, keep the function name.
    // Original: @[prefix...]::@TemplateName::@funcName
    // New:      @[prefix...]::@newTemplateName::@funcName
    SmallVector<FlatSymbolRefAttr> symPieces = getPieces(op.getCalleeAttr());
    assert(symPieces.size() >= 2 && "callee must include at least template and function names");
    symPieces.pop_back(); // remove original function name (will be re-appended)
    symPieces.pop_back(); // remove original template name
    symPieces.push_back(FlatSymbolRefAttr::get(newTemplate.getSymNameAttr()));
    symPieces.push_back(FlatSymbolRefAttr::get(callTgt.getSymNameAttr()));
    return asSymbolRefAttr(symPieces);
  }
};

/// Run function instantiation patterns once over the module.
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

  /// Fully unroll loop-like ops whose trip count is statically known.
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

/// Run loop unrolling for supported loop dialects.
LogicalResult run(ModuleOp modOp, ConversionTracker &tracker) {
  MLIRContext *ctx = modOp.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<LoopUnrollPattern<scf::ForOp>>(ctx);
  patterns.add<LoopUnrollPattern<affine::AffineForOp>>(ctx);

  return applyAndFoldGreedily(modOp, tracker, std::move(patterns));
}
} // namespace Step3_Unroll

namespace Step4_InstantiateAffineMaps {

/// Return constant integer values for all fold results, if every fold result is constant.
///
/// Adapted from `mlir::getConstantIntValues()` but that one failed in CI for an unknown reason.
/// This version uses a basic loop instead of llvm::map_to_vector().
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

/// Folds affine-map parameters using the map operands supplied at an instantiation site.
struct AffineMapFolder {
  /// Inputs that describe affine-map operands and the parameter list being folded.
  struct Input {
    /// Operand groups corresponding to affine-map parameters.
    OperandRangeRange mapOpGroups;
    /// Number of dimensions in each operand group.
    DenseI32ArrayAttr dimsPerGroup;
    /// Parameter list containing affine maps and non-map attributes.
    ArrayRef<Attribute> paramsOfStructTy;
  };

  /// Outputs after replacing foldable affine-map parameters with concrete attributes.
  struct Output {
    /// Operand groups for affine maps that could not be folded.
    SmallVector<SmallVector<Value>> mapOpGroups;
    /// Dimension counts corresponding to remaining map operand groups.
    SmallVector<int32_t> dimsPerGroup;
    /// Parameter list with folded values substituted where possible.
    SmallVector<Attribute> paramsOfStructTy;
  };

  /// Convert owned output operand groups into `ValueRange` views for op builders.
  static inline SmallVector<ValueRange> getConvertedMapOpGroups(Output out) {
    return llvm::map_to_vector(out.mapOpGroups, [](const SmallVector<Value> &grp) {
      return ValueRange(grp);
    });
  }

  /// Fold any affine-map attributes in `in.paramsOfStructTy` whose operands are all constants.
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
      // If not affine, preserve the original.
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
  /// Construct the array-creation affine-map instantiation pattern.
  InstantiateAtCreateArrayOp(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx), tracker_(tracker) {}

  /// Rewrite `array.new` when affine-map dimensions can be folded to concrete sizes.
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
  /// Construct the struct-compute call result instantiation pattern.
  InstantiateAtCallOpCompute(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx), tracker_(tracker) {}

  /// Refine the result type of calls to struct `compute` functions when parameters become known.
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

/// Run affine-map and target-type instantiation over arrays and struct `compute` calls.
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

namespace Step5_ScalarizeHeterogeneousArrays {

/// Information about a local array allocation that can be replaced with the values written into its
/// statically-known elements.
///
/// This pass only scalarizes arrays after loop unrolling and affine-map instantiation have exposed
/// all element indices and value types. The candidate array must have exactly one write for every
/// static element index, and all direct reads/member writes must happen after every element has
/// been written. Those restrictions avoid imposing a new memory semantics for partially
/// initialized arrays, repeated writes, dynamic indices, or branch-sensitive updates.
struct ScalarizedArrayInfo {
  /// The array allocation being removed.
  CreateArrayOp createOp;
  /// All static element indices in the array type, in the ArrayType's canonical order.
  SmallVector<ArrayAttr> indices;
  /// The SSA value written at each element index.
  DenseMap<ArrayAttr, Value> valueByIndex;
  /// The type of the value written at each element index.
  DenseMap<ArrayAttr, Type> typeByIndex;
  /// The write operation that defines each element index, used for dominance-like ordering checks.
  DenseMap<ArrayAttr, Operation *> writeOpByIndex;
  /// Direct writes to the local allocation.
  SmallVector<WriteArrayOp> writes;
  /// Direct reads from the local allocation.
  SmallVector<ReadArrayOp> reads;
  /// Direct writes that store the whole local allocation into a struct member.
  SmallVector<MemberWriteOp> memberWrites;
};

/// Scalar member name and type.
using MemberInfo = std::pair<StringAttr, Type>;

/// Replacement scalar members for one array-typed member.
struct SplitMemberInfo {
  /// Static element indices that were split out of the original array-typed member.
  SmallVector<ArrayAttr> indices;
  /// Replacement scalar member for each static element index.
  DenseMap<ArrayAttr, MemberInfo> memberByIndex;
};

/// Return true iff `lhs` and `rhs` contain the same array indices.
static bool haveSameIndexSet(ArrayRef<ArrayAttr> lhs, ArrayRef<ArrayAttr> rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  DenseSet<ArrayAttr> rhsSet(rhs.begin(), rhs.end());
  return llvm::all_of(lhs, [&rhsSet](ArrayAttr idx) { return rhsSet.contains(idx); });
}

/// Return the first index in `lhs` that is not present in `rhs`.
static ArrayAttr findIndexMissingFrom(ArrayRef<ArrayAttr> lhs, ArrayRef<ArrayAttr> rhs) {
  DenseSet<ArrayAttr> rhsSet(rhs.begin(), rhs.end());
  const auto *it = llvm::find_if(lhs, [&rhsSet](ArrayAttr idx) { return !rhsSet.contains(idx); });
  return it == lhs.end() ? ArrayAttr() : *it;
}

/// Replace all uses of `oldValue` without asking MLIR to enforce SSA type equality.
///
/// This is intentionally narrower than `replaceAllUsesWith()`: the whole purpose of this step is to
/// remove pseudo-homogeneous array values whose original element type no longer describes the
/// concrete value stored at each index. The caller must have already proven that every rewritten
/// use observes the value at a single static index, so replacing that use with the index-specific
/// value is type-correct for the consuming operation after the rewrite.
static void replaceAllUsesIgnoringType(Value oldValue, Value newValue) {
  for (OpOperand &use : llvm::make_early_inc_range(oldValue.getUses())) {
    use.set(newValue);
  }
}

/// Return true if `def` is in the same block as `user` and appears before it.
static bool strictlyBefore(Operation *def, Operation *user) {
  return def->getBlock() == user->getBlock() && def->isBeforeInBlock(user);
}

/// Return all direct users of `value`, sorted by their order in the containing block.
static SmallVector<Operation *> getUsersInBlockOrder(Value value) {
  SmallVector<Operation *> users(value.getUsers().begin(), value.getUsers().end());
  llvm::sort(users, [](Operation *lhs, Operation *rhs) { return lhs->isBeforeInBlock(rhs); });
  return users;
}

/// Return true if all writes for the scalarized allocation are available before `user`.
///
/// The rewrite currently handles straight-line local array construction. Requiring every write to
/// be in the same block and before the consuming read/member write keeps the replacement local and
/// avoids changing behavior for arrays updated through control flow.
inline static bool allWritesAvailableAt(const ScalarizedArrayInfo &info, Operation *user) {
  return llvm::all_of(info.writeOpByIndex, [user](const auto &entry) {
    return strictlyBefore(entry.second, user);
  });
}

/// Convert array access operands to a static index attribute, if possible.
inline static ArrayAttr getIndexAsAttr(ArrayAccessOpInterface op) {
  return op.indexOperandsToAttributeArray();
}

/// Seed the scalar value map from the explicit elements of a statically-shaped `array.new`.
static LogicalResult seedValuesFromArrayElements(
    CreateArrayOp createOp, ArrayRef<ArrayAttr> indices, DenseMap<ArrayAttr, Value> &valueByIndex
) {
  Operation::operand_range elements = createOp.getElements();
  if (elements.empty()) {
    return success();
  }
  if (elements.size() != indices.size()) {
    return failure();
  }
  for (auto [idx, value] : llvm::zip_equal(indices, elements)) {
    valueByIndex[idx] = value;
  }
  return success();
}

/// Return true iff the candidate array stores at least two non-unifying element types.
///
/// Homogeneous arrays should continue through the normal type-propagation path. This step exists
/// specifically for pseudo-homogeneous arrays, such as arrays whose element is a templated struct
/// with an affine-map parameter that becomes a different concrete struct at each unrolled index.
static bool hasMultipleIncompatibleElementTypes(const ScalarizedArrayInfo &info) {
  SmallVector<Type> previousTypes;
  for (ArrayAttr idx : info.indices) {
    Type nextType = info.typeByIndex.lookup(idx);
    if (!nextType) {
      return false;
    }
    for (Type previousType : previousTypes) {
      if (!typesUnify(previousType, nextType)) {
        return true;
      }
    }
    previousTypes.push_back(nextType);
  }
  return false;
}

/// Collect scalarization information for `op` if it is a safe heterogeneous-array candidate.
///
/// A candidate must:
/// - have a static shape and a real element type,
/// - have only direct reads, direct writes, and whole-array struct member writes as users,
/// - use only static array indices,
/// - have exactly one write for every static element index,
/// - have all reads/member writes after every element write, and
/// - store multiple incompatible element types.
static FailureOr<ScalarizedArrayInfo> getScalarizedArrayInfo(CreateArrayOp op) {
  ArrayType arrTy = op.getType();
  if (!arrTy.hasStaticShape() || llvm::isa<NoneType>(arrTy.getElementType())) {
    return failure();
  }

  std::optional<SmallVector<ArrayAttr>> maybeIndices = arrTy.getSubelementIndices();
  if (!maybeIndices) {
    return failure();
  }

  ScalarizedArrayInfo info;
  info.createOp = op;
  info.indices = std::move(*maybeIndices);
  Value arrayValue = op.getResult();

  for (Operation *user : arrayValue.getUsers()) {
    if (auto writeOp = llvm::dyn_cast<WriteArrayOp>(user)) {
      if (writeOp.getArrRef() != arrayValue) {
        return failure();
      }
      ArrayAttr idx = getIndexAsAttr(writeOp);
      if (!idx) {
        return failure();
      }
      if (info.valueByIndex.contains(idx)) {
        return failure();
      }
      info.valueByIndex[idx] = writeOp.getRvalue();
      info.typeByIndex[idx] = writeOp.getRvalue().getType();
      info.writeOpByIndex[idx] = writeOp.getOperation();
      info.writes.push_back(writeOp);
      continue;
    }
    if (auto readOp = llvm::dyn_cast<ReadArrayOp>(user)) {
      if (readOp.getArrRef() != arrayValue) {
        return failure();
      }
      ArrayAttr idx = getIndexAsAttr(readOp);
      if (!idx) {
        return failure();
      }
      info.reads.push_back(readOp);
      continue;
    }
    if (auto memberWriteOp = llvm::dyn_cast<MemberWriteOp>(user)) {
      if (memberWriteOp.getVal() != arrayValue) {
        return failure();
      }
      info.memberWrites.push_back(memberWriteOp);
      continue;
    }
    return failure();
  }

  for (ArrayAttr idx : info.indices) {
    if (!info.valueByIndex.contains(idx)) {
      return failure();
    }
  }
  for (ReadArrayOp readOp : info.reads) {
    ArrayAttr idx = getIndexAsAttr(readOp);
    if (!info.valueByIndex.contains(idx) || !allWritesAvailableAt(info, readOp)) {
      return failure();
    }
  }
  for (MemberWriteOp memberWriteOp : info.memberWrites) {
    if (!allWritesAvailableAt(info, memberWriteOp)) {
      return failure();
    }
  }
  if (!hasMultipleIncompatibleElementTypes(info)) {
    return failure();
  }
  return info;
}

/// Create scalar replacement members for `member`, or return the replacements already created.
///
/// The replacement members preserve the original member's public/signal/column and discardable
/// metadata, and rely on the containing struct's symbol table to make each generated name unique.
static SplitMemberInfo &getOrCreateSplitMemberInfo(
    MemberDefOp member, const ScalarizedArrayInfo &arrayInfo,
    DenseMap<MemberDefOp, SplitMemberInfo> &splitMembers, SymbolTableCollection &tables,
    PatternRewriter &rewriter
) {
  auto existing = splitMembers.find(member);
  if (existing != splitMembers.end()) {
    return existing->second;
  }

  SplitMemberInfo &splitInfo = splitMembers[member];
  splitInfo.indices = arrayInfo.indices;

  StructDefOp parentStruct = getParentOfType<StructDefOp>(member);
  assert(parentStruct && "MemberDefOp parent is always StructDefOp");
  SymbolTable &structSymbols = tables.getSymbolTable(parentStruct);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(member);
  for (ArrayAttr idx : arrayInfo.indices) {
    Type scalarType = arrayInfo.typeByIndex.lookup(idx);
    MemberDefOp newMember = rewriter.create<MemberDefOp>(
        member.getLoc(), member.getSymNameAttr(), scalarType, member.getSignal(), member.getColumn()
    );
    newMember.setPublicAttr(member.hasPublicAttr());
    newMember->setDiscardableAttrs(member->getDiscardableAttrDictionary());
    StringAttr actualName = structSymbols.insert(newMember);
    splitInfo.memberByIndex[idx] = std::make_pair(actualName, scalarType);
  }
  return splitInfo;
}

/// Refresh cached element values from their still-live writes.
///
/// Candidates are collected before any rewrites run, so one candidate can cache a read result from
/// another candidate. Rewriting the upstream candidate updates the downstream write operands and
/// erases the reads; re-reading the write operands here keeps this candidate from using dangling
/// values.
static LogicalResult refreshValuesFromWrites(ScalarizedArrayInfo &info) {
  for (WriteArrayOp writeOp : info.writes) {
    ArrayAttr idx = getIndexAsAttr(writeOp);
    if (!idx || !info.valueByIndex.contains(idx)) {
      return failure();
    }
    info.valueByIndex[idx] = writeOp.getRvalue();
    info.typeByIndex[idx] = writeOp.getRvalue().getType();
  }
  return success();
}

/// Rewrite one local heterogeneous array allocation into its index-specific scalar values.
///
/// Direct array reads are replaced with the value written at the requested static index.
/// Whole-array member writes are expanded into one scalar member write per index, creating
/// replacement members as needed. Once all consumers are rewritten, the original array writes and
/// allocation are erased.
static LogicalResult rewriteLocalArray(
    ScalarizedArrayInfo &info, DenseMap<MemberDefOp, SplitMemberInfo> &splitMembers,
    SymbolTableCollection &tables, PatternRewriter &rewriter
) {
  if (failed(refreshValuesFromWrites(info))) {
    return failure();
  }

  for (ReadArrayOp readOp : llvm::make_early_inc_range(info.reads)) {
    ArrayAttr idx = getIndexAsAttr(readOp);
    replaceAllUsesIgnoringType(readOp.getResult(), info.valueByIndex.lookup(idx));
    rewriter.eraseOp(readOp);
  }

  for (MemberWriteOp memberWriteOp : llvm::make_early_inc_range(info.memberWrites)) {
    auto memberDef = memberWriteOp.getMemberDefOp(tables);
    if (failed(memberDef)) {
      return failure();
    }
    SplitMemberInfo &splitInfo =
        getOrCreateSplitMemberInfo(memberDef->get(), info, splitMembers, tables, rewriter);

    rewriter.setInsertionPoint(memberWriteOp);
    DictionaryAttr discardableAttrs = memberWriteOp->getDiscardableAttrDictionary();
    for (ArrayAttr idx : info.indices) {
      MemberInfo memberInfo = splitInfo.memberByIndex.lookup(idx);
      if (!memberInfo.first) {
        return failure();
      }
      MemberWriteOp scalarWrite = rewriter.create<MemberWriteOp>(
          memberWriteOp.getLoc(), memberWriteOp.getComponent(),
          FlatSymbolRefAttr::get(memberInfo.first), info.valueByIndex.lookup(idx)
      );
      scalarWrite->setDiscardableAttrs(discardableAttrs);
    }
    rewriter.eraseOp(memberWriteOp);
  }

  for (WriteArrayOp writeOp : llvm::make_early_inc_range(info.writes)) {
    rewriter.eraseOp(writeOp);
  }
  if (info.createOp.getResult().use_empty()) {
    rewriter.eraseOp(info.createOp);
  }
  return success();
}

/// Rewrite reads from array-typed members that were split by `rewriteLocalArray()`.
///
/// The only supported use of the original whole-array member read is a static `array.read`.
/// Supporting arbitrary uses would require reconstructing a pseudo-homogeneous array value, which
/// is exactly the invalid representation this step removes.
static LogicalResult rewriteSplitMemberReads(
    ModuleOp modOp, DenseMap<MemberDefOp, SplitMemberInfo> &splitMembers,
    SymbolTableCollection &tables, PatternRewriter &rewriter
) {
  for (MemberReadOp memberReadOp : walkCollect<MemberReadOp>(modOp)) {
    auto memberDef = memberReadOp.getMemberDefOp(tables);
    if (failed(memberDef)) {
      return failure();
    }
    auto splitIt = splitMembers.find(memberDef->get());
    if (splitIt == splitMembers.end()) {
      continue;
    }

    SmallVector<ReadArrayOp> arrayReads;
    for (Operation *user : memberReadOp.getResult().getUsers()) {
      auto readOp = llvm::dyn_cast<ReadArrayOp>(user);
      if (!readOp || readOp.getArrRef() != memberReadOp.getResult() || !getIndexAsAttr(readOp)) {
        return failure();
      }
      arrayReads.push_back(readOp);
    }

    ValueRange mapOperands;
    std::optional<int32_t> numDims;
    if (!memberReadOp.getMapOperands().empty()) {
      mapOperands = memberReadOp.getMapOperands().front();
      numDims = memberReadOp.getNumDimsPerMap().front();
    }

    DenseMap<ArrayAttr, Value> scalarValueByIndex;
    rewriter.setInsertionPoint(memberReadOp);
    DictionaryAttr discardableAttrs = memberReadOp->getDiscardableAttrDictionary();
    for (ReadArrayOp readOp : arrayReads) {
      ArrayAttr idx = getIndexAsAttr(readOp);
      MemberInfo memberInfo = splitIt->second.memberByIndex.lookup(idx);
      if (!memberInfo.first) {
        return failure();
      }
      if (scalarValueByIndex.contains(idx)) {
        continue;
      }
      auto scalarRead = rewriter.create<MemberReadOp>(
          memberReadOp.getLoc(), memberInfo.second, memberReadOp.getComponent(), memberInfo.first,
          memberReadOp.getTableOffset().value_or(Attribute {}), mapOperands, numDims
      );
      scalarRead->setDiscardableAttrs(discardableAttrs);
      scalarValueByIndex[idx] = scalarRead.getResult();
    }

    for (ReadArrayOp readOp : llvm::make_early_inc_range(arrayReads)) {
      ArrayAttr idx = getIndexAsAttr(readOp);
      replaceAllUsesIgnoringType(readOp.getResult(), scalarValueByIndex.lookup(idx));
      rewriter.eraseOp(readOp);
    }
    if (memberReadOp.getResult().use_empty()) {
      rewriter.eraseOp(memberReadOp);
    }
  }
  return success();
}

/// Return the local array allocation stored by `memberWriteOp`, if it has a static shape.
static FailureOr<CreateArrayOp> getStaticLocalArrayCreate(MemberWriteOp memberWriteOp) {
  auto createOp = memberWriteOp.getVal().getDefiningOp<CreateArrayOp>();
  if (!createOp) {
    return failure();
  }
  ArrayType arrTy = createOp.getType();
  if (!arrTy.hasStaticShape() || llvm::isa<NoneType>(arrTy.getElementType())) {
    return failure();
  }
  return createOp;
}

/// Verify that `createOp` can be expanded while splitting one or more members.
///
/// Static writes update the local value map, static reads consume it, and missing indices are
/// materialized as shared `llzk.nondet` values. Other users would observe or escape the array in a
/// way this local scalarization cannot preserve.
static LogicalResult verifyExpandableLocalArrayUsers(
    CreateArrayOp createOp, const DenseSet<MemberDefOp> &splitMemberSet,
    SymbolTableCollection &tables
) {
  Value arrayValue = createOp.getResult();
  for (Operation *user : createOp.getResult().getUsers()) {
    if (auto writeOp = llvm::dyn_cast<WriteArrayOp>(user)) {
      if (writeOp.getArrRef() != arrayValue || writeOp->getBlock() != createOp->getBlock() ||
          !getIndexAsAttr(writeOp)) {
        return failure();
      }
      continue;
    }
    if (auto readOp = llvm::dyn_cast<ReadArrayOp>(user)) {
      if (readOp.getArrRef() != arrayValue || readOp->getBlock() != createOp->getBlock() ||
          !getIndexAsAttr(readOp)) {
        return failure();
      }
      continue;
    }
    if (auto userWriteOp = llvm::dyn_cast<MemberWriteOp>(user)) {
      auto memberDef = userWriteOp.getMemberDefOp(tables);
      if (userWriteOp.getVal() == arrayValue && succeeded(memberDef) &&
          splitMemberSet.contains(memberDef->get()) &&
          userWriteOp->getBlock() == createOp->getBlock()) {
        continue;
      }
    }
    return failure();
  }
  return success();
}

/// Return the value for `idx` at `type`, creating and caching a nondeterministic value if the local
/// array element has not been written.
///
/// The caller must have already rejected initialized expandable arrays that store the same SSA
/// value into indices requiring incompatible split-member types.
static FailureOr<Value> getOrCreateScalarizedLocalArrayValue(
    DenseMap<ArrayAttr, Value> &valueByIndex, ArrayAttr idx, Type type, Location loc,
    PatternRewriter &rewriter, const ConversionTracker &tracker
) {
  Value scalarValue = valueByIndex.lookup(idx);
  if (!scalarValue) {
    scalarValue = rewriter.create<NonDetOp>(loc, type).getResult();
    valueByIndex[idx] = scalarValue;
  }
  if (scalarValue.getType() == type) {
    return scalarValue;
  }
  if (!typesUnify(scalarValue.getType(), type) &&
      !tracker.isLegalConversion(
          scalarValue.getType(), type, "getOrCreateScalarizedLocalArrayValue"
      )) {
    return failure();
  }
  return scalarValue;
}

/// Reject expandable arrays that would store one SSA value into several split scalar members with
/// incompatible concrete types.
///
/// Later type propagation may refine the value operand of each scalar member write. If two writes
/// keep the same SSA value, refining one write also refines the other because MLIR values carry one
/// global type. Missing values are materialized as fresh `llzk.nondet` values during rewriting and
/// then cached by index, so the check tracks those pending materializations until an `array.write`
/// overwrites the index.
static LogicalResult verifyNoSharedValuesForIncompatibleSplitTypes(
    CreateArrayOp createOp,
    const DenseMap<MemberDefOp, DenseMap<ArrayAttr, Type>> &splitTypesByMember,
    SymbolTableCollection &tables, StringRef arrayDescription = "an expandable array"
) {
  ArrayType arrTy = createOp.getType();
  std::optional<SmallVector<ArrayAttr>> maybeIndices = arrTy.getSubelementIndices();
  if (!maybeIndices) {
    return failure();
  }
  ArrayRef<ArrayAttr> indices = *maybeIndices;

  DenseMap<ArrayAttr, Value> valueByIndex;
  if (failed(seedValuesFromArrayElements(createOp, indices, valueByIndex))) {
    return failure();
  }

  SmallVector<Operation *> users = getUsersInBlockOrder(createOp.getResult());

  DenseMap<Value, std::pair<Type, Location>> firstUseByValue;
  DenseMap<ArrayAttr, std::pair<Type, Location>> firstMaterializedUseByIndex;
  for (Operation *user : users) {
    if (auto writeOp = llvm::dyn_cast<WriteArrayOp>(user)) {
      ArrayAttr idx = getIndexAsAttr(writeOp);
      valueByIndex[idx] = writeOp.getRvalue();
      firstMaterializedUseByIndex.erase(idx);
      continue;
    }
    if (llvm::isa<ReadArrayOp>(user)) {
      continue;
    }

    auto memberWriteOp = llvm::dyn_cast<MemberWriteOp>(user);
    if (!memberWriteOp) {
      continue;
    }
    auto memberDef = memberWriteOp.getMemberDefOp(tables);
    if (failed(memberDef)) {
      return failure();
    }
    auto splitIt = splitTypesByMember.find(memberDef->get());
    if (splitIt == splitTypesByMember.end()) {
      continue;
    }

    for (const auto &[idx, targetType] : splitIt->second) {
      Value scalarValue = valueByIndex.lookup(idx);
      if (!scalarValue) {
        auto existing = firstMaterializedUseByIndex.find(idx);
        if (existing == firstMaterializedUseByIndex.end()) {
          firstMaterializedUseByIndex.try_emplace(
              idx, std::make_pair(targetType, memberWriteOp.getLoc())
          );
          continue;
        }
        Type existingType = existing->second.first;
        if (!typesUnify(existingType, targetType)) {
          InFlightDiagnostic diag =
              createOp.emitError("cannot split heterogeneous array member because ")
              << arrayDescription << " reuses one SSA value for incompatible scalar member types";
          diag.attachNote(existing->second.second)
              << "unwritten index " << idx << " is materialized for scalar member type "
              << existingType;
          diag.attachNote(memberWriteOp.getLoc())
              << "same index is also materialized for scalar member type " << targetType;
          return diag;
        }
        continue;
      }
      auto existing = firstUseByValue.find(scalarValue);
      if (existing == firstUseByValue.end()) {
        firstUseByValue.try_emplace(
            scalarValue, std::make_pair(targetType, memberWriteOp.getLoc())
        );
        continue;
      }
      Type existingType = existing->second.first;
      if (!typesUnify(existingType, targetType)) {
        InFlightDiagnostic diag =
            createOp.emitError("cannot split heterogeneous array member because ")
            << arrayDescription << " reuses one SSA value for incompatible scalar member types";
        diag.attachNote(existing->second.second)
            << "value is used for scalar member type " << existingType;
        diag.attachNote(memberWriteOp.getLoc())
            << "same value is also used for scalar member type " << targetType;
        return diag;
      }
    }
  }
  return success();
}

/// Rewrite one expandable local array allocation in block order.
static LogicalResult rewriteExpandableLocalArray(
    CreateArrayOp createOp, const DenseMap<MemberDefOp, SplitMemberInfo> &splitMembers,
    SymbolTableCollection &tables, PatternRewriter &rewriter, const ConversionTracker &tracker
) {
  SmallVector<Operation *> users = getUsersInBlockOrder(createOp.getResult());

  ArrayType arrTy = createOp.getType();
  std::optional<SmallVector<ArrayAttr>> maybeIndices = arrTy.getSubelementIndices();
  if (!maybeIndices) {
    return failure();
  }
  ArrayRef<ArrayAttr> indices = *maybeIndices;

  DenseMap<ArrayAttr, Value> valueByIndex;
  if (failed(seedValuesFromArrayElements(createOp, indices, valueByIndex))) {
    return failure();
  }
  SmallVector<WriteArrayOp> writesToErase;
  for (Operation *user : users) {
    if (auto writeOp = llvm::dyn_cast<WriteArrayOp>(user)) {
      valueByIndex[getIndexAsAttr(writeOp)] = writeOp.getRvalue();
      writesToErase.push_back(writeOp);
      continue;
    }

    if (auto readOp = llvm::dyn_cast<ReadArrayOp>(user)) {
      ArrayAttr idx = getIndexAsAttr(readOp);
      rewriter.setInsertionPoint(readOp);
      FailureOr<Value> scalarValue = getOrCreateScalarizedLocalArrayValue(
          valueByIndex, idx, readOp.getResult().getType(), readOp.getLoc(), rewriter, tracker
      );
      if (failed(scalarValue)) {
        return failure();
      }
      replaceAllUsesIgnoringType(readOp.getResult(), *scalarValue);
      rewriter.eraseOp(readOp);
      continue;
    }

    auto memberWriteOp = llvm::dyn_cast<MemberWriteOp>(user);
    if (!memberWriteOp) {
      return failure();
    }

    auto memberDef = memberWriteOp.getMemberDefOp(tables);
    if (failed(memberDef)) {
      return failure();
    }
    auto splitIt = splitMembers.find(memberDef->get());
    if (splitIt == splitMembers.end()) {
      return failure();
    }
    const SplitMemberInfo &splitInfo = splitIt->second;

    rewriter.setInsertionPoint(memberWriteOp);
    DictionaryAttr discardableAttrs = memberWriteOp->getDiscardableAttrDictionary();
    for (ArrayAttr idx : splitInfo.indices) {
      MemberInfo memberInfo = splitInfo.memberByIndex.lookup(idx);
      FailureOr<Value> scalarValue = getOrCreateScalarizedLocalArrayValue(
          valueByIndex, idx, memberInfo.second, memberWriteOp.getLoc(), rewriter, tracker
      );
      if (failed(scalarValue)) {
        return failure();
      }
      MemberWriteOp scalarWrite = rewriter.create<MemberWriteOp>(
          memberWriteOp.getLoc(), memberWriteOp.getComponent(),
          FlatSymbolRefAttr::get(memberInfo.first), *scalarValue
      );
      scalarWrite->setDiscardableAttrs(discardableAttrs);
    }
    rewriter.eraseOp(memberWriteOp);
  }

  for (WriteArrayOp writeOp : llvm::make_early_inc_range(writesToErase)) {
    rewriter.eraseOp(writeOp);
  }
  if (createOp.getResult().use_empty()) {
    rewriter.eraseOp(createOp);
  }
  return success();
}

/// Rewrite remaining expandable whole-array writes to split members.
static LogicalResult rewriteExpandableMemberWrites(
    DenseMap<MemberDefOp, SplitMemberInfo> &splitMembers, SymbolTableCollection &tables,
    PatternRewriter &rewriter, const ConversionTracker &tracker
) {
  DenseSet<MemberDefOp> splitMemberSet;
  for (const auto &entry : splitMembers) {
    splitMemberSet.insert(entry.first);
  }

  for (const auto &entry : splitMembers) {
    MemberDefOp member = entry.first;
    StructDefOp parentStruct = getParentOfType<StructDefOp>(member);
    assert(parentStruct && "MemberDefOp parent is always StructDefOp");

    auto uses = llzk::getSymbolUses(member, parentStruct);
    if (!uses) {
      return failure();
    }

    SmallVector<CreateArrayOp> createsToExpand;
    DenseSet<CreateArrayOp> seenCreates;
    for (SymbolTable::SymbolUse symUse : uses.value()) {
      auto memberWriteOp = llvm::dyn_cast<MemberWriteOp>(symUse.getUser());
      if (!memberWriteOp) {
        continue;
      }
      auto memberDef = memberWriteOp.getMemberDefOp(tables);
      FailureOr<CreateArrayOp> maybeCreateOp = getStaticLocalArrayCreate(memberWriteOp);
      if (succeeded(memberDef) && memberDef->get() == member && succeeded(maybeCreateOp) &&
          !seenCreates.contains(*maybeCreateOp)) {
        createsToExpand.push_back(*maybeCreateOp);
        seenCreates.insert(*maybeCreateOp);
      }
    }

    for (CreateArrayOp createOp : createsToExpand) {
      if (failed(verifyExpandableLocalArrayUsers(createOp, splitMemberSet, tables))) {
        return failure();
      }
      if (failed(rewriteExpandableLocalArray(createOp, splitMembers, tables, rewriter, tracker))) {
        return failure();
      }
    }
  }
  return success();
}

/// Verify that all writes to any split member are either candidate or otherwise expandable.
///
/// Splitting a member is global: after replacement, every read of the original array-typed member
/// is redirected to the split scalar members. That is only sound when every write to that member
/// can also be expanded.
static LogicalResult verifySplitMemberWritesExpandable(
    ArrayRef<ScalarizedArrayInfo> arraysToScalarize, SymbolTableCollection &tables
) {
  DenseMap<MemberDefOp, DenseSet<Operation *>> candidateWritesByMember;
  DenseMap<MemberDefOp, SmallVector<ArrayAttr>> splitIndicesByMember;
  DenseMap<MemberDefOp, Operation *> splitIndexOpsByMember;
  DenseMap<MemberDefOp, DenseMap<ArrayAttr, Type>> splitTypesByMember;
  DenseMap<MemberDefOp, DenseMap<ArrayAttr, Operation *>> splitTypeOpsByMember;
  for (const ScalarizedArrayInfo &info : arraysToScalarize) {
    for (MemberWriteOp memberWriteOp : info.memberWrites) {
      auto memberDef = memberWriteOp.getMemberDefOp(tables);
      if (succeeded(memberDef)) {
        MemberDefOp member = memberDef->get();
        candidateWritesByMember[member].insert(memberWriteOp.getOperation());
        auto [indicesIt, inserted] = splitIndicesByMember.try_emplace(member, info.indices);
        if (inserted) {
          splitIndexOpsByMember[member] = memberWriteOp.getOperation();
        } else if (!haveSameIndexSet(indicesIt->second, info.indices)) {
          InFlightDiagnostic diag = member.emitError(
              "cannot split heterogeneous array member because candidate writes use different "
              "index sets"
          );
          diag.attachNote(splitIndexOpsByMember.lookup(member)->getLoc())
              << "candidate establishes " << indicesIt->second.size() << " split member indices";
          Diagnostic &note = diag.attachNote(memberWriteOp.getLoc())
                             << "conflicting candidate has " << info.indices.size()
                             << " split member indices";
          if (ArrayAttr extraIndex = findIndexMissingFrom(info.indices, indicesIt->second)) {
            note << ", including extra index " << extraIndex;
          } else if (ArrayAttr missingIndex =
                         findIndexMissingFrom(indicesIt->second, info.indices)) {
            note << ", missing index " << missingIndex;
          }
          return diag;
        }
        DenseMap<ArrayAttr, Type> &splitTypes = splitTypesByMember[member];
        DenseMap<ArrayAttr, Operation *> &splitTypeOps = splitTypeOpsByMember[member];
        for (ArrayAttr idx : info.indices) {
          Type candidateType = info.typeByIndex.lookup(idx);
          auto existing = splitTypes.find(idx);
          if (existing == splitTypes.end()) {
            splitTypes[idx] = candidateType;
            splitTypeOps[idx] = memberWriteOp.getOperation();
            continue;
          }
          Type existingType = existing->second;
          if (!typesUnify(existingType, candidateType)) {
            InFlightDiagnostic diag = member.emitError(
                "cannot split heterogeneous array member because candidate writes require "
                "incompatible scalar member types"
            );
            diag.attachNote(splitTypeOps.lookup(idx)->getLoc())
                << "candidate writes index " << idx << " with scalar member type " << existingType;
            diag.attachNote(memberWriteOp.getLoc())
                << "conflicting candidate writes the same index with scalar member type "
                << candidateType;
            return diag;
          }
        }
      }
    }
  }
  DenseSet<MemberDefOp> splitMemberSet;
  for (const auto &entry : candidateWritesByMember) {
    splitMemberSet.insert(entry.first);
  }

  for (const ScalarizedArrayInfo &info : arraysToScalarize) {
    auto verifyNoIncompatibleShares = verifyNoSharedValuesForIncompatibleSplitTypes(
        info.createOp, splitTypesByMember, tables, "a scalarization candidate"
    );
    if (failed(verifyNoIncompatibleShares)) {
      return failure();
    }
  }

  for (const auto &entry : candidateWritesByMember) {
    MemberDefOp member = entry.first;
    const DenseSet<Operation *> &candidateWrites = entry.second;
    StructDefOp parentStruct = getParentOfType<StructDefOp>(member);
    assert(parentStruct && "MemberDefOp parent is always StructDefOp");

    auto uses = llzk::getSymbolUses(member, parentStruct);
    if (!uses) {
      return member.emitError(
          "cannot split heterogeneous array member because its symbol uses could not be inspected"
      );
    }

    for (SymbolTable::SymbolUse symUse : uses.value()) {
      auto writeOp = llvm::dyn_cast<MemberWriteOp>(symUse.getUser());
      if (!writeOp || candidateWrites.contains(writeOp.getOperation())) {
        continue;
      }
      FailureOr<CreateArrayOp> maybeCreateOp = getStaticLocalArrayCreate(writeOp);
      if (failed(maybeCreateOp) ||
          failed(verifyExpandableLocalArrayUsers(*maybeCreateOp, splitMemberSet, tables))) {
        InFlightDiagnostic diag = member.emitError(
            "cannot split heterogeneous array member because not every write to it can be "
            "scalarized"
        );
        diag.attachNote(writeOp.getLoc()) << "whole-array write is not backed by a scalarization "
                                             "candidate";
        return diag;
      }
      auto verifyNoIncompatibleShares =
          verifyNoSharedValuesForIncompatibleSplitTypes(*maybeCreateOp, splitTypesByMember, tables);
      if (failed(verifyNoIncompatibleShares)) {
        return failure();
      }
    }
  }
  return success();
}

/// Erase original array-typed members after all symbol uses have been redirected.
static void eraseUnusedOriginalMembers(
    DenseMap<MemberDefOp, SplitMemberInfo> &splitMembers, PatternRewriter &rewriter
) {
  for (const auto &[member, _] : splitMembers) {
    StructDefOp parentStruct = getParentOfType<StructDefOp>(member);
    assert(parentStruct && "MemberDefOp parent is always StructDefOp");
    auto uses = llzk::getSymbolUses(member, parentStruct);
    if (uses && uses->empty()) {
      rewriter.eraseOp(member);
    }
  }
}

/// Scalarize all safe pseudo-homogeneous arrays exposed in the current flattening iteration.
///
/// Running this before general type propagation prevents the propagation step from choosing one
/// concrete element type for an array that semantically contains a different concrete type at each
/// static index.
LogicalResult run(ModuleOp modOp, ConversionTracker &tracker) {
  SmallVector<ScalarizedArrayInfo, 2> arraysToScalarize;
  modOp.walk([&arraysToScalarize](CreateArrayOp op) {
    FailureOr<ScalarizedArrayInfo> info = getScalarizedArrayInfo(op);
    if (succeeded(info)) {
      arraysToScalarize.push_back(*info);
    }
  });
  if (arraysToScalarize.empty()) {
    return success();
  }

  PatternRewriter rewriter(modOp.getContext());
  SymbolTableCollection tables;
  if (failed(verifySplitMemberWritesExpandable(arraysToScalarize, tables))) {
    return failure();
  }

  DenseMap<MemberDefOp, SplitMemberInfo> splitMembers;
  for (ScalarizedArrayInfo &info : arraysToScalarize) {
    if (failed(rewriteLocalArray(info, splitMembers, tables, rewriter))) {
      return failure();
    }
  }
  if (failed(rewriteExpandableMemberWrites(splitMembers, tables, rewriter, tracker))) {
    return failure();
  }
  if (failed(rewriteSplitMemberReads(modOp, splitMembers, tables, rewriter))) {
    return failure();
  }
  eraseUnusedOriginalMembers(splitMembers, rewriter);
  tracker.updateModifiedFlag(true);
  return success();
}

} // namespace Step5_ScalarizeHeterogeneousArrays

namespace Step6_PropagateTypes {

/// Update the array element type by looking at the values stored into it from uses.
class UpdateNewArrayElemFromWrite final : public OpRewritePattern<CreateArrayOp> {
  ConversionTracker &tracker_;

public:
  /// Construct the create-array element-type propagation pattern.
  UpdateNewArrayElemFromWrite(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  /// Update an `array.new` result element type from compatible writes into the array.
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

/// Update the array reference type on an array access op to match a scalar element type observed
/// through that access.
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
  /// Construct the array-write based element-type propagation pattern.
  UpdateArrayElemFromArrWrite(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  /// Update the referenced array type from the write value type.
  LogicalResult matchAndRewrite(WriteArrayOp op, PatternRewriter &rewriter) const override {
    return updateArrayElemFromArrAccessOp(op, op.getRvalue().getType(), tracker_, rewriter);
  }
};

class UpdateArrayElemFromArrRead final : public OpRewritePattern<ReadArrayOp> {
  ConversionTracker &tracker_;

public:
  /// Construct the array-read based element-type propagation pattern.
  UpdateArrayElemFromArrRead(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  /// Update the referenced array type from the read result type.
  LogicalResult matchAndRewrite(ReadArrayOp op, PatternRewriter &rewriter) const override {
    return updateArrayElemFromArrAccessOp(op, op.getResult().getType(), tracker_, rewriter);
  }
};

/// Update the type of MemberDefOp instances by checking the updated types from MemberWriteOp.
class UpdateMemberDefTypeFromWrite final : public OpRewritePattern<MemberDefOp> {
  ConversionTracker &tracker_;

public:
  /// Construct the member-definition propagation pattern.
  UpdateMemberDefTypeFromWrite(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  /// Update a member definition type from compatible writes to that member.
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

/// Move all regions out of `op` so it can be recreated with updated result types.
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
  /// Construct the inferred-result propagation pattern.
  UpdateInferredResultTypes(MLIRContext *ctx, ConversionTracker &tracker)
      : OpTraitRewritePattern(ctx, 6), tracker_(tracker) {}

  /// Re-infer result types and recreate the op when the inferred types are more concrete.
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
  /// Construct the function-return based type propagation pattern.
  UpdateFuncTypeFromReturn(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  /// Update a function type from its terminator operand types.
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

/// Update CallOp result type based on the updated return type from the target FuncDefOp.
/// This only applies to free (i.e., non-struct) functions because the functions within structs
/// only return StructType or nothing and propagating those can result in bringing un-instantiated
/// types from a templated struct into the current call which will give errors.
class UpdateFreeFuncCallOpTypes final : public OpRewritePattern<CallOp> {
  ConversionTracker &tracker_;

public:
  /// Construct the free-function call result propagation pattern.
  UpdateFreeFuncCallOpTypes(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  /// Rewrite a call to a free function when the target function result types were refined.
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
    CallOp newOp = replaceOpWithNewOp<CallOp>(rewriter, op, targetFunc, op.getArgOperands());
    (void)newOp; // tell compiler it's intentionally unused in release builds
    LLVM_DEBUG(llvm::dbgs() << " with " << newOp << '\n');
    return success();
  }
};

namespace {

/// Update a member read/write value type from the referenced member definition.
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
  /// Construct the member-read value propagation pattern.
  UpdateMemberReadValFromDef(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  /// Update a member read result type from its referenced member definition.
  LogicalResult matchAndRewrite(MemberReadOp op, PatternRewriter &rewriter) const override {
    return updateMemberRefValFromMemberDef(op, tracker_, rewriter);
  }
};

/// Update the type of MemberWriteOp value based on updated types from MemberDefOp.
class UpdateMemberWriteValFromDef final : public OpRewritePattern<MemberWriteOp> {
  ConversionTracker &tracker_;

public:
  /// Construct the member-write value propagation pattern.
  UpdateMemberWriteValFromDef(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  /// Update a member write operand type from its referenced member definition.
  LogicalResult matchAndRewrite(MemberWriteOp op, PatternRewriter &rewriter) const override {
    return updateMemberRefValFromMemberDef(op, tracker_, rewriter);
  }
};

/// Run all type-propagation patterns to a local fixpoint for the current iteration.
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
} // namespace Step6_PropagateTypes

namespace Step7_Cleanup {

/// Cleanup strategy that preserves symbols reachable from an explicit keep set plus globals.
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

} // namespace Step7_Cleanup

class PassImpl : public llzk::polymorphic::impl::FlatteningPassBase<PassImpl> {
  using Base = FlatteningPassBase<PassImpl>;
  using Base::Base;

  /// If the cleanup mode is unspecified, default to `Preimage`.
  FlatteningCleanupMode getEffectiveCleanupMode() const {
    FlatteningCleanupMode m = cleanupMode.getValue();
    return m == FlatteningCleanupMode::Unspecified ? FlatteningCleanupMode::Preimage : m;
  }

  /// Run the pass on the current module and signal failure if any flattening step fails.
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

  /// Execute the full flattening pipeline until it reaches a fixpoint or the iteration limit.
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

    // Run universal cleanup as a preliminary step to satisfy the
    // `assert(!isNullOrEmpty(paramNames))` precondition in `genClone()`.
    if (failed(runPipeline(universalCleanup, modOp))) {
      return failure();
    }

    ConversionTracker tracker;
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

      LLVM_DEBUG({
        llvm::dbgs() << "[FlatteningPass(count=" << loopCount
                     << ")] Running step 2: function instantiation\n";
      });
      // Instantiate calls to templated functions.
      if (failed(Step2_InstantiateFunctions::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while instantiating functions in templates\n";
        return failure();
      }

      LLVM_DEBUG({
        llvm::dbgs() << "[FlatteningPass(count=" << loopCount
                     << ")] Running step 3: loop unrolling\n";
      });
      // Unroll loops with known iterations.
      if (failed(Step3_Unroll::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while unrolling loops\n";
        return failure();
      }

      LLVM_DEBUG({
        llvm::dbgs() << "[FlatteningPass(count=" << loopCount
                     << ")] Running step 4: affine maps instantiation\n";
      });
      // Instantiate affine_map parameters of StructType and ArrayType.
      if (failed(Step4_InstantiateAffineMaps::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while instantiating `affine_map` parameters\n";
        return failure();
      }

      // Split static arrays whose affine-map element type instantiates to different concrete
      // element types at different indices.
      LLVM_DEBUG({
        llvm::dbgs() << "[FlatteningPass(count=" << loopCount
                     << ")] Running step 5: heterogeneous array scalarization\n";
      });
      if (failed(Step5_ScalarizeHeterogeneousArrays::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while scalarizing heterogeneous arrays\n";
        return failure();
      }

      LLVM_DEBUG({
        llvm::dbgs() << "[FlatteningPass(count=" << loopCount
                     << ")] Running step 6: type propagation\n";
      });
      // Propagate updated types using the semantics of various ops.
      if (failed(Step6_PropagateTypes::run(modOp, tracker))) {
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

  /// Perform cleanup according to the effective `cleanupMode` option.
  LogicalResult cleanupSwitch(ModuleOp modOp, const ConversionTracker &tracker) {
    FlatteningCleanupMode effectiveCleanupMode = getEffectiveCleanupMode();
    LLVM_DEBUG({ llvm::dbgs() << "[FlatteningPass] Running step 7: cleanup "; });
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
    case FlatteningCleanupMode::Disabled:
      LLVM_DEBUG(llvm::dbgs() << "(disabled)\n");
      return success();
    case FlatteningCleanupMode::Unspecified:
      llvm_unreachable("`getEffectiveCleanupMode()` cannot give `Unspecified`");
    }
    llvm_unreachable("unknown cleanup mode");
  }

  /// Erase parameterized definitions that were replaced with concrete instantiations.
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

  /// Erase cleanup candidates that are unreachable from any concrete definition or global.
  LogicalResult eraseUnreachableFromConcreteDefinitions(ModuleOp rootMod) {
    SmallVector<SymbolOpInterface> roots;
    rootMod.walk([&roots](Operation *op) {
      if (isErasableDefinition(op) && !Step7_Cleanup::FromKeepSet::hasTemplateSymbolBindings(op)) {
        roots.push_back(llvm::cast<SymbolOpInterface>(op));
      }
    });

    Step7_Cleanup::FromKeepSet cleaner(
        rootMod, getAnalysis<SymbolDefTree>(), getAnalysis<SymbolUseGraph>()
    );
    return cleaner.eraseUnreachableFrom(roots);
  }

  /// Erase cleanup candidates that are unreachable from the `llzk.main` struct or globals.
  LogicalResult eraseUnreachableFromMainStruct(ModuleOp rootMod, bool emitWarning = true) {
    Step7_Cleanup::FromKeepSet cleaner(
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
