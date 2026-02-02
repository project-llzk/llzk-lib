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

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DepthFirstIterator.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>

// Include the generated base pass class definitions.
namespace llzk::polymorphic {
#define GEN_PASS_DECL_FLATTENINGPASS
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

class ConversionTracker {
  /// Tracks if some step performed a modification of the code such that another pass should be run.
  bool modified;
  /// Maps original remote (i.e., use site) type to new remote type.
  /// Note: The keys are always parameterized StructType and the values are no-parameter StructType.
  DenseMap<StructType, StructType> structInstantiations;
  /// Contains the reverse of mappings in `structInstantiations` for use in legal conversion check.
  DenseMap<StructType, StructType> reverseInstantiations;
  /// Maps new remote type (i.e., the values in 'structInstantiations') to a list of Diagnostic
  /// to report at the location(s) of the compute() that causes the instantiation to the StructType.
  DenseMap<StructType, SmallVector<Diagnostic>> delayedDiagnostics;

public:
  bool isModified() const { return modified; }
  void resetModifiedFlag() { modified = false; }
  void updateModifiedFlag(bool currStepModified) { modified |= currStepModified; }

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

  /// Collect the fully-qualified names of all structs that were instantiated.
  DenseSet<SymbolRefAttr> getInstantiatedStructNames() const {
    DenseSet<SymbolRefAttr> instantiatedNames;
    for (const auto &[origRemoteTy, _] : structInstantiations) {
      instantiatedNames.insert(origRemoteTy.getNameRef());
    }
    return instantiatedNames;
  }

  void reportDelayedDiagnostics(StructType newType, CallOp caller) {
    auto res = delayedDiagnostics.find(newType);
    if (res == delayedDiagnostics.end()) {
      return;
    }

    DiagnosticEngine &engine = caller.getContext()->getDiagEngine();
    for (Diagnostic &diag : res->second) {
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
    // Emitting a Diagnostic consumes it (per DiagnosticEngine::emit) so remove them from the map.
    // Unfortunately, this means if the key StructType is the result of instantiation at multiple
    // `compute()` calls it will only be reported at one of those locations, not all.
    delayedDiagnostics.erase(newType);
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

  template <typename T, typename U>
  inline bool areLegalConversions(T oldTypes, U newTypes, const char *patName) const {
    return llvm::all_of(
        llvm::zip_equal(oldTypes, newTypes), [this, &patName](std::tuple<Type, Type> oldThenNew) {
      return this->isLegalConversion(std::get<0>(oldThenNew), std::get<1>(oldThenNew), patName);
    }
    );
  }
};

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

template <bool AllowStructParams = true> bool isConcreteAttr(Attribute a) {
  if (TypeAttr tyAttr = dyn_cast<TypeAttr>(a)) {
    return isConcreteType(tyAttr.getValue(), AllowStructParams);
  }
  if (IntegerAttr intAttr = dyn_cast<IntegerAttr>(a)) {
    return !isDynamic(intAttr);
  }
  return false;
}

namespace Step1_InstantiateStructs {

static inline bool tableOffsetIsntSymbol(MemberReadOp op) {
  return !llvm::isa_and_present<SymbolRefAttr>(op.getTableOffset().value_or(nullptr));
}

/// Implements cloning a `StructDefOp` for a specific instantiation site, using the concrete
/// parameters from the instantiation to replace parameters from the original `StructDefOp`.
class StructCloner {
  ConversionTracker &tracker_;
  ModuleOp rootMod;
  SymbolTableCollection symTables;

  class MappedTypeConverter : public TypeConverter {
    StructType origTy;
    StructType newTy;
    const DenseMap<Attribute, Attribute> &paramNameToValue;

    inline Attribute convertIfPossible(Attribute a) const {
      auto res = this->paramNameToValue.find(a);
      return (res != this->paramNameToValue.end()) ? res->second : a;
    }

  public:
    MappedTypeConverter(
        StructType originalType, StructType newType,
        /// Instantiated values for the parameter names in `originalType`
        const DenseMap<Attribute, Attribute> &paramNameToInstantiatedValue
    )
        : TypeConverter(), origTy(originalType), newTy(newType),
          paramNameToValue(paramNameToInstantiatedValue) {

      addConversion([](Type inputTy) { return inputTy; });

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
            if (TypeAttr ta = dyn_cast<TypeAttr>(a)) {
              updated.push_back(TypeAttr::get(this->convertType(ta.getValue())));
            } else {
              updated.push_back(convertIfPossible(a));
            }
          }
          return StructType::get(
              inputTy.getNameRef(), ArrayAttr::get(inputTy.getContext(), updated)
          );
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

      addConversion([this](TypeVarType inputTy) -> Type {
        // Check for replacement of parameter symbol name with a concrete type
        if (TypeAttr tyAttr = llvm::dyn_cast<TypeAttr>(convertIfPossible(inputTy.getNameRef()))) {
          Type convertedType = tyAttr.getValue();
          // Use the new type unless it contains a TypeVarType because a TypeVarType from a
          // different struct references a parameter name from that other struct, not from the
          // current struct so the reference would be invalid.
          if (isConcreteType(convertedType)) {
            return convertedType;
          }
        }
        return inputTy;
      });
    }
  };

  template <typename Impl, typename Op, typename... HandledAttrs>
  class SymbolUserHelper : public OpConversionPattern<Op> {
  private:
    const DenseMap<Attribute, Attribute> &paramNameToValue;

    SymbolUserHelper(
        TypeConverter &converter, MLIRContext *ctx, unsigned Benefit,
        const DenseMap<Attribute, Attribute> &paramNameToInstantiatedValue
    )
        : OpConversionPattern<Op>(converter, ctx, Benefit),
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
      auto res = this->paramNameToValue.find(getNameAttr(op));
      if (res == this->paramNameToValue.end()) {
        LLVM_DEBUG(llvm::dbgs() << "[StructCloner] no instantiation for " << op << '\n');
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

  class ClonedStructConstReadOpPattern
      : public SymbolUserHelper<
            ClonedStructConstReadOpPattern, ConstReadOp, IntegerAttr, FeltConstAttr> {
    SmallVector<Diagnostic> &diagnostics;

    using super =
        SymbolUserHelper<ClonedStructConstReadOpPattern, ConstReadOp, IntegerAttr, FeltConstAttr>;

  public:
    ClonedStructConstReadOpPattern(
        TypeConverter &converter, MLIRContext *ctx,
        const DenseMap<Attribute, Attribute> &paramNameToInstantiatedValue,
        SmallVector<Diagnostic> &instantiationDiagnostics
    )
        // Must use higher benefit than GeneralTypeReplacePattern so this pattern will be applied
        // instead of the GeneralTypeReplacePattern<ConstReadOp> from newGeneralRewritePatternSet().
        : super(converter, ctx, /*benefit=*/2, paramNameToInstantiatedValue),
          diagnostics(instantiationDiagnostics) {}

    Attribute getNameAttr(ConstReadOp op) const override { return op.getConstNameAttr(); }

    LogicalResult handleRewrite(
        Attribute sym, ConstReadOp op, OpAdaptor, ConversionPatternRewriter &rewriter, IntegerAttr a
    ) const {
      APInt attrValue = a.getValue();
      Type origResTy = op.getType();
      if (llvm::isa<FeltType>(origResTy)) {
        replaceOpWithNewOp<FeltConstantOp>(
            rewriter, op, FeltConstAttr::get(getContext(), attrValue)
        );
        return success();
      }

      if (llvm::isa<IndexType>(origResTy)) {
        replaceOpWithNewOp<arith::ConstantIndexOp>(rewriter, op, fromAPInt(attrValue));
        return success();
      }

      if (origResTy.isSignlessInteger(1)) {
        // Treat 0 as false and any other value as true (but give a warning if it's not 1)
        if (attrValue.isZero()) {
          replaceOpWithNewOp<arith::ConstantIntOp>(rewriter, op, false, origResTy);
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
              << "when instantiating '" << StructDefOp::getOperationName() << "' parameter \""
              << sym << "\" for this call";
          diagnostics.push_back(std::move(diag));
        }
        replaceOpWithNewOp<arith::ConstantIntOp>(rewriter, op, true, origResTy);
        return success();
      }
      return op->emitOpError().append("unexpected result type ", origResTy);
    }

    LogicalResult handleRewrite(
        Attribute, ConstReadOp op, OpAdaptor, ConversionPatternRewriter &rewriter, FeltConstAttr a
    ) const {
      replaceOpWithNewOp<FeltConstantOp>(rewriter, op, a);
      return success();
    }
  };

  class ClonedStructMemberReadOpPattern
      : public SymbolUserHelper<
            ClonedStructMemberReadOpPattern, MemberReadOp, IntegerAttr, FeltConstAttr> {
    using super =
        SymbolUserHelper<ClonedStructMemberReadOpPattern, MemberReadOp, IntegerAttr, FeltConstAttr>;

  public:
    ClonedStructMemberReadOpPattern(
        TypeConverter &converter, MLIRContext *ctx,
        const DenseMap<Attribute, Attribute> &paramNameToInstantiatedValue
    )
        // Must use higher benefit than GeneralTypeReplacePattern so this pattern will be applied
        // instead of the GeneralTypeReplacePattern<MemberReadOp> from
        // newGeneralRewritePatternSet().
        : super(converter, ctx, /*benefit=*/2, paramNameToInstantiatedValue) {}

    Attribute getNameAttr(MemberReadOp op) const override {
      return op.getTableOffset().value_or(nullptr);
    }

    template <typename Attr>
    LogicalResult handleRewrite(
        Attribute, MemberReadOp op, OpAdaptor, ConversionPatternRewriter &rewriter, Attr a
    ) const {
      rewriter.modifyOpInPlace(op, [&]() {
        op.setTableOffsetAttr(rewriter.getIndexAttr(fromAPInt(a.getValue())));
      });

      return success();
    }

    LogicalResult matchAndRewrite(
        MemberReadOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {
      if (tableOffsetIsntSymbol(op)) {
        return failure();
      }

      return super::matchAndRewrite(op, adaptor, rewriter);
    }
  };

  FailureOr<StructType> genClone(StructType typeAtCaller, ArrayRef<Attribute> typeAtCallerParams) {
    // Find the StructDefOp for the original StructType
    FailureOr<SymbolLookupResult<StructDefOp>> r = typeAtCaller.getDefinition(symTables, rootMod);
    if (failed(r)) {
      LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   skip: cannot find StructDefOp \n");
      return failure(); // getDefinition() already emits a sufficient error message
    }

    StructDefOp origStruct = r->get();
    StructType typeAtDef = origStruct.getType();
    MLIRContext *ctx = origStruct.getContext();

    // Map of StructDefOp parameter name to concrete Attribute at the current instantiation site.
    DenseMap<Attribute, Attribute> paramNameToConcrete;
    // List of concrete Attributes from the struct instantiation with `nullptr` at any positions
    // where the original attribute from the current instantiation site was not concrete. This is
    // used for generating the new struct name. See `BuildShortTypeString::from()`.
    SmallVector<Attribute> attrsForInstantiatedNameSuffix;
    // Parameter list for the new StructDefOp containing the names that must be preserved because
    // they were not assigned concrete values at the current instantiation site.
    ArrayAttr reducedParamNameList = nullptr;
    // Reduced from `typeAtCallerParams` to contain only the non-concrete Attributes.
    ArrayAttr reducedCallerParams = nullptr;
    {
      ArrayAttr paramNames = typeAtDef.getParams();

      // pre-conditions
      assert(!isNullOrEmpty(paramNames));
      assert(paramNames.size() == typeAtCallerParams.size());

      SmallVector<Attribute> remainingNames;
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
        reducedParamNameList = ArrayAttr::get(ctx, remainingNames);
        reducedCallerParams = ArrayAttr::get(ctx, nonConcreteParams);
      }
    }

    // Clone the original struct, apply the new name, and set the parameter list of the new struct
    // to contain only those that did not have concrete instantiated values.
    StructDefOp newStruct = origStruct.clone();
    newStruct.setConstParamsAttr(reducedParamNameList);
    newStruct.setSymName(
        BuildShortTypeString::from(
            typeAtCaller.getNameRef().getLeafReference().str(), attrsForInstantiatedNameSuffix
        )
    );

    // Insert 'newStruct' into the parent ModuleOp of the original StructDefOp. Use the
    // `SymbolTable::insert()` function directly so that the name will be made unique.
    ModuleOp parentModule = origStruct.getParentOp<ModuleOp>(); // parent is ModuleOp per ODS
    symTables.getSymbolTable(parentModule).insert(newStruct, Block::iterator(origStruct));
    // Retrieve the new type AFTER inserting since the name may be appended to make it unique and
    // use the remaining non-concrete parameters from the original type.
    StructType newRemoteType = newStruct.getType(reducedCallerParams);
    LLVM_DEBUG({
      llvm::dbgs() << "[StructCloner]   original def type: " << typeAtDef << '\n';
      llvm::dbgs() << "[StructCloner]   cloned def type: " << newStruct.getType() << '\n';
      llvm::dbgs() << "[StructCloner]   original remote type: " << typeAtCaller << '\n';
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
      return paramNameToConcrete.find(op.getConstNameAttr()) == paramNameToConcrete.end();
    });

    RewritePatternSet patterns = newGeneralRewritePatternSet<EmitEqualityOp>(tyConv, ctx, target);
    patterns.add<ClonedStructConstReadOpPattern>(
        tyConv, ctx, paramNameToConcrete, tracker_.delayedDiagnosticSet(newRemoteType)
    );
    patterns.add<ClonedStructMemberReadOpPattern>(tyConv, ctx, paramNameToConcrete);
    if (failed(applyFullConversion(newStruct, target, std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   instantiating body of struct failed \n");
      return failure();
    }
    return newRemoteType;
  }

public:
  StructCloner(ConversionTracker &tracker, ModuleOp root)
      : tracker_(tracker), rootMod(root), symTables() {}

  FailureOr<StructType> createInstantiatedClone(StructType orig) {
    LLVM_DEBUG(llvm::dbgs() << "[StructCloner] orig: " << orig << '\n');
    if (ArrayAttr params = orig.getParams()) {
      return genClone(orig, params.getValue());
    }
    LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   skip: nullptr for params \n");
    return failure();
  }
};

class ParameterizedStructUseTypeConverter : public TypeConverter {
  ConversionTracker &tracker_;
  StructCloner cloner;

public:
  ParameterizedStructUseTypeConverter(ConversionTracker &tracker, ModuleOp root)
      : TypeConverter(), tracker_(tracker), cloner(tracker, root) {

    addConversion([](Type inputTy) { return inputTy; });

    addConversion([this](StructType inputTy) -> StructType {
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

class CallStructFuncPattern : public OpConversionPattern<CallOp> {
  ConversionTracker &tracker_;

public:
  CallStructFuncPattern(TypeConverter &converter, MLIRContext *ctx, ConversionTracker &tracker)
      // Must use higher benefit than CallOpClassReplacePattern so this pattern will be applied
      // instead of the CallOpClassReplacePattern from newGeneralRewritePatternSet().
      : OpConversionPattern<CallOp>(converter, ctx, /*benefit=*/2), tracker_(tracker) {}

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

// This one ensures MemberDefOp types are converted even if there are no reads/writes to them.
class MemberDefOpPattern : public OpConversionPattern<MemberDefOp> {
public:
  MemberDefOpPattern(TypeConverter &converter, MLIRContext *ctx, ConversionTracker &)
      // Must use higher benefit than GeneralTypeReplacePattern so this pattern will be applied
      // instead of the GeneralTypeReplacePattern<MemberDefOp> from newGeneralRewritePatternSet().
      : OpConversionPattern<MemberDefOp>(converter, ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(
      MemberDefOp op, OpAdaptor adapter, ConversionPatternRewriter &rewriter
  ) const override {
    LLVM_DEBUG(llvm::dbgs() << "[MemberDefOpPattern] MemberDefOp: " << op << '\n');

    Type oldMemberType = op.getType();
    Type newMemberType = getTypeConverter()->convertType(oldMemberType);
    if (oldMemberType == newMemberType) {
      // nothing changed
      return failure();
    }
    rewriter.modifyOpInPlace(op, [&op, &newMemberType]() { op.setType(newMemberType); });
    return success();
  }
};

LogicalResult run(ModuleOp modOp, ConversionTracker &tracker) {
  MLIRContext *ctx = modOp.getContext();
  ParameterizedStructUseTypeConverter tyConv(tracker, modOp);
  ConversionTarget target = newConverterDefinedTarget<>(tyConv, ctx);
  RewritePatternSet patterns = newGeneralRewritePatternSet(tyConv, ctx, target);
  patterns.add<CallStructFuncPattern, MemberDefOpPattern>(tyConv, ctx, tracker);
  return applyPartialConversion(modOp, target, std::move(patterns));
}

} // namespace Step1_InstantiateStructs

namespace Step2_Unroll {

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
} // namespace Step2_Unroll

namespace Step3_InstantiateAffineMaps {

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
        if (auto constOps = Step3_InstantiateAffineMaps::getConstantIntValues(currMapOpsCast)) {
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
                    "Cannot fold affine_map for ", aspect, " ", out.paramsOfStructTy.size(),
                    " due to divide by 0 or modulus with negative divisor"
                )
                .report();
            return failure();
          }
          if (failed(foldResult)) {
            // Diagnostic remark: could be removed for release builds if too noisy
            op->emitRemark()
                .append(
                    "Folding affine_map for ", aspect, " ", out.paramsOfStructTy.size(), " failed"
                )
                .report();
            return failure();
          }
          if (result.size() != 1) {
            // Diagnostic remark: could be removed for release builds if too noisy
            op->emitRemark()
                .append(
                    "Folding affine_map for ", aspect, " ", out.paramsOfStructTy.size(),
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
      // nothing changed
      return failure();
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
      // nothing changed
      return failure();
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
      if (!isConcreteAttr<>(fromCall)) {
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

} // namespace Step3_InstantiateAffineMaps

namespace Step4_PropagateTypes {

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
    FailureOr<StructDefOp> parentRes = getParentOfType<StructDefOp>(op);
    assert(succeeded(parentRes) && "MemberDefOp parent is always StructDefOp"); // per ODS def

    // If the symbol is used by a MemberWriteOp with a different result type then change
    // the type of the MemberDefOp to match the MemberWriteOp result type.
    Type newType = nullptr;
    if (auto memberUsers = llzk::getSymbolUses(op, parentRes.value())) {
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
                    diag.attachNote(*newTypeLoc).append("type written here is ", newType);
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
      // nothing changed
      return failure();
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
      // nothing changed
      return failure();
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
      // nothing changed
      return failure();
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
/// This only applies to global (i.e., non-struct) functions because the functions within structs
/// only return StructType or nothing and propagating those can result in bringing un-instantiated
/// types from a templated struct into the current call which will give errors.
class UpdateGlobalCallOpTypes final : public OpRewritePattern<CallOp> {
  ConversionTracker &tracker_;

public:
  UpdateGlobalCallOpTypes(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  LogicalResult matchAndRewrite(CallOp op, PatternRewriter &rewriter) const override {
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
      // nothing changed
      return failure();
    }
    if (!tracker_.areLegalConversions(
            op.getResultTypes(), targetFunc.getFunctionType().getResults(),
            "UpdateGlobalCallOpTypes"
        )) {
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "[UpdateGlobalCallOpTypes] replaced " << op);
    CallOp newOp = replaceOpWithNewOp<CallOp>(rewriter, op, targetFunc, op.getArgOperands());
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
      UpdateGlobalCallOpTypes,      // CallOp, targeting non-struct functions
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
} // namespace Step4_PropagateTypes

namespace Step5_Cleanup {

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

struct FromKeepSet : public CleanupBase {
  using CleanupBase::CleanupBase;

  /// Erase all StructDefOp that are not reachable (via calls, types, or symbol usage) from one of
  /// the StructDefOp given or from some global def or free function (since this pass does not
  /// remove either of those, any symbols reachable from them must not be removed).
  LogicalResult eraseUnreachableFrom(ArrayRef<StructDefOp> keep) {
    // Initialize roots from the given StructDefOp instances
    SetVector<SymbolOpInterface> roots(keep.begin(), keep.end());
    // Add GlobalDefOp and "free functions" to the set of roots
    rootMod.walk([&roots](Operation *op) {
      if (global::GlobalDefOp gdef = llvm::dyn_cast<global::GlobalDefOp>(op)) {
        roots.insert(gdef);
      } else if (function::FuncDefOp fdef = llvm::dyn_cast<function::FuncDefOp>(op)) {
        if (!fdef.isInStruct()) {
          roots.insert(fdef);
        }
      }
    });

    // Use a SymbolDefTree to find all Symbol defs reachable from one of the root nodes. Then
    // collect all Symbol uses reachable from those def nodes. These are the symbols that should
    // be preserved. All other symbol defs should be removed.
    llvm::df_iterator_default_set<const SymbolUseGraphNode *> symbolsToKeep;
    for (size_t i = 0; i < roots.size(); ++i) { // iterate for safe insertion
      SymbolOpInterface keepRoot = roots[i];
      LLVM_DEBUG({ llvm::dbgs() << "[EraseUnreachable] root: " << keepRoot << '\n'; });
      const SymbolDefTreeNode *keepRootNode = defTree.lookupNode(keepRoot);
      assert(keepRootNode && "every struct def must be in the def tree");
      for (const SymbolDefTreeNode *reachableDefNode : llvm::depth_first(keepRootNode)) {
        LLVM_DEBUG({
          llvm::dbgs() << "[EraseUnreachable] can reach: " << reachableDefNode->getOp() << '\n';
        });
        if (SymbolOpInterface reachableDef = reachableDefNode->getOp()) {
          // Use 'depth_first_ext()' to get all symbol uses reachable from the current Symbol def
          // node. There are no uses if the node is not in the graph. Within the loop that populates
          // 'depth_first_ext()', also check if the symbol is a StructDefOp and ensure it is in
          // 'roots' so the outer loop will ensure that all symbols reachable from it are preserved.
          if (const SymbolUseGraphNode *useGraphNodeForDef = useGraph.lookupNode(reachableDef)) {
            for (const SymbolUseGraphNode *usedSymbolNode :
                 depth_first_ext(useGraphNodeForDef, symbolsToKeep)) {
              LLVM_DEBUG({
                llvm::dbgs() << "[EraseUnreachable]   uses symbol: "
                             << usedSymbolNode->getSymbolPath() << '\n';
              });
              // Ignore struct/template parameter symbols (before doing the lookup below because it
              // would fail anyway and then cause the "failed" case to be triggered unnecessarily).
              if (usedSymbolNode->isStructParam()) {
                continue;
              }
              // If `usedSymbolNode` references a StructDefOp, ensure it's considered in the roots.
              auto lookupRes = usedSymbolNode->lookupSymbol(tables);
              if (failed(lookupRes)) {
                LLVM_DEBUG(useGraph.dumpToDotFile());
                return failure();
              }
              //  If loaded via an IncludeOp it's not in the current AST anyway so ignore.
              if (lookupRes->viaInclude()) {
                continue;
              }
              if (StructDefOp asStruct = llvm::dyn_cast<StructDefOp>(lookupRes->get())) {
                bool insertRes = roots.insert(asStruct);
                (void)insertRes; // tell compiler it's intentionally unused in release builds
                LLVM_DEBUG({
                  if (insertRes) {
                    llvm::dbgs() << "[EraseUnreachable]  found another root: " << asStruct << '\n';
                  }
                });
              }
            }
          }
        }
      }
    }

    rootMod.walk([this, &symbolsToKeep](StructDefOp op) {
      const SymbolUseGraphNode *n = this->useGraph.lookupNode(op);
      assert(n);
      if (!symbolsToKeep.contains(n)) {
        LLVM_DEBUG(llvm::dbgs() << "[EraseUnreachable] removing: " << op.getSymName() << '\n');
        op.erase();
      }

      return WalkResult::skip(); // StructDefOp cannot be nested
    });

    return success();
  }
};

struct FromEraseSet : public CleanupBase {

  /// Note: paths in `tryToErase` should be relative to `root` (which is likely the "top root")
  FromEraseSet(
      ModuleOp root, const SymbolDefTree &symDefTree, const SymbolUseGraph &symUseGraph,
      DenseSet<SymbolRefAttr> &&tryToErasePaths
  )
      : CleanupBase(root, symDefTree, symUseGraph) {
    // Convert the set of paths targeted for erasure into a set of the StructDefOp
    for (SymbolRefAttr path : tryToErasePaths) {
      Operation *lookupFrom = rootMod.getOperation();
      auto res = lookupSymbolIn<StructDefOp>(tables, path, lookupFrom, lookupFrom);
      assert(succeeded(res) && "inputs must be valid StructDefOp references");
      if (!res->viaInclude()) { // do not remove if it's from another source file
        tryToErase.insert(res->get());
      }
    }
  }

  LogicalResult eraseUnusedStructs() {
    // Collect the subset of 'tryToErase' that has no remaining uses.
    for (StructDefOp sd : tryToErase) {
      collectSafeToErase(sd);
    }
    // The `visitedPlusSafetyResult` will contain FuncDefOp w/in the StructDefOp so just a single
    // loop to `dyn_cast` and `erase()` will cause `use-after-free` errors w/in the `dyn_cast`.
    // Instead, reduce the map to only those that should be erased and erase in a separate loop.
    for (auto it = visitedPlusSafetyResult.begin(); it != visitedPlusSafetyResult.end(); ++it) {
      if (!it->second || !llvm::isa<StructDefOp>(it->first.getOperation())) {
        visitedPlusSafetyResult.erase(it);
      }
    }
    for (auto &[sym, _] : visitedPlusSafetyResult) {
      LLVM_DEBUG(llvm::dbgs() << "[EraseIfUnused] removing: " << sym.getNameAttr() << '\n');
      sym.erase();
    }
    return success();
  }

  const DenseSet<StructDefOp> &getTryToEraseSet() const { return tryToErase; }

private:
  /// The initial set of structs that this should try to erase (if there are no other uses).
  DenseSet<StructDefOp> tryToErase;
  /// Track visited nodes to avoid cycles (for example, a struct has its functions as children in
  /// the def graph but the opposite direction edges exist in the use graph) and map if they were
  /// determined safe to remove or not.
  DenseMap<SymbolOpInterface, bool> visitedPlusSafetyResult;
  /// Cache results of 'lookup()' for performance.
  DenseMap<const SymbolUseGraphNode *, SymbolOpInterface> lookupCache;

  /// The main checks to determine if a SymbolOp (but especially a StructDefOp) is safe to erase
  /// without leaving any dangling references to it.
  bool collectSafeToErase(SymbolOpInterface check) {
    assert(check); // pre-condition

    // If previously visited, return the safety result.
    auto visited = visitedPlusSafetyResult.find(check);
    if (visited != visitedPlusSafetyResult.end()) {
      return visited->second;
    }

    // If it's a StructDefOp that is not in `tryToErase` then it cannot be erased.
    if (StructDefOp sd = llvm::dyn_cast<StructDefOp>(check.getOperation())) {
      if (!tryToErase.contains(sd)) {
        visitedPlusSafetyResult[check] = false;
        return false;
      }
    }

    // Otherwise, temporarily mark as safe b/c a node cannot keep itself live (and this prevents
    // the recursion from getting stuck in an infinite loop).
    visitedPlusSafetyResult[check] = true;

    // Check if it's safe according to both the def tree and use graph.
    // Note: every symbol must have a def node but module symbols may not have a use node.
    if (collectSafeToErase(defTree.lookupNode(check))) {
      auto useNode = useGraph.lookupNode(check);
      assert(useNode || llvm::isa<ModuleOp>(check.getOperation()));
      if (!useNode || collectSafeToErase(useNode)) {
        return true;
      }
    }

    // Otherwise, revert the safety decision and return it.
    visitedPlusSafetyResult[check] = false;
    return false;
  }

  /// A def tree node is safe if it has no parent or its parent's SymbolOp is safe.
  bool collectSafeToErase(const SymbolDefTreeNode *check) {
    assert(check); // pre-condition
    if (const SymbolDefTreeNode *p = check->getParent()) {
      if (SymbolOpInterface checkOp = p->getOp()) { // safe if parent is root
        return collectSafeToErase(checkOp);
      }
    }
    return true;
  }

  /// A use graph node is safe if it has no predecessors (i.e., users) or all have safe SymbolOp.
  bool collectSafeToErase(const SymbolUseGraphNode *check) {
    assert(check); // pre-condition
    for (const SymbolUseGraphNode *p : check->predecessorIter()) {
      if (SymbolOpInterface checkOp = cachedLookup(p)) { // safe if via IncludeOp
        if (!collectSafeToErase(checkOp)) {
          return false;
        }
      }
    }
    return true;
  }

  /// Find the SymbolOpInterface for the given graph node, utilizing a cache for repeat lookups.
  /// Returns `nullptr` if the node is loaded via an IncludeOp. A symbol loaded from an included
  /// file is not subject to removal by this pass. Further, it cannot serve as an anchor/root for a
  /// symbol that is defined in the current file because it can neither define nor use such symbols.
  SymbolOpInterface cachedLookup(const SymbolUseGraphNode *node) {
    assert(node && "must provide a node"); // pre-condition
    // Check for cached result
    auto fromCache = lookupCache.find(node);
    if (fromCache != lookupCache.end()) {
      return fromCache->second;
    }
    // Otherwise, perform lookup and cache
    auto lookupRes = node->lookupSymbol(tables);
    assert(succeeded(lookupRes) && "graph contains node with invalid path");
    assert(lookupRes->get() != nullptr && "lookup must return an Operation");
    // If loaded via an IncludeOp it's not in the current AST anyway so ignore.
    // NOTE: The SymbolUseGraph does contain nodes for struct parameters which cannot cast to
    // SymbolOpInterface. However, those will always be leaf nodes in the SymbolUseGraph and
    // therefore will not be traversed by this analysis so directly casting is fine.
    SymbolOpInterface actualRes =
        lookupRes->viaInclude() ? nullptr : llvm::cast<SymbolOpInterface>(lookupRes->get());
    // Cache and return
    lookupCache[node] = actualRes;
    assert((!actualRes == lookupRes->viaInclude()) && "not found iff included"); // post-condition
    return actualRes;
  }
};

} // namespace Step5_Cleanup

class FlatteningPass : public llzk::polymorphic::impl::FlatteningPassBase<FlatteningPass> {

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
    // If the cleanup mode is set to remove anything not reachable from the main struct, do an
    // initial pass to remove things that are not reachable (as an optimization) because creating
    // an instantiated version of a struct will not cause something to become reachable that was
    // not already reachable in parameterized form.
    if (cleanupMode == StructCleanupMode::MainAsRoot) {
      if (failed(eraseUnreachableFromMainStruct(modOp))) {
        return failure();
      }
    }

    {
      // Preliminary step: remove empty parameter lists from structs
      OpPassManager nestedPM(ModuleOp::getOperationName());
      nestedPM.addPass(createEmptyParamListRemoval());
      if (failed(runPipeline(nestedPM, modOp))) {
        return failure();
      }
    }

    ConversionTracker tracker;
    unsigned loopCount = 0;
    do {
      ++loopCount;
      if (loopCount > iterationLimit) {
        llvm::errs() << DEBUG_TYPE << " exceeded the limit of " << iterationLimit
                     << " iterations!\n";
        return failure();
      }
      tracker.resetModifiedFlag();

      // Find calls to "compute()" that return a parameterized struct and replace it to call a
      // flattened version of the struct that has parameters replaced with the constant values.
      // Create the necessary instantiated/flattened struct in the same location as the original.
      if (failed(Step1_InstantiateStructs::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while replacing concrete-parameter struct types\n";
        return failure();
      }

      // Unroll loops with known iterations.
      if (failed(Step2_Unroll::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while unrolling loops\n";
        return failure();
      }

      // Instantiate affine_map parameters of StructType and ArrayType.
      if (failed(Step3_InstantiateAffineMaps::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while instantiating `affine_map` parameters\n";
        return failure();
      }

      // Propagate updated types using the semantics of various ops.
      if (failed(Step4_PropagateTypes::run(modOp, tracker))) {
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

    // Perform cleanup according to the 'cleanupMode' option.
    switch (cleanupMode) {
    case StructCleanupMode::MainAsRoot:
      return eraseUnreachableFromMainStruct(modOp, false);
    case StructCleanupMode::ConcreteAsRoot:
      return eraseUnreachableFromConcreteStructs(modOp);
    case StructCleanupMode::Preimage:
      return erasePreimageOfInstantiations(modOp, tracker);
    case StructCleanupMode::Disabled:
      return success();
    }
    llvm_unreachable("switch cases cover all options");
  }

  // Erase parameterized structs that were replaced with concrete instantiations.
  LogicalResult erasePreimageOfInstantiations(ModuleOp rootMod, const ConversionTracker &tracker) {
    // TODO: The names from getInstantiatedStructNames() are NOT guaranteed to be paths from the
    // "top root" and they also do not indicate a root module so there could be ambiguity. This is a
    // broader problem in the FlatteningPass itself so let's just assume, for now, that these are
    // paths from the "top root". See [LLZK-286].
    Step5_Cleanup::FromEraseSet cleaner(
        rootMod, getAnalysis<SymbolDefTree>(), getAnalysis<SymbolUseGraph>(),
        tracker.getInstantiatedStructNames()
    );
    LogicalResult res = cleaner.eraseUnusedStructs();
    if (succeeded(res)) {
      // Warn about any structs that were instantiated but still have uses elsewhere.
      const SymbolUseGraph *useGraph = nullptr;
      rootMod->walk([this, &cleaner, &useGraph](StructDefOp op) {
        if (cleaner.getTryToEraseSet().contains(op)) {
          // If needed, rebuild use graph to reflect deletions.
          if (!useGraph) {
            useGraph = &getAnalysis<SymbolUseGraph>();
          }
          // If the op has any users, report the warning.
          if (useGraph->lookupNode(op)->hasPredecessor()) {
            op.emitWarning("Parameterized struct still has uses!").report();
          }
        }
        return WalkResult::skip(); // StructDefOp cannot be nested
      });
    }
    return res;
  }

  LogicalResult eraseUnreachableFromConcreteStructs(ModuleOp rootMod) {
    SmallVector<StructDefOp> roots;
    rootMod.walk([&roots](StructDefOp op) {
      // Note: no need to check if the ConstParamsAttr is empty since `EmptyParamRemovalPass`
      // ran earlier.
      if (!op.hasConstParamsAttr()) {
        roots.push_back(op);
      }
      return WalkResult::skip(); // StructDefOp cannot be nested
    });

    Step5_Cleanup::FromKeepSet cleaner(
        rootMod, getAnalysis<SymbolDefTree>(), getAnalysis<SymbolUseGraph>()
    );
    return cleaner.eraseUnreachableFrom(roots);
  }

  LogicalResult eraseUnreachableFromMainStruct(ModuleOp rootMod, bool emitWarning = true) {
    Step5_Cleanup::FromKeepSet cleaner(
        rootMod, getAnalysis<SymbolDefTree>(), getAnalysis<SymbolUseGraph>()
    );
    FailureOr<SymbolLookupResult<StructDefOp>> mainOpt =
        getMainInstanceDef(cleaner.tables, rootMod.getOperation());
    if (failed(mainOpt)) {
      return failure();
    }
    SymbolLookupResult<StructDefOp> main = mainOpt.value();
    if (emitWarning && !main) {
      // Emit warning if there is no main specified because all structs may be removed (only
      // structs that are reachable from a global def or free function will be preserved since
      // those constructs are not candidate for removal in this pass).
      rootMod.emitWarning()
          .append(
              "using option '", cleanupMode.getArgStr(), '=',
              stringifyStructCleanupMode(StructCleanupMode::MainAsRoot), "' with no \"",
              MAIN_ATTR_NAME, "\" attribute on the top-level module may remove all structs!"
          )
          .report();
    }
    return cleaner.eraseUnreachableFrom(
        main ? ArrayRef<StructDefOp> {*main} : ArrayRef<StructDefOp> {}
    );
  }
};

} // namespace

std::unique_ptr<Pass> llzk::polymorphic::createFlatteningPass() {
  return std::make_unique<FlatteningPass>();
};
