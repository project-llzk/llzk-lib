//===-- TypeVarInferencePass.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-infer-tvar` pass.
///
/// The pass specializes polymorphic template type variables in place when the
/// body of a template proves that a `!poly.tvar` must actually be a concrete
/// type. The motivating case is a `poly.unifiable_cast` between a value whose
/// type mentions a template type variable and a value whose type is concrete.
///
/// For each `poly.template`, the pass:
///   1. Collects concrete type inferences for `poly.param` definitions of the
///      form `poly.param @T : !poly.tvar<@T>`.
///   2. Rejects conflicting inferences for the same template parameter or the
///      same SSA value.
///   3. Rewrites function signatures, body value types, and type-bearing
///      attributes that mention the inferred type variables.
///   4. Removes `poly.unifiable_cast` operations that become identity casts.
///   5. Removes the resolved `poly.param` definitions and trims callers'
///      template parameter lists accordingly.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolLookup.h"
#include "llzk/Util/SymbolTableLLZK.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>

#include <memory>

// Include the generated base pass class definitions.
namespace llzk::polymorphic {
#define GEN_PASS_DEF_TYPEVARINFERENCEPASS
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h.inc"
} // namespace llzk::polymorphic

#define DEBUG_TYPE "llzk-infer-tvar"

using namespace mlir;
using namespace llzk;
using namespace llzk::array;
using namespace llzk::component;
using namespace llzk::function;
using namespace llzk::pod;
using namespace llzk::polymorphic;

namespace {

/// A concrete type inferred at a source location.
///
/// The location is retained so a later conflicting inference can point back to
/// the original proof site with a note.
struct InferredType {
  Type type;
  Location loc;
};

/// Return the pieces of a symbol reference as `StringAttr`s.
///
/// The pass uses `StringAttr` as the stable DenseMap/DenseSet key for template
/// parameter names and symbol paths. This helper preserves the symbol-reference
/// structure while avoiding repeated leaf extraction at call sites.
static SmallVector<StringAttr> getStringPieces(SymbolRefAttr ref) {
  return llvm::to_vector(llvm::map_range(getPieces(ref), [](FlatSymbolRefAttr piece) {
    return piece.getAttr();
  }));
}

/// Return true when `op` is exactly the self-typed type variable parameter this
/// pass knows how to remove.
///
/// A parameter is eligible only when its declared type is `!poly.tvar<@Name>`
/// and the referenced type variable name matches the parameter symbol name. This
/// prevents the pass from deleting value parameters or constrained parameters
/// that merely mention a type variable.
static bool isTypeVarParam(TemplateParamOp op) {
  std::optional<Type> declaredTy = op.getTypeOpt();
  if (!declaredTy) {
    return false;
  }
  auto tvarTy = llvm::dyn_cast<TypeVarType>(*declaredTy);
  return tvarTy && tvarTy.getRefName() == op.getName();
}

/// Merge nested array dimensions produced by replacing an array element type.
///
/// If `array<4 x !poly.tvar<@T>>` is rewritten with `@T -> array<8 x index>`, the canonical
/// aggregate shape should become `array<4,8 x index>` rather than an array whose element type
/// is another array because the latter is not allowed in LLZK IR.
static ArrayType flattenInstantiatedArrayType(ArrayType inputTy, Type convertedElemTy) {
  SmallVector<Attribute> mergedDims(inputTy.getDimensionSizes());
  while (ArrayType nestedArrTy = llvm::dyn_cast<ArrayType>(convertedElemTy)) {
    llvm::append_range(mergedDims, nestedArrTy.getDimensionSizes());
    convertedElemTy = nestedArrTy.getElementType();
  }
  return ArrayType::get(convertedElemTy, mergedDims);
}

/// Converts types and attributes by replacing inferred template type variables.
///
/// The converter is scoped to one `poly.template`. Besides direct
/// `!poly.tvar<@T>` replacement, it recursively rewrites aggregate types and
/// trims template parameter lists whose positions correspond to resolved
/// parameters. Positional trimming matters because callers and owned struct
/// types encode template arguments as arrays rather than by name.
class TypeVarReplacementConverter {
  /// Context used to allocate replacement aggregate types and attributes.
  MLIRContext *ctx_;
  /// Fully-qualified symbol path of the template currently being rewritten.
  SmallVector<StringAttr> templatePath_;
  /// Original template parameter order before any resolved parameters are removed.
  SmallVector<StringAttr> oldParamOrder_;
  /// Names of parameters that will be removed from the template.
  DenseSet<StringAttr> removedParams_;
  /// Concrete type replacement for each inferred template type variable.
  DenseMap<StringAttr, Type> replacements_;

public:
  /// Create a converter for a single template rewrite.
  ///
  /// `oldParamOrder` is captured before erasing any `poly.param` operations, so
  /// call-site template argument lists can be trimmed using their original
  /// positional layout.
  TypeVarReplacementConverter(
      MLIRContext *ctx, ArrayRef<StringAttr> templatePath, ArrayRef<StringAttr> oldParamOrder,
      const DenseMap<StringAttr, Type> &replacements
  )
      : ctx_(ctx), templatePath_(templatePath), oldParamOrder_(oldParamOrder),
        replacements_(replacements) {
    for (const auto &entry : replacements_) {
      removedParams_.insert(entry.first);
    }
  }

  /// Convert a type by recursively replacing inferred `!poly.tvar` occurrences.
  ///
  /// Supported containers are array, struct, POD, and function types. Unknown
  /// or unsupported types are returned unchanged.
  Type convertType(Type ty) const {
    if (!ty) {
      return ty;
    }
    if (auto tvarTy = llvm::dyn_cast<TypeVarType>(ty)) {
      auto it = replacements_.find(tvarTy.getNameRef().getAttr());
      return it == replacements_.end() ? ty : it->second;
    }
    if (auto arrTy = llvm::dyn_cast<ArrayType>(ty)) {
      Type newElemTy = convertType(arrTy.getElementType());
      if (newElemTy == arrTy.getElementType()) {
        return ty;
      }
      return flattenInstantiatedArrayType(arrTy, newElemTy);
    }
    if (auto structTy = llvm::dyn_cast<StructType>(ty)) {
      return convertStructType(structTy);
    }
    if (auto podTy = llvm::dyn_cast<PodType>(ty)) {
      SmallVector<RecordAttr> newRecords;
      bool changed = false;
      for (RecordAttr record : podTy.getRecords()) {
        Type newRecordTy = convertType(record.getType());
        newRecords.push_back(RecordAttr::get(ctx_, record.getName(), newRecordTy));
        changed |= newRecordTy != record.getType();
      }
      return changed ? PodType::get(ctx_, newRecords) : ty;
    }
    if (auto funcTy = llvm::dyn_cast<FunctionType>(ty)) {
      SmallVector<Type> newInputs;
      SmallVector<Type> newResults;
      bool changed = convertTypes(funcTy.getInputs(), newInputs);
      changed |= convertTypes(funcTy.getResults(), newResults);
      return changed ? FunctionType::get(ctx_, newInputs, newResults) : ty;
    }
    return ty;
  }

  /// Convert an attribute that may contain types needing replacement.
  ///
  /// This currently handles `TypeAttr` and nested `ArrayAttr`, which covers the
  /// template parameter lists and type-encoded op attributes used by the
  /// polymorphic/function dialects.
  Attribute convertAttr(Attribute attr) const {
    if (!attr) {
      return attr;
    }
    if (auto tyAttr = llvm::dyn_cast<TypeAttr>(attr)) {
      Type newTy = convertType(tyAttr.getValue());
      return newTy == tyAttr.getValue() ? attr : TypeAttr::get(newTy);
    }
    if (auto arrAttr = llvm::dyn_cast<ArrayAttr>(attr)) {
      SmallVector<Attribute> newAttrs;
      bool changed = false;
      for (Attribute nested : arrAttr.getValue()) {
        Attribute newNested = convertAttr(nested);
        newAttrs.push_back(newNested);
        changed |= newNested != nested;
      }
      return changed ? ArrayAttr::get(ctx_, newAttrs) : attr;
    }
    return attr;
  }

  /// Convert and trim a call-site or type-site template parameter list.
  ///
  /// If the list still has the original arity, entries corresponding to resolved
  /// `poly.param`s are removed. Remaining entries are recursively converted in
  /// case they mention a now-concrete type variable. A fully removed list is
  /// represented as `nullptr` so the printer elides the template argument
  /// syntax.
  ArrayAttr convertTemplateParams(ArrayAttr params) const {
    if (!params) {
      return nullptr;
    }
    if (params.size() != oldParamOrder_.size()) {
      return params;
    }
    SmallVector<Attribute> kept;
    for (auto [paramName, attr] : llvm::zip_equal(oldParamOrder_, params.getValue())) {
      if (!removedParams_.contains(paramName)) {
        kept.push_back(convertAttr(attr));
      }
    }
    return kept.empty() ? nullptr : ArrayAttr::get(ctx_, kept);
  }

private:
  /// Convert every type in `oldTypes`, returning whether any entry changed.
  bool convertTypes(TypeRange oldTypes, SmallVectorImpl<Type> &newTypes) const {
    bool changed = false;
    newTypes.reserve(oldTypes.size());
    for (Type oldTy : oldTypes) {
      Type newTy = convertType(oldTy);
      newTypes.push_back(newTy);
      changed |= newTy != oldTy;
    }
    return changed;
  }

  /// Return true when `structTy` names a symbol nested inside the template being
  /// rewritten.
  ///
  /// Owned struct types use the same template parameter list as their enclosing
  /// template, so resolved parameter positions can be removed from their
  /// `StructType` parameter arrays.
  bool isOwnedStructType(StructType structTy) const {
    SmallVector<StringAttr> structPath = getStringPieces(structTy.getNameRef());
    return structPath.size() > templatePath_.size() &&
           std::equal(templatePath_.begin(), templatePath_.end(), structPath.begin());
  }

  /// Convert a `StructType`, including its parameter attributes.
  ///
  /// For structs defined inside the current template, parameter entries for
  /// resolved type variables are removed by original position. For all structs,
  /// remaining parameter attributes are recursively converted.
  StructType convertStructType(StructType structTy) const {
    ArrayAttr params = structTy.getParams();
    if (!params) {
      return structTy;
    }

    SmallVector<Attribute> newParams;
    bool changed = false;
    bool removeOwnedParams = isOwnedStructType(structTy) && params.size() == oldParamOrder_.size();
    for (auto indexedAttr : llvm::enumerate(params.getValue())) {
      unsigned index = indexedAttr.index();
      Attribute attr = indexedAttr.value();
      if (removeOwnedParams && removedParams_.contains(oldParamOrder_[index])) {
        changed = true;
        continue;
      }
      Attribute newAttr = convertAttr(attr);
      newParams.push_back(newAttr);
      changed |= newAttr != attr;
    }
    return changed ? StructType::get(structTy.getNameRef(), ArrayAttr::get(ctx_, newParams))
                   : structTy;
  }
};

/// All inference state collected for one `poly.template`.
struct TemplateInferenceInfo {
  /// The template being analyzed and potentially rewritten.
  TemplateOp templateOp;
  /// Fully-qualified symbol path of `templateOp`.
  SmallVector<StringAttr> templatePath;
  /// Template parameter names in their original order.
  SmallVector<StringAttr> oldParamOrder;
  /// Eligible `poly.param @T : !poly.tvar<@T>` definitions by parameter name.
  DenseMap<StringAttr, TemplateParamOp> typeVarParams;
  /// Concrete replacement type inferred for each eligible type variable.
  DenseMap<StringAttr, InferredType> replacements;
};

/// Walk a template body and collect all type-variable inferences from
/// `poly.unifiable_cast` operations.
///
/// `poly.unifiable_cast` is the proof source for this pass: when one side of the
/// cast is a template type variable and the other side is concrete, the cast
/// establishes the concrete replacement that later makes the cast redundant.
class TypeVarInferenceCollector {
  /// Per-template inference output and eligible parameter metadata.
  TemplateInferenceInfo &inferenceInfo;
  /// Concrete inferences for SSA values observed during collection.
  DenseMap<Value, InferredType> byValue;

public:
  explicit TypeVarInferenceCollector(TemplateInferenceInfo &info) : inferenceInfo(info) {}

  LogicalResult collect() {
    WalkResult result = inferenceInfo.templateOp.walk([this](UnifiableCastOp castOp) {
      return WalkResult(collectTypePairInferences(
          castOp.getInput().getType(), castOp.getResult().getType(), castOp.getInput(),
          castOp.getResult(), castOp.getLoc()
      ));
    });
    return failure(result.wasInterrupted());
  }

private:
  /// Record one concrete type inference and diagnose conflicts.
  ///
  /// Inferences are tracked in two dimensions:
  ///   * by template parameter, so `@T` cannot be proven to be two different
  ///     concrete types;
  ///   * by SSA value, so a single value cannot be rewritten to incompatible
  ///     concrete types through separate casts.
  ///
  /// Non-eligible parameters and non-concrete candidate types are ignored because
  /// this pass only removes template type variables once a concrete replacement is
  /// known.
  LogicalResult recordInference(StringAttr paramName, Type inferredTy, Value value, Location loc) {
    if (!inferenceInfo.typeVarParams.contains(paramName) || !isConcreteType(inferredTy)) {
      return success();
    }

    auto reportConflict = [&](StringRef kind, Location originalLoc, Type originalTy) {
      InFlightDiagnostic diag = emitError(loc) << "conflicting inferred type for " << kind << " @"
                                               << paramName.getValue() << ": " << originalTy
                                               << " vs " << inferredTy;
      diag.attachNote(originalLoc) << "previous inference here";
      return diag;
    };

    auto byParamIt = inferenceInfo.replacements.find(paramName);
    if (byParamIt == inferenceInfo.replacements.end()) {
      inferenceInfo.replacements.try_emplace(paramName, InferredType {inferredTy, loc});
    } else if (byParamIt->second.type != inferredTy) {
      return reportConflict("template parameter", byParamIt->second.loc, byParamIt->second.type);
    }

    if (value) {
      auto byValueIt = byValue.find(value);
      if (byValueIt == byValue.end()) {
        byValue.try_emplace(value, InferredType {inferredTy, loc});
      } else if (byValueIt->second.type != inferredTy) {
        return reportConflict("SSA value using", byValueIt->second.loc, byValueIt->second.type);
      }
    }
    return success();
  }

  /// Collect inferences from a pair of types known to unify.
  ///
  /// The direct case is `!poly.tvar<@T>` on one side and a concrete type on the
  /// other. The function also descends into array element types and matching
  /// struct type parameters so an aggregate shape can force a nested type
  /// variable. If one side is still a type variable but its SSA value already has
  /// a concrete inference from an earlier cast, that concrete value-level proof is
  /// propagated through chained casts.
  LogicalResult
  collectTypePairInferences(Type lhs, Type rhs, Value lhsValue, Value rhsValue, Location loc) {
    if (auto lhsTvar = llvm::dyn_cast<TypeVarType>(lhs)) {
      if (failed(recordInference(lhsTvar.getNameRef().getAttr(), rhs, lhsValue, loc))) {
        return failure();
      }
      if (rhsValue) {
        auto rhsValueIt = byValue.find(rhsValue);
        if (rhsValueIt != byValue.end() &&
            failed(recordInference(
                lhsTvar.getNameRef().getAttr(), rhsValueIt->second.type, lhsValue, loc
            ))) {
          return failure();
        }
      }
    }
    if (auto rhsTvar = llvm::dyn_cast<TypeVarType>(rhs)) {
      if (failed(recordInference(rhsTvar.getNameRef().getAttr(), lhs, rhsValue, loc))) {
        return failure();
      }
      if (lhsValue) {
        auto lhsValueIt = byValue.find(lhsValue);
        if (lhsValueIt != byValue.end() &&
            failed(recordInference(
                rhsTvar.getNameRef().getAttr(), lhsValueIt->second.type, rhsValue, loc
            ))) {
          return failure();
        }
      }
    }

    if (auto lhsArr = llvm::dyn_cast<ArrayType>(lhs)) {
      if (auto rhsArr = llvm::dyn_cast<ArrayType>(rhs)) {
        return collectTypePairInferences(
            lhsArr.getElementType(), rhsArr.getElementType(), Value(), Value(), loc
        );
      }
    }

    if (auto lhsStruct = llvm::dyn_cast<StructType>(lhs)) {
      auto rhsStruct = llvm::dyn_cast<StructType>(rhs);
      if (!rhsStruct) {
        return success();
      }
      ArrayRef<Attribute> lhsParams =
          lhsStruct.getParams() ? lhsStruct.getParams().getValue() : ArrayRef<Attribute> {};
      ArrayRef<Attribute> rhsParams =
          rhsStruct.getParams() ? rhsStruct.getParams().getValue() : ArrayRef<Attribute> {};
      if (lhsParams.size() != rhsParams.size()) {
        return success();
      }
      for (auto [lhsAttr, rhsAttr] : llvm::zip_equal(lhsParams, rhsParams)) {
        auto lhsTyAttr = llvm::dyn_cast<TypeAttr>(lhsAttr);
        auto rhsTyAttr = llvm::dyn_cast<TypeAttr>(rhsAttr);
        if (lhsTyAttr && rhsTyAttr &&
            failed(collectTypePairInferences(
                lhsTyAttr.getValue(), rhsTyAttr.getValue(), Value(), Value(), loc
            ))) {
          return failure();
        }
      }
    }

    return success();
  }
};

/// Rewrite a function type and keep the entry block arguments in sync.
///
/// `function.def` stores its function signature separately from the entry block
/// argument types, so both surfaces must be updated when an argument slot is
/// rewritten from `!poly.tvar` to a concrete type.
static void updateFuncSignature(FuncDefOp func, const TypeVarReplacementConverter &converter) {
  FunctionType oldFuncTy = func.getFunctionType();
  Type converted = converter.convertType(oldFuncTy);
  auto newFuncTy = llvm::cast<FunctionType>(converted);
  if (oldFuncTy == newFuncTy) {
    return;
  }

  func.setType(newFuncTy);
  if (func.getFunctionBody().empty()) {
    return;
  }

  Block &entryBlock = func.getFunctionBody().front();
  assert(entryBlock.getNumArguments() == newFuncTy.getNumInputs());
  for (auto [arg, newTy] : llvm::zip_equal(entryBlock.getArguments(), newFuncTy.getInputs())) {
    arg.setType(newTy);
  }
}

/// Convert every type-bearing surface on `op` that can mention an inferred type
/// variable.
///
/// This handles operation results, region block arguments, function signatures,
/// and attributes containing types. The pass mutates the IR directly because the
/// transformation preserves operation semantics and only changes already-proven
/// type annotations.
static bool convertOperationTypes(Operation *op, const TypeVarReplacementConverter &converter) {
  bool changed = false;

  if (auto func = llvm::dyn_cast<FuncDefOp>(op)) {
    FunctionType oldFuncTy = func.getFunctionType();
    updateFuncSignature(func, converter);
    changed |= oldFuncTy != func.getFunctionType();
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (BlockArgument arg : block.getArguments()) {
        Type newTy = converter.convertType(arg.getType());
        if (newTy != arg.getType()) {
          arg.setType(newTy);
          changed = true;
        }
      }
    }
  }

  for (Value result : op->getResults()) {
    Type newTy = converter.convertType(result.getType());
    if (newTy != result.getType()) {
      result.setType(newTy);
      changed = true;
    }
  }

  SmallVector<NamedAttribute> newAttrs;
  bool attrsChanged = false;
  newAttrs.reserve(op->getAttrs().size());
  for (NamedAttribute attr : op->getAttrs()) {
    Attribute newAttr = converter.convertAttr(attr.getValue());
    newAttrs.emplace_back(attr.getName(), newAttr);
    attrsChanged |= newAttr != attr.getValue();
  }
  if (attrsChanged) {
    op->setAttrs(DictionaryAttr::get(op->getContext(), newAttrs));
    changed = true;
  }

  return changed;
}

/// Remove `poly.unifiable_cast` operations that became identity casts.
///
/// After type replacement, a cast such as
/// `!poly.tvar<@T> -> !array.type<4 x index>` becomes
/// `!array.type<4 x index> -> !array.type<4 x index>`. Such casts no longer
/// carry information and can be replaced with their input value.
static void removeIdentityCasts(TemplateOp templateOp) {
  SmallVector<UnifiableCastOp> erase;
  templateOp.walk([&erase](UnifiableCastOp castOp) {
    if (castOp.getInput().getType() == castOp.getResult().getType()) {
      erase.push_back(castOp);
    }
  });
  for (UnifiableCastOp castOp : erase) {
    castOp.getResult().replaceAllUsesWith(castOp.getInput());
    castOp.erase();
  }
}

/// Erase `poly.param` definitions whose type variables were fully resolved.
///
/// The erase happens after body and call-site rewrites so symbol uses in types
/// and template parameter lists have already been removed.
static void removeResolvedParams(TemplateInferenceInfo &info) {
  for (auto paramOp : llvm::make_early_inc_range(info.templateOp.getConstOps<TemplateParamOp>())) {
    auto name = paramOp.getSymNameAttr();
    if (info.replacements.contains(name)) {
      paramOp.erase();
    }
  }
}

/// Update calls to functions inside rewritten templates.
///
/// Call operations keep their explicit template argument list, if any. When a
/// callee template loses resolved `poly.param`s, call-site argument lists must
/// drop the same positions. Result types are also converted because call results
/// may mention an inferred return type variable.
static bool updateCallTemplateParams(
    ModuleOp module, DenseMap<Operation *, const TypeVarReplacementConverter *> &converters
) {
  bool modified = false;
  SymbolTableCollection tables;
  module.walk([&](CallOp callOp) {
    FailureOr<SymbolLookupResult<FuncDefOp>> target = callOp.getCalleeTarget(tables);
    if (failed(target)) {
      return;
    }
    TemplateOp parentTemplate = getParentOfType<TemplateOp>(target->get().getOperation());
    if (!parentTemplate) {
      return;
    }
    const TypeVarReplacementConverter *converter = converters.lookup(parentTemplate.getOperation());
    if (!converter) {
      return;
    }

    ArrayAttr oldParams = callOp.getTemplateParamsAttr();
    ArrayAttr newParams = converter->convertTemplateParams(oldParams);
    if (oldParams != newParams) {
      callOp.setTemplateParamsAttr(newParams);
      modified = true;
    }

    for (Value result : callOp.getResults()) {
      Type newTy = converter->convertType(result.getType());
      if (newTy != result.getType()) {
        result.setType(newTy);
        modified = true;
      }
    }
  });
  return modified;
}

/// Build all analysis state needed to rewrite one template.
///
/// The function records the original template parameter order before any
/// rewrites, identifies eligible type-variable parameters, and collects concrete
/// replacements from the template body.
static FailureOr<TemplateInferenceInfo> buildInfo(TemplateOp templateOp) {
  TemplateInferenceInfo info;
  info.templateOp = templateOp;
  FailureOr<SymbolRefAttr> templatePath =
      getPathFromRoot(llvm::cast<SymbolOpInterface>(templateOp.getOperation()));
  if (failed(templatePath)) {
    return failure();
  }
  info.templatePath = getStringPieces(*templatePath);
  for (TemplateParamOp paramOp : templateOp.getConstOps<TemplateParamOp>()) {
    auto name = paramOp.getSymNameAttr();
    info.oldParamOrder.push_back(name);
    if (isTypeVarParam(paramOp)) {
      info.typeVarParams.try_emplace(name, paramOp);
    }
  }
  if (failed(TypeVarInferenceCollector(info).collect())) {
    return failure();
  }
  return info;
}

/// Pass driver for `-llzk-infer-tvar`.
///
/// The driver intentionally separates collection from mutation. It first walks
/// all templates and records every rewrite that should happen, then applies
/// conversions, updates callers, and finally erases resolved parameters. This
/// ordering avoids invalidating symbol/parameter information while it is still
/// needed to rewrite call sites.
class PassImpl : public llzk::polymorphic::impl::TypeVarInferencePassBase<PassImpl> {
public:
  using Base = TypeVarInferencePassBase<PassImpl>;
  using Base::Base;

private:
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Collect all template-local inferences before mutating IR. Conflicts are
    // reported during collection and abort the pass.
    SmallVector<TemplateInferenceInfo> rewrites;
    WalkResult collectResult = module.walk([&rewrites](TemplateOp templateOp) {
      FailureOr<TemplateInferenceInfo> info = buildInfo(templateOp);
      if (failed(info)) {
        return WalkResult::interrupt();
      }
      if (!info->replacements.empty()) {
        rewrites.push_back(std::move(*info));
      }
      return WalkResult::advance();
    });
    if (collectResult.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    // Rewrite each affected template in place, but keep the converters alive so
    // call-site rewriting can use the same positional template-parameter map.
    SmallVector<std::unique_ptr<TypeVarReplacementConverter>> converterStorage;
    DenseMap<Operation *, const TypeVarReplacementConverter *> convertersByTemplate;
    for (TemplateInferenceInfo &info : rewrites) {
      DenseMap<StringAttr, Type> replacements;
      for (const auto &entry : info.replacements) {
        replacements.try_emplace(entry.first, entry.second.type);
      }
      auto converter = std::make_unique<TypeVarReplacementConverter>(
          module.getContext(), info.templatePath, info.oldParamOrder, replacements
      );
      convertersByTemplate.try_emplace(info.templateOp.getOperation(), converter.get());

      info.templateOp.walk([&converter](Operation *op) { convertOperationTypes(op, *converter); });
      removeIdentityCasts(info.templateOp);
      converterStorage.push_back(std::move(converter));
    }

    // Calls are outside the rewritten template bodies, so update them after all
    // target templates have their converters registered.
    updateCallTemplateParams(module, convertersByTemplate);

    // Erase resolved parameters last; before this point, their original order is
    // still useful for template argument list conversion.
    for (TemplateInferenceInfo &info : rewrites) {
      removeResolvedParams(info);
    }
  }
};

} // namespace
