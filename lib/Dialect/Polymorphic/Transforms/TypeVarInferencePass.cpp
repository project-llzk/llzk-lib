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

/// Return the name of a flat symbol attribute, or null for non-symbol/nested refs.
static StringAttr getFlatSymbolName(Attribute attr) {
  auto symRef = llvm::dyn_cast_if_present<SymbolRefAttr>(attr);
  if (!symRef || !symRef.getNestedReferences().empty()) {
    return {};
  }
  return symRef.getRootReference();
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
  /// Whether struct/call template argument lists should remove resolved parameter positions.
  bool trimResolvedParams_;

public:
  /// Create a converter for a single template rewrite.
  ///
  /// `oldParamOrder` is captured before erasing any `poly.param` operations, so
  /// call-site template argument lists can be trimmed using their original
  /// positional layout.
  TypeVarReplacementConverter(
      MLIRContext *ctx, ArrayRef<StringAttr> templatePath, ArrayRef<StringAttr> oldParamOrder,
      const DenseMap<StringAttr, Type> &replacements, bool trimResolvedParams = true
  )
      : ctx_(ctx), templatePath_(templatePath), oldParamOrder_(oldParamOrder),
        replacements_(replacements), trimResolvedParams_(trimResolvedParams) {
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
        Attribute newNested = convertTemplateArgAttr(nested);
        newAttrs.push_back(newNested);
        changed |= newNested != nested;
      }
      return changed ? ArrayAttr::get(ctx_, newAttrs) : attr;
    }
    return attr;
  }

  /// Convert and trim a call-site or type-site template parameter list.
  ///
  /// If the list still has the original arity, entries corresponding to resolved `poly.param`s are
  /// removed. Remaining entries are recursively converted in case they mention a now-concrete type
  /// variable. A fully removed list is represented as `nullptr` so the printer elides the template
  /// argument syntax. A removed argument must agree with the inferred replacement before it is
  /// dropped.
  FailureOr<ArrayAttr> convertTemplateParams(ArrayAttr params, Operation *diagnosticOp) const {
    if (!params) {
      return ArrayAttr();
    }
    if (params.size() != oldParamOrder_.size()) {
      return params;
    }
    SmallVector<Attribute> kept;
    for (auto [paramName, attr] : llvm::zip_equal(oldParamOrder_, params.getValue())) {
      if (removedParams_.contains(paramName)) {
        if (failed(checkRemovedTemplateParam(paramName, attr, diagnosticOp))) {
          return failure();
        }
        continue;
      }
      kept.push_back(convertTemplateArgAttr(attr));
    }
    return kept.empty() ? ArrayAttr() : ArrayAttr::get(ctx_, kept);
  }

private:
  /// Convert a template argument attribute.
  ///
  /// Direct `SymbolRefAttr` template arguments can name self-typed type-variable
  /// parameters (`@T` for `poly.param @T : !poly.tvar<@T>`). Once such a
  /// parameter has a concrete replacement, the symbol argument becomes the
  /// corresponding `TypeAttr`.
  Attribute convertTemplateArgAttr(Attribute attr) const {
    if (StringAttr symbolName = getFlatSymbolName(attr)) {
      auto it = replacements_.find(symbolName);
      if (it != replacements_.end()) {
        return TypeAttr::get(it->second);
      }
    }
    return convertAttr(attr);
  }

  /// Check that an explicit argument for a removed parameter matches its replacement.
  LogicalResult
  checkRemovedTemplateParam(StringAttr paramName, Attribute attr, Operation *diagnosticOp) const {
    auto replacementIt = replacements_.find(paramName);
    assert(replacementIt != replacements_.end() && "removed parameter must have a replacement");
    Attribute convertedAttr = convertTemplateArgAttr(attr);
    Attribute expectedAttr = TypeAttr::get(replacementIt->second);
    if (convertedAttr == expectedAttr) {
      return success();
    }

    InFlightDiagnostic diag = diagnosticOp->emitError()
                              << "explicit template argument for inferred parameter @"
                              << paramName.getValue() << " must match inferred type "
                              << replacementIt->second << ", but found ";
    if (auto typeAttr = llvm::dyn_cast<TypeAttr>(attr)) {
      diag << typeAttr.getValue();
    } else {
      diag << attr;
    }
    return diag;
  }

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
      if (trimResolvedParams_ && removeOwnedParams &&
          removedParams_.contains(oldParamOrder_[index])) {
        changed = true;
        continue;
      }
      Attribute newAttr = convertTemplateArgAttr(attr);
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
  /// Function-local concrete replacements before the template-wide safety merge.
  DenseMap<Operation *, DenseMap<StringAttr, InferredType>> functionReplacements;
};

/// Return whether a flat symbol reference names `paramName`.
static bool symbolMatchesParam(SymbolRefAttr symRef, StringAttr paramName) {
  return symRef && symRef.getNestedReferences().empty() && symRef.getRootReference() == paramName;
}

/// Return whether a type mentions an eligible type-variable parameter.
static bool typeMentionsParam(Type ty, StringAttr paramName);

/// Return whether an attribute mentions an eligible type-variable parameter.
///
/// Symbol references only count in template-argument positions, such as struct
/// type parameter arrays or call-site template parameters. Top-level operation
/// symbol attributes like callee or member names are not type-variable uses.
static bool attrMentionsParam(Attribute attr, StringAttr paramName, bool allowSymbolRefs) {
  if (!attr) {
    return false;
  }
  if (auto typeAttr = llvm::dyn_cast<TypeAttr>(attr)) {
    return typeMentionsParam(typeAttr.getValue(), paramName);
  }
  if (auto arrayAttr = llvm::dyn_cast<ArrayAttr>(attr)) {
    return llvm::any_of(arrayAttr, [paramName](Attribute nested) {
      return attrMentionsParam(nested, paramName, /*allowSymbolRefs=*/true);
    });
  }
  return allowSymbolRefs && symbolMatchesParam(llvm::dyn_cast<SymbolRefAttr>(attr), paramName);
}

/// Return whether a type mentions an eligible type-variable parameter.
static bool typeMentionsParam(Type ty, StringAttr paramName) {
  if (!ty) {
    return false;
  }
  if (auto tvarTy = llvm::dyn_cast<TypeVarType>(ty)) {
    return tvarTy.getNameRef().getAttr() == paramName;
  }
  if (auto arrayTy = llvm::dyn_cast<ArrayType>(ty)) {
    return typeMentionsParam(arrayTy.getElementType(), paramName) ||
           llvm::any_of(arrayTy.getDimensionSizes(), [paramName](Attribute dim) {
      return attrMentionsParam(dim, paramName, /*allowSymbolRefs=*/true);
    });
  }
  if (auto structTy = llvm::dyn_cast<StructType>(ty)) {
    ArrayAttr params = structTy.getParams();
    return params && attrMentionsParam(params, paramName, /*allowSymbolRefs=*/true);
  }
  if (auto podTy = llvm::dyn_cast<PodType>(ty)) {
    return llvm::any_of(podTy.getRecords(), [paramName](RecordAttr record) {
      return typeMentionsParam(record.getType(), paramName);
    });
  }
  if (auto funcTy = llvm::dyn_cast<FunctionType>(ty)) {
    return llvm::any_of(funcTy.getInputs(), [paramName](Type input) {
      return typeMentionsParam(input, paramName);
    }) || llvm::any_of(funcTy.getResults(), [paramName](Type result) {
      return typeMentionsParam(result, paramName);
    });
  }
  return false;
}

/// Return whether `op` mentions an eligible type-variable parameter in a type-bearing surface.
static bool operationMentionsParam(Operation *op, StringAttr paramName) {
  if (llvm::any_of(op->getResultTypes(), [paramName](Type ty) {
    return typeMentionsParam(ty, paramName);
  })) {
    return true;
  }
  for (Region &region : op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      if (llvm::any_of(block.getArgumentTypes(), [paramName](Type ty) {
        return typeMentionsParam(ty, paramName);
      })) {
        return true;
      }
    }
  }
  return llvm::any_of(op->getAttrs(), [paramName](NamedAttribute attr) {
    return attrMentionsParam(attr.getValue(), paramName, /*allowSymbolRefs=*/false);
  });
}

/// Return whether a function body/signature mentions an eligible type-variable parameter.
static bool funcMentionsParam(FuncDefOp func, StringAttr paramName) {
  if (operationMentionsParam(func.getOperation(), paramName)) {
    return true;
  }
  WalkResult result = func.walk([paramName](Operation *op) {
    return operationMentionsParam(op, paramName) ? WalkResult::interrupt() : WalkResult::advance();
  });
  return result.wasInterrupted();
}

/// Walk a template body and collect all type-variable inferences from
/// `poly.unifiable_cast` operations.
///
/// `poly.unifiable_cast` is the proof source for this pass: when one side of the
/// cast is a template type variable and the other side is concrete, the cast
/// establishes the concrete replacement that later makes the cast redundant.
class TypeVarInferenceCollector {
  /// Per-template inference output and eligible parameter metadata.
  TemplateInferenceInfo &inferenceInfo;
  /// Concrete replacements proven within the current inference scope.
  DenseMap<StringAttr, InferredType> &replacements;
  /// Concrete inferences for SSA values observed during collection.
  DenseMap<Value, InferredType> byValue;
  /// Whether the current collection iteration learned a new concrete fact.
  bool changedInIteration = false;

public:
  /// Create a collector for one function's inference state.
  TypeVarInferenceCollector(
      TemplateInferenceInfo &info, DenseMap<StringAttr, InferredType> &scopeReplacements
  )
      : inferenceInfo(info), replacements(scopeReplacements) {}

  /// Visit all casts until no new concrete parameter or SSA value facts appear.
  LogicalResult collect(FuncDefOp func) {
    do {
      changedInIteration = false;
      WalkResult result = func.walk([this](UnifiableCastOp castOp) {
        return WalkResult(collectTypePairInferences(
            castOp.getInput().getType(), castOp.getResult().getType(), castOp.getInput(),
            castOp.getResult(), castOp.getLoc()
        ));
      });
      if (result.wasInterrupted()) {
        return failure();
      }
    } while (changedInIteration);
    return success();
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

    auto byParamIt = replacements.find(paramName);
    if (byParamIt == replacements.end()) {
      replacements.try_emplace(paramName, InferredType {inferredTy, loc});
      changedInIteration = true;
    } else if (byParamIt->second.type != inferredTy) {
      return reportConflict("template parameter", byParamIt->second.loc, byParamIt->second.type);
    }

    if (value) {
      auto byValueIt = byValue.find(value);
      if (byValueIt == byValue.end()) {
        byValue.try_emplace(value, InferredType {inferredTy, loc});
        changedInIteration = true;
      } else if (byValueIt->second.type != inferredTy) {
        return reportConflict("SSA value using", byValueIt->second.loc, byValueIt->second.type);
      }
    }
    return success();
  }

  /// Collect inferences from a pair of types known to unify.
  ///
  /// The direct case is `!poly.tvar<@T>` on one side and a concrete type on the other. The function
  /// also descends into aggregate type contents so an aggregate shape can force a nested type
  /// variable. The collector runs to a fixed point, so if one side is still a type variable but the
  /// opposite SSA value has a concrete inference from any cast, that concrete value-level proof is
  /// propagated through chained casts regardless of operation order.
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
      if (auto rhsStruct = llvm::dyn_cast<StructType>(rhs)) {
        ArrayRef<Attribute> lhsParams =
            lhsStruct.getParams() ? lhsStruct.getParams().getValue() : ArrayRef<Attribute> {};
        ArrayRef<Attribute> rhsParams =
            rhsStruct.getParams() ? rhsStruct.getParams().getValue() : ArrayRef<Attribute> {};
        if (lhsParams.size() != rhsParams.size()) {
          return success();
        }
        for (auto [lhsAttr, rhsAttr] : llvm::zip_equal(lhsParams, rhsParams)) {
          if (failed(collectTemplateArgInferences(lhsAttr, rhsAttr, loc))) {
            return failure();
          }
        }
      }
    }

    if (auto lhsPod = llvm::dyn_cast<PodType>(lhs)) {
      if (auto rhsPod = llvm::dyn_cast<PodType>(rhs)) {
        ArrayRef<RecordAttr> lhsRecords = lhsPod.getRecords();
        ArrayRef<RecordAttr> rhsRecords = rhsPod.getRecords();
        if (lhsRecords.size() != rhsRecords.size()) {
          return success();
        }
        for (auto [lhsRecord, rhsRecord] : llvm::zip_equal(lhsRecords, rhsRecords)) {
          if (lhsRecord.getName() != rhsRecord.getName()) {
            return success();
          }
          if (failed(collectTypePairInferences(
                  lhsRecord.getType(), rhsRecord.getType(), Value(), Value(), loc
              ))) {
            return failure();
          }
        }
      }
    }

    return success();
  }

  /// Collect inferences from a pair of struct template argument attributes.
  ///
  /// Type arguments encoded as `TypeAttr` are recursively unified by type. A
  /// flat `SymbolRefAttr` is treated as an inference target only when it names
  /// an eligible self-typed type-variable parameter; `recordInference` filters
  /// out ordinary value parameters and non-concrete candidates.
  LogicalResult collectTemplateArgInferences(Attribute lhsAttr, Attribute rhsAttr, Location loc) {
    auto lhsTyAttr = llvm::dyn_cast<TypeAttr>(lhsAttr);
    auto rhsTyAttr = llvm::dyn_cast<TypeAttr>(rhsAttr);
    if (lhsTyAttr && rhsTyAttr) {
      return collectTypePairInferences(
          lhsTyAttr.getValue(), rhsTyAttr.getValue(), Value(), Value(), loc
      );
    }

    if (StringAttr lhsSymbolName = getFlatSymbolName(lhsAttr)) {
      if (rhsTyAttr && failed(recordInference(lhsSymbolName, rhsTyAttr.getValue(), Value(), loc))) {
        return failure();
      }
    }
    if (StringAttr rhsSymbolName = getFlatSymbolName(rhsAttr)) {
      if (lhsTyAttr && failed(recordInference(rhsSymbolName, lhsTyAttr.getValue(), Value(), loc))) {
        return failure();
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
static void removeIdentityCasts(Operation *root) {
  SmallVector<UnifiableCastOp> erase;
  root->walk([&erase](UnifiableCastOp castOp) {
    if (castOp.getInput().getType() == castOp.getResult().getType()) {
      erase.push_back(castOp);
    }
  });
  for (UnifiableCastOp castOp : erase) {
    castOp.getResult().replaceAllUsesWith(castOp.getInput());
    castOp.erase();
  }
}

/// Build concrete replacement types from function-local inferences that still
/// need call-site cloning after any template-wide replacements have been applied.
static DenseMap<StringAttr, Type>
getResidualFunctionReplacements(const TemplateInferenceInfo &info, FuncDefOp func) {
  DenseMap<StringAttr, Type> replacements;
  auto funcIt = info.functionReplacements.find(func.getOperation());
  if (funcIt == info.functionReplacements.end()) {
    return replacements;
  }
  for (const auto &entry : funcIt->second) {
    if (!info.replacements.contains(entry.first)) {
      replacements.try_emplace(entry.first, entry.second.type);
    }
  }
  return replacements;
}

/// Return the original template parameter index for `paramName`, if present.
static std::optional<unsigned>
getParamIndex(ArrayRef<StringAttr> paramOrder, StringAttr paramName) {
  for (auto indexedParam : llvm::enumerate(paramOrder)) {
    if (indexedParam.value() == paramName) {
      return indexedParam.index();
    }
  }
  return std::nullopt;
}

/// Return true if explicit call-site template parameters match the inferred
/// concrete function-local replacements.
static bool callParamsMatchReplacements(
    ArrayAttr callParams, ArrayRef<StringAttr> oldParamOrder,
    const DenseMap<StringAttr, Type> &replacements
) {
  if (!callParams || callParams.size() != oldParamOrder.size()) {
    return false;
  }
  for (const auto &entry : replacements) {
    std::optional<unsigned> index = getParamIndex(oldParamOrder, entry.first);
    if (!index) {
      return false;
    }
    if (callParams[*index] != TypeAttr::get(entry.second)) {
      return false;
    }
  }
  return true;
}

/// Build concrete type-variable replacements from an explicit call instantiation.
///
/// A module-level function clone has no enclosing `poly.template`, so every
/// type-variable parameter mentioned by the cloned function must be replaced by
/// a concrete type from the call's template argument list.
static DenseMap<StringAttr, Type> getConcreteCallSiteReplacements(
    ArrayAttr callParams, const TemplateInferenceInfo &info, FuncDefOp func
) {
  DenseMap<StringAttr, Type> replacements;
  if (!callParams || callParams.size() != info.oldParamOrder.size()) {
    return replacements;
  }

  for (const auto &entry : info.typeVarParams) {
    StringAttr paramName = entry.first;
    if (!funcMentionsParam(func, paramName)) {
      continue;
    }
    std::optional<unsigned> index = getParamIndex(info.oldParamOrder, paramName);
    assert(index && "eligible type-variable parameter must appear in parameter order");
    Attribute attr = callParams[*index];
    auto tyAttr = llvm::dyn_cast<TypeAttr>(attr);
    if (!tyAttr || !isConcreteType(tyAttr.getValue())) {
      return DenseMap<StringAttr, Type>();
    }
    replacements.try_emplace(paramName, tyAttr.getValue());
  }
  return replacements;
}

/// Create or reuse a module-level clone for a concrete function-local tvar instantiation.
static FailureOr<FuncDefOp> getOrCreateSpecializedFunctionClone(
    ModuleOp module, TemplateOp templateOp, FuncDefOp func, ArrayAttr callParams,
    const TemplateInferenceInfo &info, const DenseMap<StringAttr, Type> &replacements,
    SymbolTableCollection &tables
) {
  std::string cloneName = buildInstantiatedFunctionName(
      templateOp.getSymName(), func.getSymName(), callParams.getValue()
  );
  SymbolTable &moduleSymbols = tables.getSymbolTable(module);
  if (Operation *existing = moduleSymbols.lookup(cloneName)) {
    if (auto existingFunc = llvm::dyn_cast<FuncDefOp>(existing)) {
      return existingFunc;
    }
    return failure();
  }

  FuncDefOp clone = func.clone();
  clone.setSymName(cloneName);
  TypeVarReplacementConverter converter(
      module.getContext(), info.templatePath, info.oldParamOrder, replacements,
      /*trimResolvedParams=*/false
  );
  clone.walk([&converter](Operation *op) { convertOperationTypes(op, converter); });
  removeIdentityCasts(clone.getOperation());
  moduleSymbols.insert(clone, Block::iterator(templateOp));
  return clone;
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
static FailureOr<bool> updateCallTemplateParams(
    ModuleOp module, DenseMap<Operation *, const TypeVarReplacementConverter *> &converters
) {
  bool modified = false;
  bool failedConversion = false;
  SymbolTableCollection tables;
  module.walk([&](CallOp callOp) {
    if (failedConversion) {
      return;
    }
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
    FailureOr<ArrayAttr> newParams = converter->convertTemplateParams(oldParams, callOp);
    if (failed(newParams)) {
      failedConversion = true;
      return;
    }
    if (oldParams != *newParams) {
      callOp.setTemplateParamsAttr(*newParams);
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
  if (failedConversion) {
    return failure();
  }
  return modified;
}

/// Specialize calls to template functions with concrete tvar instantiations.
///
/// This intentionally handles only fully-concrete explicit call-site type
/// arguments for free functions directly nested in a template. More general
/// partial instantiations should follow the flattening pass' template-cloning
/// machinery, but this covers the tvar-only cases without invoking flattening's
/// struct/array transformations. Function-local inference facts are used to
/// reject call-site instantiations that disagree with a proof in the body.
static LogicalResult specializeFunctionLocalCalls(
    ModuleOp module, DenseMap<Operation *, const TemplateInferenceInfo *> &infoByTemplate
) {
  bool failedClone = false;
  SymbolTableCollection tables;
  module.walk([&](CallOp callOp) {
    if (failedClone) {
      return;
    }
    FailureOr<SymbolLookupResult<FuncDefOp>> target = callOp.getCalleeTarget(tables);
    if (failed(target)) {
      return;
    }
    FuncDefOp targetFunc = target->get();
    auto parentTemplate = llvm::dyn_cast_or_null<TemplateOp>(targetFunc->getParentOp());
    if (!parentTemplate) {
      return;
    }
    const TemplateInferenceInfo *info = infoByTemplate.lookup(parentTemplate.getOperation());
    if (!info) {
      return;
    }

    ArrayAttr callParams = callOp.getTemplateParamsAttr();
    DenseMap<StringAttr, Type> replacements =
        getConcreteCallSiteReplacements(callParams, *info, targetFunc);
    if (replacements.empty()) {
      return;
    }
    DenseMap<StringAttr, Type> inferredReplacements =
        getResidualFunctionReplacements(*info, targetFunc);
    if (!callParamsMatchReplacements(callParams, info->oldParamOrder, inferredReplacements)) {
      return;
    }

    FailureOr<FuncDefOp> clone = getOrCreateSpecializedFunctionClone(
        module, parentTemplate, targetFunc, callParams, *info, replacements, tables
    );
    if (failed(clone)) {
      failedClone = true;
      return;
    }

    callOp.setCalleeAttr(
        getInstantiatedFunctionCallee(callOp.getCalleeAttr(), clone->getSymNameAttr())
    );
    callOp.setTemplateParamsAttr(nullptr);
  });
  return failure(failedClone);
}

/// Build all analysis state needed to rewrite one template.
///
/// The function records the original template parameter order before any
/// rewrites, identifies eligible type-variable parameters, and collects concrete
/// replacements only when every function that mentions a parameter proves the
/// same replacement. This preserves per-call instantiation for sibling entry
/// points that remain unconstrained.
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

  SmallVector<FuncDefOp> funcs;
  templateOp.walk([&funcs](FuncDefOp func) { funcs.push_back(func); });

  DenseMap<StringAttr, unsigned> mentionCounts;
  DenseMap<StringAttr, unsigned> proofCounts;
  DenseMap<StringAttr, InferredType> commonReplacements;
  DenseSet<StringAttr> incompatibleReplacements;
  for (FuncDefOp func : funcs) {
    for (const auto &entry : info.typeVarParams) {
      if (funcMentionsParam(func, entry.first)) {
        ++mentionCounts[entry.first];
      }
    }

    DenseMap<StringAttr, InferredType> funcReplacements;
    if (failed(TypeVarInferenceCollector(info, funcReplacements).collect(func))) {
      return failure();
    }
    if (!funcReplacements.empty()) {
      info.functionReplacements.try_emplace(func.getOperation(), funcReplacements);
    }
    for (const auto &entry : funcReplacements) {
      ++proofCounts[entry.first];
      auto commonIt = commonReplacements.find(entry.first);
      if (commonIt == commonReplacements.end()) {
        commonReplacements.try_emplace(entry.first, entry.second);
      } else if (commonIt->second.type != entry.second.type) {
        incompatibleReplacements.insert(entry.first);
      }
    }
  }

  for (const auto &entry : commonReplacements) {
    StringAttr paramName = entry.first;
    if (!incompatibleReplacements.contains(paramName) &&
        proofCounts.lookup(paramName) == mentionCounts.lookup(paramName)) {
      info.replacements.try_emplace(paramName, entry.second);
    }
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
      if (!info->replacements.empty() || !info->functionReplacements.empty()) {
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
    DenseMap<Operation *, const TemplateInferenceInfo *> infoByTemplate;
    for (TemplateInferenceInfo &info : rewrites) {
      infoByTemplate.try_emplace(info.templateOp.getOperation(), &info);
      DenseMap<StringAttr, Type> replacements;
      for (const auto &entry : info.replacements) {
        replacements.try_emplace(entry.first, entry.second.type);
      }
      auto converter = std::make_unique<TypeVarReplacementConverter>(
          module.getContext(), info.templatePath, info.oldParamOrder, replacements
      );
      convertersByTemplate.try_emplace(info.templateOp.getOperation(), converter.get());

      info.templateOp.walk([&converter](Operation *op) { convertOperationTypes(op, *converter); });
      removeIdentityCasts(info.templateOp.getOperation());
      converterStorage.push_back(std::move(converter));
    }

    // Calls are outside the rewritten template bodies, so update them after all
    // target templates have their converters registered.
    if (failed(updateCallTemplateParams(module, convertersByTemplate))) {
      signalPassFailure();
      return;
    }
    if (failed(specializeFunctionLocalCalls(module, infoByTemplate))) {
      signalPassFailure();
      return;
    }

    // Erase resolved parameters last; before this point, their original order is
    // still useful for template argument list conversion.
    for (TemplateInferenceInfo &info : rewrites) {
      removeResolvedParams(info);
    }
  }
};

} // namespace
