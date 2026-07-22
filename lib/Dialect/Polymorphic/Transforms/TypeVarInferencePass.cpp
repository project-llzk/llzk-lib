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
///   1. Collects concrete type inferences from function and `poly.expr` bodies
///      for `poly.param` definitions of the form `poly.param @T : !poly.tvar<@T>`.
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
#include "llzk/Dialect/Array/Util/ArrayTypeHelper.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/Verif/IR/Ops.h"
#include "llzk/Util/Constants.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolLookup.h"
#include "llzk/Util/SymbolTableLLZK.h"
#include "llzk/Util/TypeHelper.h"
#include "llzk/Util/Walk.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/raw_ostream.h>

#include <memory>
#include <vector>

// Include the generated base pass class definitions.
namespace llzk::polymorphic {
#define GEN_PASS_DEF_TYPEVARINFERENCEPASS
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h.inc"
} // namespace llzk::polymorphic

#include "SharedImpl.h"

#define DEBUG_TYPE "llzk-infer-tvar"

using namespace mlir;
using namespace llzk;
using namespace llzk::array;
using namespace llzk::component;
using namespace llzk::function;
using namespace llzk::pod;
using namespace llzk::polymorphic;
using namespace llzk::polymorphic::detail;

namespace {

/// A concrete type inferred at a source location.
///
/// The location is retained so a later conflicting inference can point back to
/// the original proof site with a note.
struct InferredType {
  Type type;
  Location loc;
};

/// Template-local clone callees created by this pass, keyed by the exact
/// replacement list used to create each clone.
///
/// The cache deliberately records only pass-created clones. A user-defined symbol
/// may already occupy the preferred name, and different exact type lists may also
/// share the same preferred name. `SymbolTable::insert` will rename the clone in
/// those cases. Reusing the cached callee avoids confusing such symbols with
/// generated clones and keeps repeated calls to the same exact instantiation
/// pointed at the same uniquely-named clone.
using SpecializedCallableCloneCache = DenseMap<Operation *, llvm::StringMap<SymbolRefAttr>>;
using SpecializedTemplateCloneCache = DenseMap<Operation *, llvm::StringMap<StringAttr>>;

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

/// Convert every type in `oldTypes`, returning whether any entry changed.
template <typename ConverterT>
static bool
convertTypeRange(TypeRange oldTypes, SmallVectorImpl<Type> &newTypes, ConverterT &converter) {
  bool changed = false;
  newTypes.reserve(oldTypes.size());
  for (Type oldTy : oldTypes) {
    Type newTy = converter.convertType(oldTy);
    newTypes.push_back(newTy);
    changed |= newTy != oldTy;
  }
  return changed;
}

/// Convert a POD type by recursively converting its record types.
template <typename ConverterT>
static Type convertPodType(PodType podTy, MLIRContext *ctx, ConverterT &converter) {
  SmallVector<RecordAttr> newRecords;
  bool changed = false;
  for (RecordAttr record : podTy.getRecords()) {
    Type newRecordTy = converter.convertType(record.getType());
    newRecords.push_back(RecordAttr::get(ctx, record.getName(), newRecordTy));
    changed |= newRecordTy != record.getType();
  }
  return changed ? PodType::get(ctx, newRecords) : podTy;
}

/// Convert an array element type and flatten nested array replacements.
template <typename ConvertElementFn>
static Type convertArrayElementType(ArrayType arrTy, ConvertElementFn convertElement) {
  Type newElemTy = convertElement(arrTy.getElementType());
  if (newElemTy == arrTy.getElementType()) {
    return arrTy;
  }
  return llzk::polymorphic::detail::flattenInstantiatedArrayType(arrTy, newElemTy);
}

/// Convert a function type by recursively converting inputs and results.
template <typename ConverterT>
static Type convertFunctionType(FunctionType funcTy, ConverterT &converter) {
  SmallVector<Type> newInputs;
  SmallVector<Type> newResults;
  bool changed = convertTypeRange(funcTy.getInputs(), newInputs, converter);
  changed |= convertTypeRange(funcTy.getResults(), newResults, converter);
  return changed ? FunctionType::get(funcTy.getContext(), newInputs, newResults) : funcTy;
}

/// Convert `TypeAttr` and nested `ArrayAttr` values with caller-provided recursion behavior.
template <typename ConvertTypeFn, typename ConvertNestedAttrFn>
static Attribute convertTypeOrArrayAttr(
    Attribute attr, MLIRContext *ctx, ConvertTypeFn convertType,
    ConvertNestedAttrFn convertNestedAttr
) {
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
      Attribute newNested = convertNestedAttr(nested);
      newAttrs.push_back(newNested);
      changed |= newNested != nested;
    }
    return changed ? ArrayAttr::get(ctx, newAttrs) : attr;
  }
  return attr;
}

/// Return true when a template argument is compatible with an inferred concrete
/// type under LLZK type-parameter unification rules.
static bool templateArgUnifiesWithType(Attribute attr, Type expectedTy) {
  Attribute expectedAttr = TypeAttr::get(expectedTy);
  return typeParamsUnify(ArrayRef<Attribute> {attr}, ArrayRef<Attribute> {expectedAttr});
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
      MLIRContext *c, ArrayRef<StringAttr> templatePath, ArrayRef<StringAttr> oldParamOrder,
      const DenseMap<StringAttr, Type> &replacements, bool trimResolvedParams = true
  )
      : ctx_(c), templatePath_(templatePath), oldParamOrder_(oldParamOrder),
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
    DenseSet<StringAttr> resolvingParams;
    return convertType(ty, resolvingParams);
  }

private:
  Type convertType(Type ty, DenseSet<StringAttr> &resolvingParams) const {
    if (!ty) {
      return ty;
    }
    if (auto tvarTy = llvm::dyn_cast<TypeVarType>(ty)) {
      StringAttr paramName = tvarTy.getNameRef().getAttr();
      auto it = replacements_.find(paramName);
      if (it == replacements_.end()) {
        return ty;
      }
      if (!resolvingParams.insert(paramName).second) {
        return ty;
      }
      Type replacement = convertType(it->second, resolvingParams);
      resolvingParams.erase(paramName);
      return replacement;
    }
    if (auto arrTy = llvm::dyn_cast<ArrayType>(ty)) {
      return convertArrayElementType(arrTy, [this, &resolvingParams](Type elemTy) {
        return convertType(elemTy, resolvingParams);
      });
    }
    if (auto structTy = llvm::dyn_cast<StructType>(ty)) {
      return convertStructType(structTy, resolvingParams);
    }
    if (auto podTy = llvm::dyn_cast<PodType>(ty)) {
      return convertPodType(podTy, ctx_, *this);
    }
    if (auto funcTy = llvm::dyn_cast<FunctionType>(ty)) {
      return convertFunctionType(funcTy, *this);
    }
    return ty;
  }

public:
  /// Convert an attribute that may contain types needing replacement.
  ///
  /// This currently handles `TypeAttr` and nested `ArrayAttr`, which covers the
  /// template parameter lists and type-encoded op attributes used by the
  /// polymorphic/function dialects.
  Attribute convertAttr(Attribute attr) const {
    DenseSet<StringAttr> resolvingParams;
    return convertAttr(attr, resolvingParams);
  }

  /// Validate type-bearing surfaces before resolved owned-struct parameters are
  /// dropped.
  LogicalResult validateOperation(Operation *op) const {
    if (auto createOp = llvm::dyn_cast<CreateArrayOp>(op)) {
      if (failed(validateCreateArrayOp(createOp))) {
        return failure();
      }
    }
    if (auto func = llvm::dyn_cast<FuncDefOp>(op)) {
      if (failed(validateType(func.getFunctionType(), op))) {
        return failure();
      }
    }
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        for (Type argTy : block.getArgumentTypes()) {
          if (failed(validateType(argTy, op))) {
            return failure();
          }
        }
      }
    }
    for (Type resultTy : op->getResultTypes()) {
      if (failed(validateType(resultTy, op))) {
        return failure();
      }
    }
    for (NamedAttribute attr : op->getAttrs()) {
      if (failed(validateAttr(attr.getValue(), op))) {
        return failure();
      }
    }
    return success();
  }

private:
  /// Reject initialized nested array rewrites when the original initializer list cannot be mapped
  /// to concrete outer indices.
  ///
  /// When a `!poly.tvar` element is inferred as an array type, `convertOperationTypes` lowers
  /// `array.new %a, %b : <2 x !poly.tvar<@T>>` by creating the flattened result array and inserting
  /// each subarray operand at its original outer position. That lowering works even if the inferred
  /// subarray has dynamic dimensions, but the original array shape must still be static so each
  /// initializer operand has a known insertion index.
  LogicalResult validateCreateArrayOp(CreateArrayOp createOp) const {
    if (createOp.getElements().empty()) {
      return success();
    }

    ArrayType oldResultTy = createOp.getType();
    auto newResultTy = llvm::dyn_cast<ArrayType>(convertType(oldResultTy));
    auto newElemTy = llvm::dyn_cast<ArrayType>(convertType(oldResultTy.getElementType()));
    if (!newResultTy || !newElemTy || newResultTy == oldResultTy || oldResultTy.hasStaticShape()) {
      return success();
    }

    return createOp.emitError()
           << "cannot rewrite initialized array.new with non-static initializer shape "
           << oldResultTy;
  }

  LogicalResult validateType(Type ty, Operation *diagnosticOp) const {
    if (!ty) {
      return success();
    }
    if (auto arrTy = llvm::dyn_cast<ArrayType>(ty)) {
      return validateType(arrTy.getElementType(), diagnosticOp);
    }
    if (auto structTy = llvm::dyn_cast<StructType>(ty)) {
      return validateStructType(structTy, diagnosticOp);
    }
    if (auto podTy = llvm::dyn_cast<PodType>(ty)) {
      for (RecordAttr record : podTy.getRecords()) {
        if (failed(validateType(record.getType(), diagnosticOp))) {
          return failure();
        }
      }
      return success();
    }
    if (auto funcTy = llvm::dyn_cast<FunctionType>(ty)) {
      for (Type inputTy : funcTy.getInputs()) {
        if (failed(validateType(inputTy, diagnosticOp))) {
          return failure();
        }
      }
      for (Type resultTy : funcTy.getResults()) {
        if (failed(validateType(resultTy, diagnosticOp))) {
          return failure();
        }
      }
    }
    return success();
  }

  LogicalResult validateAttr(Attribute attr, Operation *diagnosticOp) const {
    if (!attr) {
      return success();
    }
    if (auto tyAttr = llvm::dyn_cast<TypeAttr>(attr)) {
      return validateType(tyAttr.getValue(), diagnosticOp);
    }
    if (auto arrAttr = llvm::dyn_cast<ArrayAttr>(attr)) {
      for (Attribute nested : arrAttr.getValue()) {
        if (failed(validateAttr(nested, diagnosticOp))) {
          return failure();
        }
      }
    }
    return success();
  }

  Attribute convertAttr(Attribute attr, DenseSet<StringAttr> &resolvingParams) const {
    return convertTypeOrArrayAttr(attr, ctx_, [this, &resolvingParams](Type ty) {
      return convertType(ty, resolvingParams);
    }, [this, &resolvingParams](Attribute nested) {
      return convertTemplateArgAttr(nested, resolvingParams);
    });
  }

public:
  /// Convert and trim a call-site or type-site template parameter list.
  ///
  /// If the list still has the original arity, entries corresponding to resolved `poly.param`s are
  /// removed. Remaining entries are recursively converted in case they mention a now-concrete type
  /// variable. A fully removed list is represented as `nullptr` so the printer elides the template
  /// argument syntax. A removed argument must agree with the inferred replacement before it is
  /// dropped.
  FailureOr<ArrayAttr> convertTemplateParams(
      ArrayAttr params, Operation *diagnosticOp, bool resolveTemplateSymbolArgs = true
  ) const {
    if (!params) {
      return ArrayAttr();
    }
    if (params.size() != oldParamOrder_.size()) {
      return params;
    }
    SmallVector<Attribute> kept;
    for (auto [paramName, attr] : llvm::zip_equal(oldParamOrder_, params.getValue())) {
      if (removedParams_.contains(paramName)) {
        if (failed(
                checkRemovedTemplateParam(paramName, attr, diagnosticOp, resolveTemplateSymbolArgs)
            )) {
          return failure();
        }
        continue;
      }
      kept.push_back(resolveTemplateSymbolArgs ? convertTemplateArgAttr(attr) : attr);
    }
    return kept.empty() ? ArrayAttr() : ArrayAttr::get(ctx_, kept);
  }

  /// Return true if a template argument list uses `?` for a parameter that this
  /// converter removes.
  bool hasWildcardForRemovedParam(ArrayAttr params) const {
    if (!params || params.size() != oldParamOrder_.size()) {
      return false;
    }
    for (auto [paramName, attr] : llvm::zip_equal(oldParamOrder_, params.getValue())) {
      if (!removedParams_.contains(paramName)) {
        continue;
      }
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr); intAttr && isDynamic(intAttr)) {
        return true;
      }
    }
    return false;
  }

private:
  Attribute convertTemplateArgAttr(Attribute attr) const {
    DenseSet<StringAttr> resolvingParams;
    return convertTemplateArgAttr(attr, resolvingParams);
  }

  /// Convert a template argument attribute.
  ///
  /// Direct `SymbolRefAttr` template arguments can name self-typed type-variable
  /// parameters (`@T` for `poly.param @T : !poly.tvar<@T>`). Once such a
  /// parameter has a concrete replacement, the symbol argument becomes the
  /// corresponding `TypeAttr`.
  Attribute convertTemplateArgAttr(Attribute attr, DenseSet<StringAttr> &resolvingParams) const {
    if (StringAttr symbolName = getFlatSymbolName(attr)) {
      auto it = replacements_.find(symbolName);
      if (it != replacements_.end()) {
        if (!resolvingParams.insert(symbolName).second) {
          return attr;
        }
        Type replacement = convertType(it->second, resolvingParams);
        resolvingParams.erase(symbolName);
        return TypeAttr::get(replacement);
      }
    }
    return convertAttr(attr, resolvingParams);
  }

  /// Check that an explicit argument for a removed parameter matches its replacement.
  LogicalResult checkRemovedTemplateParam(
      StringAttr paramName, Attribute attr, Operation *diagnosticOp, bool resolveTemplateSymbolArgs
  ) const {
    auto replacementIt = replacements_.find(paramName);
    assert(replacementIt != replacements_.end() && "removed parameter must have a replacement");
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr); intAttr && isDynamic(intAttr)) {
      return success();
    }
    Attribute convertedAttr = resolveTemplateSymbolArgs ? convertTemplateArgAttr(attr) : attr;
    if (getFlatSymbolName(convertedAttr)) {
      return emitRemovedTemplateParamMismatch(paramName, attr, replacementIt->second, diagnosticOp);
    }
    if (templateArgUnifiesWithType(convertedAttr, convertType(replacementIt->second))) {
      return success();
    }

    return emitRemovedTemplateParamMismatch(paramName, attr, replacementIt->second, diagnosticOp);
  }

  LogicalResult emitRemovedTemplateParamMismatch(
      StringAttr paramName, Attribute attr, Type replacementTy, Operation *diagnosticOp
  ) const {
    InFlightDiagnostic diag = diagnosticOp->emitError()
                              << "explicit template argument for inferred parameter @"
                              << paramName.getValue() << " must match inferred type "
                              << replacementTy << ", but found ";
    if (auto typeAttr = llvm::dyn_cast<TypeAttr>(attr)) {
      diag << typeAttr.getValue();
    } else {
      diag << attr;
    }
    return diag;
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
  StructType convertStructType(StructType structTy, DenseSet<StringAttr> &resolvingParams) const {
    ArrayAttr params = structTy.getParams();
    if (!params) {
      return structTy;
    }

    SmallVector<Attribute> newParams;
    bool changed = false;
    bool removeOwnedParams = isOwnedStructType(structTy) && params.size() == oldParamOrder_.size();
    for (auto [index, attr] : llvm::enumerate(params.getValue())) {
      if (trimResolvedParams_ && removeOwnedParams &&
          removedParams_.contains(oldParamOrder_[index])) {
        changed = true;
        continue;
      }
      Attribute newAttr = convertTemplateArgAttr(attr, resolvingParams);
      newParams.push_back(newAttr);
      changed |= newAttr != attr;
    }
    return changed ? getStructTypeWithParams(structTy.getNameRef(), ctx_, newParams) : structTy;
  }

  LogicalResult validateStructType(StructType structTy, Operation *diagnosticOp) const {
    ArrayAttr params = structTy.getParams();
    if (!params) {
      return success();
    }
    bool removeOwnedParams = isOwnedStructType(structTy) && params.size() == oldParamOrder_.size();
    for (auto indexedAttr : llvm::enumerate(params.getValue())) {
      unsigned index = indexedAttr.index();
      Attribute attr = indexedAttr.value();
      if (trimResolvedParams_ && removeOwnedParams &&
          removedParams_.contains(oldParamOrder_[index])) {
        if (failed(checkRemovedTemplateParam(
                oldParamOrder_[index], attr, diagnosticOp, /*resolveTemplateSymbolArgs=*/true
            ))) {
          return failure();
        }
        continue;
      }
      if (failed(validateAttr(attr, diagnosticOp))) {
        return failure();
      }
    }
    return success();
  }
};

/// Rewrite a callable type and keep the entry block arguments in sync.
template <typename ConverterT, typename SetSignatureFn>
static void updateCallableSignature(
    FunctionType oldFuncTy, Region &body, ConverterT &converter, SetSignatureFn setSignature
) {
  Type converted = converter.convertType(oldFuncTy);
  auto newFuncTy = llvm::cast<FunctionType>(converted);
  if (oldFuncTy == newFuncTy) {
    return;
  }

  setSignature(newFuncTy);
  if (body.empty()) {
    return;
  }

  Block &entryBlock = body.front();
  assert(entryBlock.getNumArguments() == newFuncTy.getNumInputs());
  for (auto [arg, newTy] : llvm::zip_equal(entryBlock.getArguments(), newFuncTy.getInputs())) {
    arg.setType(newTy);
  }
}

/// Rewrite a function type and keep the entry block arguments in sync.
///
/// `function.def` stores its function signature separately from the entry block
/// argument types, so both surfaces must be updated when an argument slot is
/// rewritten from `!poly.tvar` to a concrete type.
template <typename ConverterT>
static void updateFuncSignature(FuncDefOp func, ConverterT &converter) {
  updateCallableSignature(
      func.getFunctionType(), func.getFunctionBody(), converter,
      [func](FunctionType newFuncTy) mutable { func.setType(newFuncTy); }
  );
}

/// Rewrite a contract type and keep the entry block arguments in sync.
template <typename ConverterT>
static void updateContractSignature(verif::ContractOp contract, ConverterT &converter) {
  updateCallableSignature(
      contract.getFunctionType(), contract.getBody(), converter,
      [contract](FunctionType newFuncTy) mutable { contract.setFunctionType(newFuncTy); }
  );
}

/// Convert call targets when the active converter knows how to keep symbol
/// references in sync with converted types.
template <typename ConverterT> static bool convertCallCallee(CallOp callOp, ConverterT &converter) {
  if constexpr (requires { converter.convertCallCallee(callOp); }) {
    SymbolRefAttr oldCallee = callOp.getCalleeAttr();
    SymbolRefAttr newCallee = converter.convertCallCallee(callOp);
    if (newCallee != oldCallee) {
      callOp.setCalleeAttr(newCallee);
      return true;
    }
  }
  return false;
}

/// Return whether `converter` recorded a conversion failure, if it exposes that state.
template <typename ConverterT> static bool converterFailed(ConverterT &converter) {
  if constexpr (requires { converter.hadFailure(); }) {
    return converter.hadFailure();
  }
  return false;
}

/// Notify converters that track per-operation diagnostic state.
template <typename ConverterT>
static void startConverterOperation(ConverterT &converter, Operation *op) {
  if constexpr (requires { converter.startOperation(op); }) {
    converter.startOperation(op);
  }
}

/// Convert every type-bearing surface on `op` that can mention an inferred type variable.
///
/// This handles operation results, region block arguments, function signatures, and attributes
/// containing types. The pass mutates the IR directly because the transformation preserves
/// operation semantics and only changes already-proven type annotations.
template <typename ConverterT>
static FailureOr<bool> convertOperationTypes(Operation *op, ConverterT &converter) {
  if (auto createOp = llvm::dyn_cast<CreateArrayOp>(op)) {
    ArrayType oldResultTy = createOp.getType();
    auto newResultTy = llvm::dyn_cast<ArrayType>(converter.convertType(oldResultTy));
    auto newElemTy = llvm::dyn_cast<ArrayType>(converter.convertType(oldResultTy.getElementType()));
    if (converterFailed(converter)) {
      return failure();
    }
    if (newResultTy && newElemTy && newResultTy != oldResultTy && !createOp.getElements().empty() &&
        oldResultTy.hasStaticShape()) {
      // Validate every init value in the `array.new` before creating replacement ops to avoid
      // dangling IR if one of the init values is invalid.
      SmallVector<Type> newElementValueTypes;
      newElementValueTypes.reserve(createOp.getElements().size());
      for (auto [index, element] : llvm::enumerate(createOp.getElements())) {
        Type newElementValueTy = converter.convertType(element.getType());
        if (converterFailed(converter)) {
          return failure();
        }
        if (newElementValueTy != newElemTy) {
          createOp.emitError() << "cannot rewrite initialized array.new: initializer " << index
                               << " converts to " << newElementValueTy << ", but expected "
                               << newElemTy;
          return failure();
        }
        newElementValueTypes.push_back(newElementValueTy);
      }

      OpBuilder builder(createOp);
      Location loc = createOp.getLoc();
      CreateArrayOp newCreate = builder.create<CreateArrayOp>(loc, newResultTy);
      ArrayIndexGen idxGen = ArrayIndexGen::from(oldResultTy);
      for (auto [index, element] : llvm::enumerate(createOp.getElements())) {
        Type newElementValueTy = newElementValueTypes[index];
        if (element.getType() != newElementValueTy) {
          element.setType(newElementValueTy);
        }
        std::optional<SmallVector<Value>> indices =
            idxGen.delinearize(checkedCast<int64_t>(index), loc, builder);
        assert(indices && "static array initializer index should delinearize");
        builder.create<InsertArrayOp>(loc, newCreate.getResult(), ValueRange(*indices), element);
      }
      createOp.getResult().replaceAllUsesWith(newCreate.getResult());
      createOp.erase();
      return true;
    }
  }

  if (auto readOp = llvm::dyn_cast<ReadArrayOp>(op)) {
    Type newResultTy = converter.convertType(readOp.getResult().getType());
    if (converterFailed(converter)) {
      return failure();
    }
    if (auto newArrayTy = llvm::dyn_cast<ArrayType>(newResultTy)) {
      OpBuilder builder(readOp);
      auto extractOp = builder.create<ExtractArrayOp>(
          readOp.getLoc(), newArrayTy, readOp.getArrRef(), readOp.getIndices()
      );
      readOp.getResult().replaceAllUsesWith(extractOp.getResult());
      readOp.erase();
      return true;
    }
  }

  if (auto writeOp = llvm::dyn_cast<WriteArrayOp>(op)) {
    Type newRvalueTy = converter.convertType(writeOp.getRvalue().getType());
    if (converterFailed(converter)) {
      return failure();
    }
    if (llvm::isa<ArrayType>(newRvalueTy)) {
      OpBuilder builder(writeOp);
      builder.create<InsertArrayOp>(
          writeOp.getLoc(), writeOp.getArrRef(), writeOp.getIndices(), writeOp.getRvalue()
      );
      writeOp.erase();
      return true;
    }
  }

  bool changed = false;

  if (auto func = llvm::dyn_cast<FuncDefOp>(op)) {
    FunctionType oldFuncTy = func.getFunctionType();
    updateFuncSignature(func, converter);
    if (converterFailed(converter)) {
      return failure();
    }
    changed |= oldFuncTy != func.getFunctionType();
  }
  if (auto contract = llvm::dyn_cast<verif::ContractOp>(op)) {
    FunctionType oldFuncTy = contract.getFunctionType();
    updateContractSignature(contract, converter);
    if (converterFailed(converter)) {
      return failure();
    }
    changed |= oldFuncTy != contract.getFunctionType();
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (BlockArgument arg : block.getArguments()) {
        Type newTy = converter.convertType(arg.getType());
        if (converterFailed(converter)) {
          return failure();
        }
        if (newTy != arg.getType()) {
          arg.setType(newTy);
          changed = true;
        }
      }
    }
  }

  for (Value result : op->getResults()) {
    Type newTy = converter.convertType(result.getType());
    if (converterFailed(converter)) {
      return failure();
    }
    if (newTy != result.getType()) {
      result.setType(newTy);
      changed = true;
    }
  }

  if (auto callOp = llvm::dyn_cast<CallOp>(op)) {
    changed |= convertCallCallee(callOp, converter);
    if (converterFailed(converter)) {
      return failure();
    }
  }

  SmallVector<NamedAttribute> newAttrs;
  bool attrsChanged = false;
  newAttrs.reserve(op->getAttrs().size());
  for (NamedAttribute attr : op->getAttrs()) {
    Attribute newAttr = converter.convertAttr(attr.getValue());
    if (converterFailed(converter)) {
      return failure();
    }
    newAttrs.emplace_back(attr.getName(), newAttr);
    attrsChanged |= newAttr != attr.getValue();
  }
  if (attrsChanged) {
    op->setAttrs(DictionaryAttr::get(op->getContext(), newAttrs));
    changed = true;
  }

  return changed;
}

/// Convert all type-bearing surfaces under `root`, interrupting on the first failure.
template <typename ConverterT>
static FailureOr<bool> convertOperationTypesInAndTrack(Operation *root, ConverterT &converter) {
  bool changed = false;
  WalkResult res = root->walk([&converter, &changed](Operation *op) {
    startConverterOperation(converter, op);
    FailureOr<bool> opChanged = convertOperationTypes(op, converter);
    if (failed(opChanged)) {
      return WalkResult::interrupt();
    }
    changed |= *opChanged;
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) {
    return failure();
  }
  return changed;
}

/// Convert all type-bearing surfaces under `root`, interrupting on the first failure.
template <typename ConverterT>
static LogicalResult convertOperationTypesIn(Operation *root, ConverterT &converter) {
  return failure(failed(convertOperationTypesInAndTrack(root, converter)));
}

/// Remove `poly.unifiable_cast` operations that became identity casts.
///
/// After type replacement, a cast such as
/// `!poly.tvar<@T> -> !array.type<4 x index>` becomes
/// `!array.type<4 x index> -> !array.type<4 x index>`. Such casts no longer
/// carry information and can be replaced with their input value.
static void removeIdentityCasts(Operation *root) {
  for (UnifiableCastOp castOp : walkCollect<UnifiableCastOp>(*root)) {
    if (castOp.getInput().getType() != castOp.getResult().getType()) {
      continue;
    }
    castOp.getResult().replaceAllUsesWith(castOp.getInput());
    castOp.erase();
  }
}

/// Convert type-bearing surfaces in template expressions copied into a sibling template.
template <typename ConverterT>
static LogicalResult convertTemplateExprTypesIn(TemplateOp templateOp, ConverterT &converter) {
  for (TemplateExprOp expr : templateOp.getConstOps<TemplateExprOp>()) {
    if (failed(convertOperationTypesIn(expr.getOperation(), converter))) {
      return failure();
    }
    removeIdentityCasts(expr.getOperation());
  }
  return success();
}

/// Build a lossless key for a specialized sibling template.
///
/// The emitted template symbol uses `BuildShortTypeString`, which intentionally shortens some
/// types. This key prints full replacement types so distinct instantiations that prefer the same
/// name still get distinct generated templates.
static std::string buildSpecializedTemplateCloneCacheKey(
    StringRef templateName, ArrayRef<StringAttr> oldParamOrder,
    const DenseMap<Attribute, Attribute> &paramNameToConcrete
) {
  std::string key;
  llvm::raw_string_ostream os(key);
  os << templateName.size() << ':' << templateName;
  for (StringAttr paramName : oldParamOrder) {
    os << '|';
    os << paramName.getValue().size() << ':' << paramName.getValue() << '=';
    auto concreteIt = paramNameToConcrete.find(FlatSymbolRefAttr::get(paramName));
    if (concreteIt == paramNameToConcrete.end()) {
      os << '_';
      continue;
    }

    std::string attrText;
    llvm::raw_string_ostream attrOs(attrText);
    concreteIt->second.print(attrOs);
    os << attrText.size() << ':' << attrText;
  }
  return key;
}

/// Return true if `expr` only reads constants that will still be bound in a sibling template.
static bool canCopyTemplateExprToSpecialization(
    TemplateExprOp expr, const DenseSet<StringAttr> &availableNames
) {
  bool canCopy = true;
  expr->walk([&](ConstReadOp readOp) {
    if (!availableNames.contains(readOp.getConstNameAttr().getAttr())) {
      canCopy = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return canCopy;
}

/// Copy preserved `poly.expr` definitions into a sibling template.
///
/// A specialized clone keeps only the not-yet-concrete template parameters. Any
/// expression copied into it must therefore read only those remaining bindings.
static void copyPreservedTemplateExprs(
    TemplateOp parentTemplate, Block &newTemplateBody, ArrayRef<Attribute> remainingNames
) {
  DenseSet<StringAttr> availableNames;
  for (Attribute name : remainingNames) {
    FlatSymbolRefAttr nameSym = llvm::cast<FlatSymbolRefAttr>(name);
    availableNames.insert(nameSym.getAttr());
  }

  for (TemplateExprOp expr : parentTemplate.getConstOps<TemplateExprOp>()) {
    if (canCopyTemplateExprToSpecialization(expr, availableNames)) {
      newTemplateBody.push_back(expr->clone());
    }
  }
}

/// Create or reuse a sibling template named with the concrete replacement values.
static FailureOr<TemplateOp> getOrCreateSpecializedTemplateClone(
    TemplateOp parentTemplate, ArrayRef<StringAttr> oldParamOrder,
    const DenseMap<Attribute, Attribute> &paramNameToConcrete, ArrayAttr callParams,
    SymbolTableCollection &tables, llvm::StringMap<StringAttr> &templateClones,
    InstantiationLayout &layout
) {
  layout = buildInstantiationLayout(parentTemplate, callParams, paramNameToConcrete);
  std::string cacheKey = buildSpecializedTemplateCloneCacheKey(
      parentTemplate.getSymName(), oldParamOrder, paramNameToConcrete
  );

  ModuleOp parentModule = getParentOfType<ModuleOp>(parentTemplate);
  if (!parentModule) {
    return failure();
  }
  SymbolTable &moduleSymbols = tables.getSymbolTable(parentModule);

  auto cachedName = templateClones.find(cacheKey);
  if (cachedName != templateClones.end()) {
    Operation *existing = moduleSymbols.lookup(cachedName->second);
    if (auto existingTemplate = llvm::dyn_cast_or_null<TemplateOp>(existing)) {
      return existingTemplate;
    }
    return failure();
  }

  TemplateOp newTemplate = parentTemplate.cloneWithoutRegions();
  newTemplate.setSymName(layout.templateNameWithAttrs);
  assert(newTemplate->getNumRegions() > 0 && "region exists");
  newTemplate.getBodyRegion().emplaceBlock();
  Block &newTemplateBody = newTemplate.getBodyRegion().front();
  SymbolTable &parentTemplateSymbols = tables.getSymbolTable(parentTemplate);
  for (Attribute name : layout.remainingNames) {
    FlatSymbolRefAttr nameSym = llvm::cast<FlatSymbolRefAttr>(name);
    Operation *paramOp = parentTemplateSymbols.lookup(nameSym.getAttr());
    assert(paramOp && "symbol must exist");
    newTemplateBody.push_back(paramOp->clone());
  }
  copyPreservedTemplateExprs(parentTemplate, newTemplateBody, layout.remainingNames);

  moduleSymbols.insert(newTemplate, Block::iterator(parentTemplate));
  templateClones.try_emplace(cacheKey, newTemplate.getSymNameAttr());
  return newTemplate;
}

/// Convert concrete parameterized struct uses into sibling-template struct clones.
///
/// This intentionally implements only the fully-concrete case needed after
/// type-variable inference exposes an external instantiation such as
/// `!struct.type<@TBox::@Box<[index]>>`. More general partial struct
/// instantiation remains the flattening pass' job.
class ConcreteStructInstantiationConverter {
  struct StructInstantiationTypes {
    /// Type used inside the cloned struct body as the required self type.
    StructType localType;
    /// Type used by external concrete references to the clone.
    StructType remoteType;
  };

  /// Context used to allocate converted aggregate types and attributes.
  MLIRContext *ctx_;
  /// Root operation used for resolving struct definitions.
  ModuleOp module_;
  /// Symbol tables reused for lookup and insertion.
  SymbolTableCollection &tables_;
  /// Already-created instantiations keyed by the original concrete struct type.
  DenseMap<StructType, StructInstantiationTypes> instantiations_;
  /// Pass-created sibling templates keyed by owning template and concrete instantiation layout.
  SpecializedTemplateCloneCache templateClones_;
  /// Fully-qualified names of sibling-template clones created by this converter.
  DenseSet<SymbolRefAttr> instantiatedCloneNames_;
  /// Specialized self types active while rewriting a cloned struct body.
  DenseMap<StructType, StructType> activeLocalStructReplacements_;
  /// Type replacements active while rewriting the body of one cloned struct.
  DenseMap<StringAttr, Type> activeTypeReplacements_;
  /// Whether a nested conversion failed and emitted a diagnostic.
  bool hasFailure = false;

public:
  /// Create a converter for concrete struct instantiations in `module`.
  ConcreteStructInstantiationConverter(MLIRContext *c, ModuleOp m, SymbolTableCollection &t)
      : ctx_(c), module_(m), tables_(t) {}

  /// Return whether a nested conversion failed and emitted a diagnostic.
  bool hadFailure() const { return hasFailure; }

  /// Convert a type by recursively instantiating concrete parameterized structs.
  Type convertType(Type ty) {
    if (!ty || hasFailure) {
      return ty;
    }
    if (auto tvarTy = llvm::dyn_cast<TypeVarType>(ty)) {
      auto it = activeTypeReplacements_.find(tvarTy.getNameRef().getAttr());
      return it == activeTypeReplacements_.end() ? ty : it->second;
    }
    if (auto arrTy = llvm::dyn_cast<ArrayType>(ty)) {
      return convertArrayElementType(arrTy, [this](Type elemTy) { return convertType(elemTy); });
    }
    if (auto podTy = llvm::dyn_cast<PodType>(ty)) {
      return convertPodType(podTy, ctx_, *this);
    }
    if (auto funcTy = llvm::dyn_cast<FunctionType>(ty)) {
      return convertFunctionType(funcTy, *this);
    }
    if (auto structTy = llvm::dyn_cast<StructType>(ty)) {
      return convertStructType(structTy);
    }
    return ty;
  }

  /// Convert an attribute that can contain type references.
  Attribute convertAttr(Attribute attr) {
    if (hasFailure) {
      return attr;
    }
    return convertTypeOrArrayAttr(attr, ctx_, [this](Type ty) {
      return convertType(ty);
    }, [this](Attribute nested) { return convertTemplateArgAttr(nested); });
  }

  /// Convert a struct-function callee whose self/result type was instantiated.
  SymbolRefAttr convertCallCallee(CallOp callOp) {
    SymbolRefAttr callee = callOp.getCalleeAttr();
    if (!callee || callee.getNestedReferences().empty()) {
      return callee;
    }

    StructType targetStructTy = getStructFunctionTargetType(callOp);
    if (!targetStructTy) {
      return callee;
    }
    auto convertedStructTy = llvm::dyn_cast<StructType>(convertType(targetStructTy));
    if (!convertedStructTy) {
      return callee;
    }

    SymbolRefAttr convertedStructName = convertedStructTy.getNameRef();
    if (convertedStructName == getPrefixAsSymbolRefAttr(callee)) {
      return callee;
    }
    SmallVector<FlatSymbolRefAttr> pieces = getPieces(convertedStructName);
    pieces.push_back(FlatSymbolRefAttr::get(callee.getLeafReference()));
    return asSymbolRefAttr(pieces);
  }

private:
  /// Return the struct type that determines a struct function call's target.
  static StructType getStructFunctionTargetType(CallOp callOp) {
    StringAttr calleeLeaf = callOp.getCallee().getLeafReference();
    if (calleeLeaf == FUNC_NAME_CONSTRAIN) {
      return dyn_cast<StructType>(callOp.getSelfValueFromConstrain().getType());
    }
    if (calleeLeaf == FUNC_NAME_COMPUTE || calleeLeaf == FUNC_NAME_PRODUCT) {
      return callOp.getSingleResultTypeOfWitnessGen();
    }
    return {};
  }

  /// Convert an attribute that is known to be in a template-argument position.
  Attribute convertTemplateArgAttr(Attribute attr) {
    if (StringAttr symbolName = getFlatSymbolName(attr)) {
      auto it = activeTypeReplacements_.find(symbolName);
      if (it != activeTypeReplacements_.end()) {
        return TypeAttr::get(it->second);
      }
    }
    return convertAttr(attr);
  }

  /// Convert struct parameters and instantiate the struct if all parameters are concrete.
  StructType convertStructType(StructType structTy) {
    ArrayAttr params = structTy.getParams();
    if (!params) {
      return structTy;
    }

    SmallVector<Attribute> newParams;
    bool changed = false;
    for (Attribute attr : params.getValue()) {
      Attribute newAttr = convertTemplateArgAttr(attr);
      newParams.push_back(newAttr);
      changed |= newAttr != attr;
    }
    StructType convertedTy =
        changed ? StructType::get(structTy.getNameRef(), ArrayAttr::get(ctx_, newParams))
                : structTy;

    if (auto it = activeLocalStructReplacements_.find(convertedTy);
        it != activeLocalStructReplacements_.end()) {
      return it->second;
    }
    if (instantiatedCloneNames_.contains(convertedTy.getNameRef())) {
      return convertedTy;
    }

    if (llvm::any_of(newParams, [](Attribute attr) {
      return !isConcreteStructParamAttr(attr, /*allowStructParams=*/false);
    })) {
      return convertedTy;
    }

    FailureOr<StructType> cloneTy = getOrCreateStructClone(convertedTy, newParams);
    if (failed(cloneTy)) {
      hasFailure = true;
      return convertedTy;
    }
    return *cloneTy;
  }

  /// Create or reuse a pass-created sibling-template clone for `concreteStructTy`.
  FailureOr<StructType>
  getOrCreateStructClone(StructType concreteStructTy, ArrayRef<Attribute> concreteParams) {
    if (auto it = instantiations_.find(concreteStructTy); it != instantiations_.end()) {
      return it->second.remoteType;
    }

    FailureOr<SymbolLookupResult<StructDefOp>> lookup =
        concreteStructTy.getDefinition(tables_, module_, /*emitError=*/false);
    if (failed(lookup)) {
      return failure();
    }

    StructDefOp origStruct = lookup->get();
    TemplateOp parentTemplate = getParentOfType<TemplateOp>(origStruct);
    if (!parentTemplate) {
      return failure();
    }
    StructType typeAtDef = origStruct.getType();
    ArrayAttr paramNames = typeAtDef.getParams();
    if (!paramNames || paramNames.size() != concreteParams.size()) {
      return failure();
    }

    DenseMap<StringAttr, Type> typeReplacements;
    DenseMap<Attribute, Attribute> paramNameToConcrete;
    SmallVector<StringAttr> oldParamOrder;
    oldParamOrder.reserve(paramNames.size());
    SmallVector<Attribute> convertedSourceParams;
    convertedSourceParams.reserve(concreteParams.size());
    for (auto [paramName, concreteAttr] : llvm::zip_equal(paramNames.getValue(), concreteParams)) {
      auto paramSym = llvm::dyn_cast<FlatSymbolRefAttr>(paramName);
      if (!paramSym) {
        return failure();
      }
      oldParamOrder.push_back(paramSym.getAttr());
      auto concreteType = llvm::dyn_cast<TypeAttr>(concreteAttr);
      if (concreteType) {
        paramNameToConcrete.try_emplace(FlatSymbolRefAttr::get(paramSym.getAttr()), concreteAttr);
        typeReplacements.try_emplace(paramSym.getAttr(), concreteType.getValue());
        convertedSourceParams.push_back(concreteType);
      } else {
        convertedSourceParams.push_back(paramName);
      }
    }
    if (paramNameToConcrete.empty()) {
      return concreteStructTy;
    }

    InstantiationLayout layout;
    ArrayAttr concreteParamArray = ArrayAttr::get(ctx_, concreteParams);
    FailureOr<TemplateOp> newTemplate = getOrCreateSpecializedTemplateClone(
        parentTemplate, oldParamOrder, paramNameToConcrete, concreteParamArray, tables_,
        templateClones_[parentTemplate.getOperation()], layout
    );
    if (failed(newTemplate)) {
      return failure();
    }

    SymbolTable &templateSymbols = tables_.getSymbolTable(*newTemplate);
    StructDefOp clone = origStruct.clone();
    templateSymbols.insert(clone);
    StructType localTy =
        getStructTypeWithParams(clone.getFullyQualifiedName(), clone.getType().getParams());
    StructType remoteTy =
        getStructTypeWithParams(clone.getFullyQualifiedName(), layout.rewrittenCallParams);
    instantiations_.try_emplace(concreteStructTy, StructInstantiationTypes {localTy, remoteTy});
    instantiatedCloneNames_.insert(clone.getFullyQualifiedName());

    DenseMap<StringAttr, Type> previousReplacements = std::move(activeTypeReplacements_);
    DenseMap<StructType, StructType> previousLocalStructReplacements =
        std::move(activeLocalStructReplacements_);
    activeTypeReplacements_ = std::move(typeReplacements);
    activeLocalStructReplacements_.clear();
    activeLocalStructReplacements_.try_emplace(concreteStructTy, localTy);
    activeLocalStructReplacements_.try_emplace(
        StructType::get(typeAtDef.getNameRef(), ArrayAttr::get(ctx_, convertedSourceParams)),
        localTy
    );
    activeLocalStructReplacements_.try_emplace(
        StructType::get(localTy.getNameRef(), ArrayAttr::get(ctx_, convertedSourceParams)), localTy
    );
    activeLocalStructReplacements_.try_emplace(
        StructType::get(localTy.getNameRef(), ArrayAttr::get(ctx_, concreteParams)), localTy
    );
    SymbolRefAttr cloneNameRef = clone.getFullyQualifiedName();
    if (failed(convertTemplateExprTypesIn(*newTemplate, *this))) {
      activeLocalStructReplacements_ = std::move(previousLocalStructReplacements);
      activeTypeReplacements_ = std::move(previousReplacements);
      instantiations_.erase(concreteStructTy);
      instantiatedCloneNames_.erase(cloneNameRef);
      clone.erase();
      return failure();
    }
    if (failed(convertOperationTypesIn(clone.getOperation(), *this))) {
      activeLocalStructReplacements_ = std::move(previousLocalStructReplacements);
      activeTypeReplacements_ = std::move(previousReplacements);
      instantiations_.erase(concreteStructTy);
      instantiatedCloneNames_.erase(cloneNameRef);
      clone.erase();
      return failure();
    }
    removeIdentityCasts(clone.getOperation());
    activeLocalStructReplacements_ = std::move(previousLocalStructReplacements);
    activeTypeReplacements_ = std::move(previousReplacements);
    return remoteTy;
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
  /// Template-scope concrete replacements proven by `poly.expr` bodies.
  DenseMap<StringAttr, InferredType> templateScopeReplacements;
  /// Function-local concrete replacements before the template-wide safety merge.
  DenseMap<Operation *, DenseMap<StringAttr, InferredType>> functionReplacements;
};

/// Return whether a flat symbol reference names `paramName`.
static bool symbolMatchesParam(SymbolRefAttr symRef, StringAttr paramName) {
  return symRef && symRef.getNestedReferences().empty() && symRef.getRootReference() == paramName;
}

/// Checks whether types and type-bearing attributes mention one template parameter.
class ParamMentionChecker {
  StringAttr paramName_;

public:
  explicit ParamMentionChecker(StringAttr paramName) : paramName_(paramName) {}

  /// Return whether a type mentions the tracked type-variable parameter.
  bool typeMentions(Type ty) const {
    if (!ty) {
      return false;
    }
    if (auto tvarTy = llvm::dyn_cast<TypeVarType>(ty)) {
      return tvarTy.getNameRef().getAttr() == paramName_;
    }
    if (auto arrayTy = llvm::dyn_cast<ArrayType>(ty)) {
      return typeMentions(arrayTy.getElementType()) ||
             llvm::any_of(arrayTy.getDimensionSizes(), [this](Attribute dim) {
        return attrMentions(dim, /*allowSymbolRefs=*/true);
      });
    }
    if (auto structTy = llvm::dyn_cast<StructType>(ty)) {
      ArrayAttr params = structTy.getParams();
      return params && attrMentions(params, /*allowSymbolRefs=*/true);
    }
    if (auto podTy = llvm::dyn_cast<PodType>(ty)) {
      return llvm::any_of(podTy.getRecords(), [this](RecordAttr record) {
        return typeMentions(record.getType());
      });
    }
    if (auto funcTy = llvm::dyn_cast<FunctionType>(ty)) {
      return llvm::any_of(funcTy.getInputs(), [this](Type input) {
        return typeMentions(input);
      }) || llvm::any_of(funcTy.getResults(), [this](Type result) { return typeMentions(result); });
    }
    return false;
  }

  /// Return whether an attribute mentions the tracked type-variable parameter.
  ///
  /// Symbol references only count in template-argument positions, such as struct
  /// type parameter arrays or call-site template parameters. Top-level operation
  /// symbol attributes like callee or member names are not type-variable uses.
  bool attrMentions(Attribute attr, bool allowSymbolRefs) const {
    if (!attr) {
      return false;
    }
    if (auto typeAttr = llvm::dyn_cast<TypeAttr>(attr)) {
      return typeMentions(typeAttr.getValue());
    }
    if (auto arrayAttr = llvm::dyn_cast<ArrayAttr>(attr)) {
      return llvm::any_of(arrayAttr, [this](Attribute nested) {
        return attrMentions(nested, /*allowSymbolRefs=*/true);
      });
    }
    return allowSymbolRefs && symbolMatchesParam(llvm::dyn_cast<SymbolRefAttr>(attr), paramName_);
  }
};

/// Return whether `op` mentions an eligible type-variable parameter in a type-bearing surface.
static bool operationMentionsParam(Operation *op, StringAttr paramName) {
  ParamMentionChecker mentions(paramName);
  if (llvm::any_of(op->getResultTypes(), [&mentions](Type ty) {
    return mentions.typeMentions(ty);
  })) {
    return true;
  }
  for (Region &region : op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      if (llvm::any_of(block.getArgumentTypes(), [&mentions](Type ty) {
        return mentions.typeMentions(ty);
      })) {
        return true;
      }
    }
  }
  return llvm::any_of(op->getAttrs(), [&mentions](NamedAttribute attr) {
    return mentions.attrMentions(attr.getValue(), /*allowSymbolRefs=*/false);
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

/// Return whether a contract body/signature mentions an eligible type-variable parameter.
static bool contractMentionsParam(verif::ContractOp contract, StringAttr paramName) {
  if (operationMentionsParam(contract.getOperation(), paramName)) {
    return true;
  }
  WalkResult result = contract.walk([paramName](Operation *op) {
    return operationMentionsParam(op, paramName) ? WalkResult::interrupt() : WalkResult::advance();
  });
  return result.wasInterrupted();
}

static bool targetMentionsParam(FuncDefOp func, StringAttr paramName) {
  return funcMentionsParam(func, paramName);
}

static bool targetMentionsParam(verif::ContractOp contract, StringAttr paramName) {
  return contractMentionsParam(contract, paramName);
}

/// Return whether a `poly.expr` body mentions an eligible type-variable parameter.
static bool exprMentionsParam(TemplateExprOp expr, StringAttr paramName) {
  WalkResult result = expr.walk([paramName](Operation *op) {
    return operationMentionsParam(op, paramName) ? WalkResult::interrupt() : WalkResult::advance();
  });
  return result.wasInterrupted();
}

/// Return whether a struct's non-function mentions are covered by its own
/// structural function proofs.
static bool structUseCoveredByFunctionProof(
    StructDefOp structOp, StringAttr paramName, Type replacementType,
    const DenseMap<Operation *, DenseMap<StringAttr, InferredType>> &functionReplacements
) {
  bool sawMention = false;
  bool missingProof = false;
  structOp.walk([&](FuncDefOp func) {
    if (func->getParentOfType<StructDefOp>() != structOp || !funcMentionsParam(func, paramName)) {
      return;
    }
    sawMention = true;
    auto funcIt = functionReplacements.find(func.getOperation());
    if (funcIt == functionReplacements.end()) {
      missingProof = true;
      return;
    }
    auto replacementIt = funcIt->second.find(paramName);
    if (replacementIt == funcIt->second.end() || replacementIt->second.type != replacementType) {
      missingProof = true;
    }
  });
  return sawMention && !missingProof;
}

/// Return whether `templateOp` has non-function type-bearing mentions that are
/// not covered by their own structural function proofs.
///
/// Uncovered uses do not have function-local proof sites, so a concrete
/// inference from one function cannot safely become a template-wide replacement
/// while they remain. Keeping the replacement function-local preserves sibling
/// symbols such as generic struct members.
static bool hasUncoveredNonFunctionMention(
    TemplateOp templateOp, StringAttr paramName, Type replacementType,
    const DenseMap<Operation *, DenseMap<StringAttr, InferredType>> &functionReplacements
) {
  WalkResult result =
      templateOp.walk([paramName, replacementType, &functionReplacements](Operation *op) {
    if (llvm::isa<FuncDefOp, TemplateExprOp, TemplateParamOp, verif::ContractOp>(op) ||
        hasParentThatIsa<FuncDefOp, TemplateExprOp, verif::ContractOp>(op)) {
      return WalkResult::advance();
    }
    if (!operationMentionsParam(op, paramName)) {
      return WalkResult::advance();
    }
    if (StructDefOp structOp = op->getParentOfType<StructDefOp>()) {
      if (structUseCoveredByFunctionProof(
              structOp, paramName, replacementType, functionReplacements
          )) {
        return WalkResult::advance();
      }
    }
    return WalkResult::interrupt();
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
  /// Type-variable parameters proven to be equal by casts or aggregate contents.
  DenseMap<StringAttr, DenseSet<StringAttr>> paramRelations;
  /// Whether the current collection iteration learned a new concrete fact.
  bool changedInIteration = false;

public:
  /// Create a collector for one function's inference state.
  TypeVarInferenceCollector(
      TemplateInferenceInfo &info, DenseMap<StringAttr, InferredType> &scopeReplacements
  )
      : inferenceInfo(info), replacements(scopeReplacements) {}

  /// Visit all casts until no new concrete parameter or SSA value facts appear.
  LogicalResult collect(Operation *root) {
    do {
      changedInIteration = false;
      WalkResult result = root->walk([this](UnifiableCastOp castOp) {
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

  /// Collect inferences from one proven type relationship.
  LogicalResult collectTypeInferences(Type lhs, Type rhs, Location loc) {
    do {
      changedInIteration = false;
      if (failed(collectTypePairInferences(lhs, rhs, Value(), Value(), loc))) {
        return failure();
      }
    } while (changedInIteration);
    return success();
  }

  /// Walk `root` and collect cross-template inferences from every type-bearing surface.
  ///
  /// This extends the normal cast-based collection with facts implied by struct type
  /// instantiations. If a type site in the current template references another template-owned
  /// struct whose owner already proved a parameter concrete, the corresponding argument at the use
  /// site is unified with that concrete replacement. For example, when `@TBox::@T` is known to be
  /// `index`, a use of `!struct.type<@TBox::@Box<[@U]>>` records `@U = index` before `@TBox::@T`
  /// is erased.
  LogicalResult collectStructTemplateParamInferences(
      Operation *root, ModuleOp module, DenseMap<Operation *, TemplateInferenceInfo *> &infos,
      SymbolTableCollection &tables
  ) {
    WalkResult result = root->walk([&](Operation *op) {
      Location loc = op->getLoc();

      if (auto func = llvm::dyn_cast<FuncDefOp>(op)) {
        if (failed(collectStructTemplateParamInferences(
                func.getFunctionType(), loc, module, infos, tables
            ))) {
          return WalkResult::interrupt();
        }
      }

      for (Region &region : op->getRegions()) {
        for (Block &block : region.getBlocks()) {
          for (Type argTy : block.getArgumentTypes()) {
            if (failed(collectStructTemplateParamInferences(argTy, loc, module, infos, tables))) {
              return WalkResult::interrupt();
            }
          }
        }
      }

      for (Type resultTy : op->getResultTypes()) {
        if (failed(collectStructTemplateParamInferences(resultTy, loc, module, infos, tables))) {
          return WalkResult::interrupt();
        }
      }

      if (auto callOp = llvm::dyn_cast<CallOp>(op)) {
        if (failed(collectCallableTemplateParamInferences(callOp, infos, tables))) {
          return WalkResult::interrupt();
        }
      }
      if (auto includeOp = llvm::dyn_cast<verif::IncludeOp>(op)) {
        if (failed(collectIncludeTemplateParamInferences(includeOp, infos, tables))) {
          return WalkResult::interrupt();
        }
      }

      for (NamedAttribute attr : op->getAttrs()) {
        if (failed(
                collectStructTemplateParamInferences(attr.getValue(), loc, module, infos, tables)
            )) {
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

private:
  /// Recurse through an attribute that may contain type-site struct references.
  ///
  /// Struct template arguments can be nested in `TypeAttr` or in array-shaped attributes used by
  /// template parameter lists, so this preserves the same template-argument traversal shape used by
  /// the rewrite converters.
  LogicalResult collectStructTemplateParamInferences(
      Attribute attr, Location loc, ModuleOp module,
      DenseMap<Operation *, TemplateInferenceInfo *> &infos, SymbolTableCollection &tables
  ) {
    if (!attr) {
      return success();
    }
    if (auto tyAttr = llvm::dyn_cast<TypeAttr>(attr)) {
      return collectStructTemplateParamInferences(tyAttr.getValue(), loc, module, infos, tables);
    }
    if (auto arrAttr = llvm::dyn_cast<ArrayAttr>(attr)) {
      for (Attribute nested : arrAttr.getValue()) {
        if (failed(collectStructTemplateParamInferences(nested, loc, module, infos, tables))) {
          return failure();
        }
      }
    }
    return success();
  }

  /// Recurse through a type and unify arguments of referenced rewritten struct templates.
  ///
  /// Aggregate containers are searched structurally. At each `StructType`, the referenced
  /// definition is resolved; if its owning template has concrete replacements and the type still
  /// has the original parameter arity, each soon-to-be-erased parameter position is checked with
  /// `collectTemplateArgInferences`. That deliberately uses unification rather than equality, so a
  /// symbolic argument from the current template can become a concrete replacement instead of being
  /// rejected as a mismatch.
  LogicalResult collectStructTemplateParamInferences(
      Type ty, Location loc, ModuleOp module, DenseMap<Operation *, TemplateInferenceInfo *> &infos,
      SymbolTableCollection &tables
  ) {
    if (!ty) {
      return success();
    }
    if (auto arrTy = llvm::dyn_cast<ArrayType>(ty)) {
      return collectStructTemplateParamInferences(
          arrTy.getElementType(), loc, module, infos, tables
      );
    }
    if (auto podTy = llvm::dyn_cast<PodType>(ty)) {
      for (RecordAttr record : podTy.getRecords()) {
        if (failed(
                collectStructTemplateParamInferences(record.getType(), loc, module, infos, tables)
            )) {
          return failure();
        }
      }
      return success();
    }
    if (auto funcTy = llvm::dyn_cast<FunctionType>(ty)) {
      for (Type inputTy : funcTy.getInputs()) {
        if (failed(collectStructTemplateParamInferences(inputTy, loc, module, infos, tables))) {
          return failure();
        }
      }
      for (Type resultTy : funcTy.getResults()) {
        if (failed(collectStructTemplateParamInferences(resultTy, loc, module, infos, tables))) {
          return failure();
        }
      }
      return success();
    }
    auto structTy = llvm::dyn_cast<StructType>(ty);
    if (!structTy) {
      return success();
    }

    ArrayAttr params = structTy.getParams();
    if (!params) {
      return success();
    }
    for (Attribute attr : params.getValue()) {
      if (failed(collectStructTemplateParamInferences(attr, loc, module, infos, tables))) {
        return failure();
      }
    }

    FailureOr<SymbolLookupResult<StructDefOp>> lookup =
        structTy.getDefinition(tables, module, /*reportMissing=*/false);
    if (failed(lookup)) {
      return success();
    }
    TemplateOp parentTemplate = getParentOfType<TemplateOp>(lookup->get().getOperation());
    if (!parentTemplate) {
      return success();
    }
    TemplateInferenceInfo *targetInfo = infos.lookup(parentTemplate.getOperation());
    if (!targetInfo || targetInfo->replacements.empty() ||
        params.size() != targetInfo->oldParamOrder.size()) {
      return success();
    }

    for (auto [paramName, attr] : llvm::zip_equal(targetInfo->oldParamOrder, params.getValue())) {
      auto replacementIt = targetInfo->replacements.find(paramName);
      if (replacementIt == targetInfo->replacements.end()) {
        continue;
      }
      Attribute expectedAttr = TypeAttr::get(replacementIt->second.type);
      if (failed(collectTemplateArgInferences(attr, expectedAttr, loc))) {
        return failure();
      }
    }
    return success();
  }

  /// Return the arguments inferred for an omitted target template parameter.
  ///
  /// The unifier records a direct RHS entry when a target type variable is matched by a concrete
  /// type. If both sides are type variables, it may instead record the caller-side variable as a
  /// LHS entry whose value is the target parameter. That still represents a forwarded template
  /// argument and must be propagated before the target parameter is erased.
  SmallVector<Attribute>
  getInferredOmittedTemplateArgs(const UnificationMap &unifyResult, StringAttr targetParamName) {
    SmallVector<Attribute> inferredAttrs;
    auto targetRef = FlatSymbolRefAttr::get(targetParamName);
    auto inferredIt = unifyResult.find({targetRef, Side::RHS});
    if (inferredIt != unifyResult.end()) {
      inferredAttrs.push_back(inferredIt->second);
    }
    for (const auto &entry : unifyResult) {
      if (entry.first.second == Side::LHS && entry.second == targetRef) {
        inferredAttrs.push_back(entry.first.first);
      }
    }
    return inferredAttrs;
  }

  /// Return template inference state for the template that owns a resolved target operation.
  template <typename TargetOp>
  TemplateInferenceInfo *
  getTargetTemplateInfo(TargetOp targetOp, DenseMap<Operation *, TemplateInferenceInfo *> &infos) {
    TemplateOp parentTemplate = getParentOfType<TemplateOp>(targetOp.getOperation());
    if (!parentTemplate) {
      return nullptr;
    }
    TemplateInferenceInfo *targetInfo = infos.lookup(parentTemplate.getOperation());
    if (!targetInfo || targetInfo->replacements.empty()) {
      return nullptr;
    }
    return targetInfo;
  }

  /// Infer current-template parameters from a call/include use of a rewritten template target.
  ///
  /// Function calls and contract includes have the same template-argument surface. Explicit
  /// arguments carry forwarded parameter facts directly, while omitted arguments are recovered by
  /// unifying the use-site signature with the target's original signature.
  template <typename CallableOp>
  LogicalResult collectCallableUseTemplateParamInferences(
      CallableOp callableOp, FunctionType targetSignature, TemplateInferenceInfo &targetInfo
  ) {
    ArrayAttr params = callableOp.getTemplateParamsAttr();
    if (!isNullOrEmpty(params)) {
      if (params.size() != targetInfo.oldParamOrder.size()) {
        return success();
      }
      for (auto [paramName, attr] : llvm::zip_equal(targetInfo.oldParamOrder, params)) {
        auto replacementIt = targetInfo.replacements.find(paramName);
        if (replacementIt == targetInfo.replacements.end()) {
          continue;
        }
        Attribute expectedAttr = TypeAttr::get(replacementIt->second.type);
        if (failed(collectTemplateArgInferences(attr, expectedAttr, callableOp.getLoc()))) {
          return failure();
        }
      }
      return success();
    }

    FailureOr<UnificationMap> unifyResult = callableOp.unifyTypeSignature(targetSignature);
    if (failed(unifyResult)) {
      return success();
    }

    for (StringAttr paramName : targetInfo.oldParamOrder) {
      auto replacementIt = targetInfo.replacements.find(paramName);
      if (replacementIt == targetInfo.replacements.end()) {
        continue;
      }
      SmallVector<Attribute> inferredAttrs =
          getInferredOmittedTemplateArgs(*unifyResult, paramName);
      if (inferredAttrs.empty()) {
        continue;
      }

      Attribute expectedAttr = TypeAttr::get(replacementIt->second.type);
      for (Attribute inferredAttr : inferredAttrs) {
        if (!inferredAttr || !typeParamsUnify({inferredAttr}, {expectedAttr})) {
          InFlightDiagnostic diag = callableOp.emitError()
                                    << "implicit template argument for inferred parameter @"
                                    << paramName.getValue() << " must match inferred type "
                                    << replacementIt->second.type << ", but found ";
          if (auto typeAttr = llvm::dyn_cast_if_present<TypeAttr>(inferredAttr)) {
            diag << typeAttr.getValue();
          } else {
            diag << inferredAttr;
          }
          return diag;
        }
        if (failed(collectTemplateArgInferences(inferredAttr, expectedAttr, callableOp.getLoc()))) {
          return failure();
        }
      }
    }
    return success();
  }

  /// Infer current-template parameters from calls into rewritten templates.
  ///
  /// A call may omit its template argument list when the original callee signature exposes every
  /// target template parameter. If that target template has since proven a parameter concrete, the
  /// omitted argument inferred from the call signature must either agree with the concrete proof or
  /// propagate the proof into the caller's template parameter before the callee signature is
  /// rewritten. Explicit template arguments carry the same cross-template facts directly in the
  /// argument list, so forwarded caller parameters are unified with the target's concrete proof
  /// before resolved callee parameters are trimmed.
  LogicalResult collectCallableTemplateParamInferences(
      CallOp callOp, DenseMap<Operation *, TemplateInferenceInfo *> &infos,
      SymbolTableCollection &tables
  ) {
    FailureOr<SymbolLookupResult<FuncDefOp>> target = callOp.getCalleeTarget(tables);
    if (failed(target)) {
      return success();
    }
    FuncDefOp targetFunc = target->get();
    TemplateInferenceInfo *targetInfo = getTargetTemplateInfo(targetFunc, infos);
    if (!targetInfo) {
      return success();
    }
    return collectCallableUseTemplateParamInferences(
        callOp, targetFunc.getFunctionType(), *targetInfo
    );
  }

  /// Infer current-template parameters from contract includes into rewritten templates.
  ///
  /// `verif.include` has the same optional template-parameter behavior as function calls. When an
  /// include omits arguments, the target contract signature can expose the target template
  /// parameter and let unification prove the current template's forwarded parameter. Explicit
  /// include arguments carry the same fact directly in the argument list.
  LogicalResult collectIncludeTemplateParamInferences(
      verif::IncludeOp includeOp, DenseMap<Operation *, TemplateInferenceInfo *> &infos,
      SymbolTableCollection &tables
  ) {
    FailureOr<SymbolLookupResult<verif::ContractOp>> target = includeOp.getCalleeTarget(tables);
    if (failed(target)) {
      return success();
    }
    verif::ContractOp targetContract = target->get();
    TemplateInferenceInfo *targetInfo = getTargetTemplateInfo(targetContract, infos);
    if (!targetInfo) {
      return success();
    }
    return collectCallableUseTemplateParamInferences(
        includeOp, targetContract.getFunctionType(), *targetInfo
    );
  }

  /// Record one concrete type inference and diagnose conflicts.
  ///
  /// Inferences are tracked in two dimensions:
  ///   * by template parameter, so `@T` cannot be proven to be two different
  ///     concrete types;
  ///   * by SSA value, so a single value cannot be rewritten to incompatible
  ///     concrete types through separate casts.
  ///
  /// Non-eligible parameters are ignored. A candidate type variable records a
  /// parameter equality instead of a concrete replacement, so later concrete
  /// proofs can propagate across aggregate casts such as `array<T> -> array<U>`.
  /// Other non-concrete candidates are ignored because this pass only removes
  /// template type variables once a concrete replacement is known.
  LogicalResult recordInference(StringAttr paramName, Type inferredTy, Value value, Location loc) {
    if (!inferenceInfo.typeVarParams.contains(paramName)) {
      return success();
    }
    if (auto inferredTvar = llvm::dyn_cast<TypeVarType>(inferredTy)) {
      return recordParamRelation(paramName, inferredTvar.getNameRef().getAttr(), loc);
    }
    if (!isConcreteType(inferredTy)) {
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
    bool learnedParamInference = false;
    if (byParamIt == replacements.end()) {
      replacements.try_emplace(paramName, InferredType {inferredTy, loc});
      changedInIteration = true;
      learnedParamInference = true;
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

    if (learnedParamInference) {
      auto relatedIt = paramRelations.find(paramName);
      if (relatedIt != paramRelations.end()) {
        for (StringAttr relatedParam : relatedIt->second) {
          if (failed(recordInference(relatedParam, inferredTy, Value(), loc))) {
            return failure();
          }
        }
      }
    }
    return success();
  }

  /// Record that two eligible type-variable parameters must have the same type.
  LogicalResult recordParamRelation(StringAttr lhsParam, StringAttr rhsParam, Location loc) {
    if (lhsParam == rhsParam || !inferenceInfo.typeVarParams.contains(rhsParam)) {
      return success();
    }

    bool inserted = paramRelations[lhsParam].insert(rhsParam).second;
    inserted |= paramRelations[rhsParam].insert(lhsParam).second;
    if (!inserted) {
      return success();
    }

    changedInIteration = true;

    auto lhsIt = replacements.find(lhsParam);
    if (lhsIt != replacements.end() &&
        failed(recordInference(rhsParam, lhsIt->second.type, Value(), loc))) {
      return failure();
    }
    auto rhsIt = replacements.find(rhsParam);
    if (rhsIt != replacements.end() &&
        failed(recordInference(lhsParam, rhsIt->second.type, Value(), loc))) {
      return failure();
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
  /// an eligible self-typed type-variable parameter; two flat symbol arguments
  /// record a parameter equality. `recordInference` and `recordParamRelation`
  /// filter out ordinary value parameters and non-concrete candidates.
  LogicalResult collectTemplateArgInferences(Attribute lhsAttr, Attribute rhsAttr, Location loc) {
    auto lhsTyAttr = llvm::dyn_cast<TypeAttr>(lhsAttr);
    auto rhsTyAttr = llvm::dyn_cast<TypeAttr>(rhsAttr);
    if (lhsTyAttr && rhsTyAttr) {
      return collectTypePairInferences(
          lhsTyAttr.getValue(), rhsTyAttr.getValue(), Value(), Value(), loc
      );
    }

    if (StringAttr lhsSymbolName = getFlatSymbolName(lhsAttr)) {
      if (StringAttr rhsSymbolName = getFlatSymbolName(rhsAttr)) {
        return recordParamRelation(lhsSymbolName, rhsSymbolName, loc);
      }
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

/// Build concrete replacement types proven by a function body.
static DenseMap<StringAttr, Type>
getFunctionProofReplacements(const TemplateInferenceInfo &info, FuncDefOp func) {
  DenseMap<StringAttr, Type> replacements;
  auto funcIt = info.functionReplacements.find(func.getOperation());
  if (funcIt == info.functionReplacements.end()) {
    return replacements;
  }
  for (const auto &entry : funcIt->second) {
    replacements.try_emplace(entry.first, entry.second.type);
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

/// Convert a concrete type-variable replacement map to the attribute map consumed by shared
/// template-instantiation layout helpers.
static DenseMap<Attribute, Attribute>
buildParamNameToConcrete(const DenseMap<StringAttr, Type> &replacements) {
  DenseMap<Attribute, Attribute> paramNameToConcrete;
  for (const auto &entry : replacements) {
    paramNameToConcrete.try_emplace(
        FlatSymbolRefAttr::get(entry.first), TypeAttr::get(entry.second)
    );
  }
  return paramNameToConcrete;
}

/// Return true if explicit call-site template parameters match the inferred
/// concrete function-local replacements.
static Type
substituteExplicitCallTypeArgs(Type ty, ArrayAttr callParams, ArrayRef<StringAttr> oldParamOrder) {
  if (!callParams || callParams.size() != oldParamOrder.size()) {
    return ty;
  }

  DenseMap<StringAttr, Type> callTypeArgs;
  for (auto [paramName, attr] : llvm::zip_equal(oldParamOrder, callParams.getValue())) {
    if (auto tyAttr = llvm::dyn_cast<TypeAttr>(attr)) {
      callTypeArgs.try_emplace(paramName, tyAttr.getValue());
    }
  }
  TypeVarReplacementConverter converter(
      ty.getContext(), ArrayRef<StringAttr> {}, oldParamOrder, callTypeArgs,
      /*trimResolvedParams=*/false
  );
  return converter.convertType(ty);
}

/// Return true if explicit call-site template parameters match the inferred
/// concrete function-local replacements after substituting remaining explicit
/// type arguments into symbolic replacement types.
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
    Type expectedTy = substituteExplicitCallTypeArgs(entry.second, callParams, oldParamOrder);
    if (!templateArgUnifiesWithType(callParams[*index], expectedTy)) {
      return false;
    }
  }
  return true;
}

/// Diagnose call-site template parameters that disagree with concrete
/// replacements proven inside the callee body.
///
/// Omitted call parameters can be materialized from the call signature before
/// this check runs. The `explicitCallParams` flag keeps the diagnostic aligned
/// with the user's source syntax.
static LogicalResult diagnoseCallParamsMismatch(
    Operation *callableOp, ArrayAttr callParams, ArrayRef<StringAttr> oldParamOrder,
    const DenseMap<StringAttr, Type> &replacements, bool explicitCallParams
) {
  if (callParamsMatchReplacements(callParams, oldParamOrder, replacements)) {
    return success();
  }
  for (const auto &entry : replacements) {
    std::optional<unsigned> index = getParamIndex(oldParamOrder, entry.first);
    if (!index || !callParams || *index >= callParams.size()) {
      continue;
    }
    Attribute attr = callParams[*index];
    Type expectedTy = substituteExplicitCallTypeArgs(entry.second, callParams, oldParamOrder);
    if (templateArgUnifiesWithType(attr, expectedTy)) {
      continue;
    }

    InFlightDiagnostic diag = callableOp->emitError()
                              << (explicitCallParams ? "explicit" : "implicit")
                              << " template argument for inferred parameter @"
                              << entry.first.getValue() << " must match inferred type "
                              << expectedTy << ", but found ";
    if (auto typeAttr = llvm::dyn_cast<TypeAttr>(attr)) {
      diag << typeAttr.getValue();
    } else {
      diag << attr;
    }
    return diag;
  }
  return callableOp->emitError() << (explicitCallParams ? "explicit" : "implicit")
                                 << " template arguments do not match inferred callee types";
}

/// Expand a call/include template argument list that has already had
/// template-wide resolved parameters trimmed back to the original parameter
/// order used by inference metadata.
static ArrayAttr expandCurrentTemplateParamsToOriginalOrder(
    ArrayAttr callParams, const TemplateInferenceInfo &info
) {
  if (!callParams || callParams.size() == info.oldParamOrder.size()) {
    return callParams;
  }

  unsigned currentParamCount = llvm::count_if(info.oldParamOrder, [&info](StringAttr paramName) {
    return !info.replacements.contains(paramName);
  });
  if (callParams.size() != currentParamCount) {
    return callParams;
  }

  SmallVector<Attribute> expandedParams;
  expandedParams.reserve(info.oldParamOrder.size());
  unsigned currentIndex = 0;
  for (StringAttr paramName : info.oldParamOrder) {
    auto replacementIt = info.replacements.find(paramName);
    if (replacementIt != info.replacements.end()) {
      expandedParams.push_back(TypeAttr::get(replacementIt->second.type));
      continue;
    }
    expandedParams.push_back(callParams[currentIndex++]);
  }
  return ArrayAttr::get(callParams.getContext(), expandedParams);
}

/// Build concrete type-variable replacements from an explicit call instantiation.
///
/// A specialized function clone keeps the original enclosing `poly.template`,
/// so non-type parameters remain available to operations such as
/// `poly.read_const`. Type-variable parameters mentioned by the cloned function
/// are replaced by concrete types from the call's template argument list.
template <typename TargetOp>
static DenseMap<StringAttr, Type> getConcreteCallSiteReplacements(
    ArrayAttr callParams, const TemplateInferenceInfo &info, TargetOp targetOp
) {
  DenseMap<StringAttr, Type> replacements;
  if (!callParams || callParams.size() != info.oldParamOrder.size()) {
    return replacements;
  }

  bool sawResidualReplacement = false;
  for (const auto &entry : info.typeVarParams) {
    StringAttr paramName = entry.first;
    if (info.replacements.contains(paramName)) {
      continue;
    }
    if (!targetMentionsParam(targetOp, paramName)) {
      continue;
    }
    std::optional<unsigned> index = getParamIndex(info.oldParamOrder, paramName);
    assert(index && "eligible type-variable parameter must appear in parameter order");
    Attribute attr = callParams[*index];
    auto tyAttr = llvm::dyn_cast<TypeAttr>(attr);
    if (!tyAttr) {
      return DenseMap<StringAttr, Type>();
    }
    replacements.try_emplace(paramName, tyAttr.getValue());
    sawResidualReplacement = true;
  }
  if (!sawResidualReplacement) {
    return DenseMap<StringAttr, Type>();
  }

  for (const auto &entry : info.replacements) {
    replacements.try_emplace(entry.first, entry.second.type);
  }

  TypeVarReplacementConverter converter(
      info.templatePath.front().getContext(), info.templatePath, info.oldParamOrder, replacements,
      /*trimResolvedParams=*/false
  );
  for (const auto &entry : replacements) {
    if (!isConcreteType(converter.convertType(entry.second))) {
      return DenseMap<StringAttr, Type>();
    }
  }
  return replacements;
}

/// Return true when an attribute is the parser's `?` template-argument marker.
static bool isWildcardTemplateArg(Attribute attr) {
  auto intAttr = llvm::dyn_cast_if_present<IntegerAttr>(attr);
  return intAttr && isDynamic(intAttr);
}

/// Append `ty` to `types` unless it is already present.
static void appendUniqueType(SmallVectorImpl<Type> &types, Type ty) {
  if (!llvm::any_of(types, [ty](Type existing) { return existing == ty; })) {
    types.push_back(ty);
  }
}

/// Collect call-side types that bind to `targetRef` at matching callee type positions.
static void collectRhsTypeVarCandidates(
    Type lhsTy, Type rhsTy, SymbolRefAttr targetRef, SmallVectorImpl<Type> &candidates
);

/// Collect call-side types from matching template parameter attribute lists.
static void collectRhsTypeVarCandidates(
    ArrayRef<Attribute> lhsAttrs, ArrayRef<Attribute> rhsAttrs, SymbolRefAttr targetRef,
    SmallVectorImpl<Type> &candidates
) {
  if (lhsAttrs.size() != rhsAttrs.size()) {
    return;
  }
  for (auto [lhsAttr, rhsAttr] : llvm::zip_equal(lhsAttrs, rhsAttrs)) {
    collectRhsTypeVarCandidates(lhsAttr, rhsAttr, targetRef, candidates);
  }
}

/// Collect call-side types from matching template parameter attributes.
static void collectRhsTypeVarCandidates(
    Attribute lhsAttr, Attribute rhsAttr, SymbolRefAttr targetRef, SmallVectorImpl<Type> &candidates
) {
  if (auto rhsTypeAttr = llvm::dyn_cast_if_present<TypeAttr>(rhsAttr)) {
    if (auto lhsTypeAttr = llvm::dyn_cast_if_present<TypeAttr>(lhsAttr)) {
      collectRhsTypeVarCandidates(
          lhsTypeAttr.getValue(), rhsTypeAttr.getValue(), targetRef, candidates
      );
    }
    return;
  }

  auto lhsArray = llvm::dyn_cast_if_present<ArrayAttr>(lhsAttr);
  auto rhsArray = llvm::dyn_cast_if_present<ArrayAttr>(rhsAttr);
  if (!lhsArray || !rhsArray || lhsArray.size() != rhsArray.size()) {
    return;
  }
  collectRhsTypeVarCandidates(lhsArray.getValue(), rhsArray.getValue(), targetRef, candidates);
}

/// Collect call-side types from matching type structure.
///
/// The normal unifier records a null value after two different types bind to
/// the same RHS type variable. This helper reconstructs those concrete
/// candidates for diagnostics by walking the same call/callee type positions.
static void collectRhsTypeVarCandidates(
    Type lhsTy, Type rhsTy, SymbolRefAttr targetRef, SmallVectorImpl<Type> &candidates
) {
  if (auto rhsTvar = llvm::dyn_cast<TypeVarType>(rhsTy);
      rhsTvar && rhsTvar.getNameRef() == targetRef) {
    appendUniqueType(candidates, lhsTy);
    return;
  }

  if (auto lhsArray = llvm::dyn_cast<ArrayType>(lhsTy)) {
    if (auto rhsArray = llvm::dyn_cast<ArrayType>(rhsTy)) {
      collectRhsTypeVarCandidates(
          lhsArray.getElementType(), rhsArray.getElementType(), targetRef, candidates
      );
      collectRhsTypeVarCandidates(
          lhsArray.getDimensionSizes(), rhsArray.getDimensionSizes(), targetRef, candidates
      );
    }
    return;
  }

  if (auto lhsStruct = llvm::dyn_cast<StructType>(lhsTy)) {
    if (auto rhsStruct = llvm::dyn_cast<StructType>(rhsTy)) {
      collectRhsTypeVarCandidates(
          lhsStruct.getParams(), rhsStruct.getParams(), targetRef, candidates
      );
    }
    return;
  }

  if (auto lhsPod = llvm::dyn_cast<PodType>(lhsTy)) {
    if (auto rhsPod = llvm::dyn_cast<PodType>(rhsTy);
        rhsPod && lhsPod.getRecords().size() == rhsPod.getRecords().size()) {
      for (auto [lhsRecord, rhsRecord] :
           llvm::zip_equal(lhsPod.getRecords(), rhsPod.getRecords())) {
        collectRhsTypeVarCandidates(
            lhsRecord.getType(), rhsRecord.getType(), targetRef, candidates
        );
      }
    }
    return;
  }

  if (auto lhsFunc = llvm::dyn_cast<FunctionType>(lhsTy)) {
    if (auto rhsFunc = llvm::dyn_cast<FunctionType>(rhsTy)) {
      for (auto [lhsInput, rhsInput] : llvm::zip_equal(lhsFunc.getInputs(), rhsFunc.getInputs())) {
        collectRhsTypeVarCandidates(lhsInput, rhsInput, targetRef, candidates);
      }
      for (auto [lhsResult, rhsResult] :
           llvm::zip_equal(lhsFunc.getResults(), rhsFunc.getResults())) {
        collectRhsTypeVarCandidates(lhsResult, rhsResult, targetRef, candidates);
      }
    }
  }
}

/// Return unique call-side types that conflict for an omitted callee type variable.
static SmallVector<Type>
getConflictingRhsTypeVarCandidates(FunctionType lhs, FunctionType rhs, SymbolRefAttr targetRef) {
  SmallVector<Type> candidates;
  if (lhs.getNumInputs() != rhs.getNumInputs() || lhs.getNumResults() != rhs.getNumResults()) {
    return candidates;
  }
  for (auto [lhsInput, rhsInput] : llvm::zip_equal(lhs.getInputs(), rhs.getInputs())) {
    collectRhsTypeVarCandidates(lhsInput, rhsInput, targetRef, candidates);
  }
  for (auto [lhsResult, rhsResult] : llvm::zip_equal(lhs.getResults(), rhs.getResults())) {
    collectRhsTypeVarCandidates(lhsResult, rhsResult, targetRef, candidates);
  }
  return candidates;
}

/// Emit the shared diagnostic for ambiguous call-signature inference.
template <typename CallableOp, typename TargetOp>
static LogicalResult
emitConflictingInferredTypes(CallableOp callableOp, TargetOp targetOp, StringAttr paramName) {
  InFlightDiagnostic diag = callableOp.emitError()
                            << "conflicting inferred types for @" << paramName.getValue();
  SmallVector<Type> candidates = getConflictingRhsTypeVarCandidates(
      callableOp.getTypeSignature(), targetOp.getFunctionType(), FlatSymbolRefAttr::get(paramName)
  );
  if (candidates.size() >= 2) {
    diag << ": " << candidates.front();
    for (Type candidate : llvm::drop_begin(candidates)) {
      diag << " vs " << candidate;
    }
  }
  return diag;
}

/// Return one inferred RHS template argument from a call-signature unification.
template <typename CallableOp, typename TargetOp>
static FailureOr<Attribute> getInferredRhsTemplateArg(
    CallableOp callableOp, TargetOp targetOp, const UnificationMap &unifyResult,
    StringAttr paramName, StringRef argKind
) {
  auto inferredIt = unifyResult.find({FlatSymbolRefAttr::get(paramName), Side::RHS});
  if (inferredIt == unifyResult.end()) {
    return callableOp.emitError() << "could not infer " << argKind
                                  << " template argument for parameter @" << paramName.getValue()
                                  << " from callee signature";
  }
  if (!inferredIt->second) {
    return emitConflictingInferredTypes(callableOp, targetOp, paramName);
  }
  return inferredIt->second;
}

/// Infer the call-site template argument list from call/callee type unification.
template <typename CallableOp, typename TargetOp>
static FailureOr<ArrayAttr> getCallSignatureTemplateParams(
    CallableOp callableOp, const TemplateInferenceInfo &info, TargetOp targetOp
) {
  FailureOr<UnificationMap> unifyResult = callableOp.unifyTypeSignature(targetOp.getFunctionType());
  if (failed(unifyResult)) {
    return callableOp.emitError()
           << "could not infer omitted template arguments from callee signature";
  }

  SmallVector<Attribute> params;
  params.reserve(info.oldParamOrder.size());
  for (StringAttr paramName : info.oldParamOrder) {
    auto replacementIt = info.replacements.find(paramName);
    if (replacementIt != info.replacements.end()) {
      params.push_back(TypeAttr::get(replacementIt->second.type));
      continue;
    }
    FailureOr<Attribute> inferredAttr =
        getInferredRhsTemplateArg(callableOp, targetOp, *unifyResult, paramName, "omitted");
    if (failed(inferredAttr)) {
      return failure();
    }
    params.push_back(*inferredAttr);
  }
  return ArrayAttr::get(callableOp.getContext(), params);
}

/// Return true when a call's explicit template arguments contain `?` for a
/// residual type-variable parameter mentioned by `func`.
template <typename TargetOp>
static bool hasResidualFunctionTvarWildcard(
    ArrayAttr callParams, const TemplateInferenceInfo &info, TargetOp targetOp
) {
  if (!callParams || callParams.size() != info.oldParamOrder.size()) {
    return false;
  }
  for (const auto &entry : info.typeVarParams) {
    StringAttr paramName = entry.first;
    if (!targetMentionsParam(targetOp, paramName)) {
      continue;
    }
    std::optional<unsigned> index = getParamIndex(info.oldParamOrder, paramName);
    assert(index && "eligible type-variable parameter must appear in parameter order");
    if (isWildcardTemplateArg(callParams[*index])) {
      return true;
    }
  }
  return false;
}

/// Materialize explicit `?` type arguments from the call signature before
/// choosing a function-local clone.
template <typename CallableOp, typename TargetOp>
static FailureOr<ArrayAttr> materializeResidualFunctionTvarWildcards(
    CallableOp callableOp, ArrayAttr callParams, const TemplateInferenceInfo &info,
    TargetOp targetOp
) {
  FailureOr<UnificationMap> unifyResult = callableOp.unifyTypeSignature(targetOp.getFunctionType());
  if (failed(unifyResult)) {
    return callableOp.emitError()
           << "could not infer wildcard template arguments from callee signature";
  }

  SmallVector<Attribute> params(callParams.begin(), callParams.end());
  for (const auto &entry : info.typeVarParams) {
    StringAttr paramName = entry.first;
    if (!targetMentionsParam(targetOp, paramName)) {
      continue;
    }
    std::optional<unsigned> index = getParamIndex(info.oldParamOrder, paramName);
    assert(index && "eligible type-variable parameter must appear in parameter order");
    if (!isWildcardTemplateArg(params[*index])) {
      continue;
    }

    FailureOr<Attribute> inferredAttr =
        getInferredRhsTemplateArg(callableOp, targetOp, *unifyResult, paramName, "wildcard");
    if (failed(inferredAttr)) {
      return failure();
    }
    params[*index] = *inferredAttr;
  }
  return ArrayAttr::get(callableOp.getContext(), params);
}

/// Return true when `func` still needs per-call cloning after template-wide rewrites.
///
/// Parameters in `info.replacements` are safe to rewrite in the template itself.
/// A clone is only necessary when the function mentions an eligible type
/// variable that was not proven for every relevant entry point.
template <typename TargetOp>
static bool hasResidualFunctionTvar(const TemplateInferenceInfo &info, TargetOp targetOp) {
  return llvm::any_of(info.typeVarParams, [&](const auto &entry) {
    return !info.replacements.contains(entry.first) && targetMentionsParam(targetOp, entry.first);
  });
}

/// Common call/include specialization inputs after template arguments are materialized.
struct CallableSpecializationInputs {
  ArrayAttr params;
  bool explicitParams = false;
  bool paramsChanged = false;
  DenseMap<StringAttr, Type> replacements;
};

/// Materialize call/include template arguments and compute concrete tvar replacements.
template <typename CallableOp, typename TargetOp>
static LogicalResult prepareCallableSpecialization(
    CallableOp callableOp, const TemplateInferenceInfo &info, TargetOp targetOp,
    CallableSpecializationInputs &inputs, bool &shouldSpecialize
) {
  shouldSpecialize = false;
  inputs.params = {};
  inputs.explicitParams = false;
  inputs.paramsChanged = false;
  inputs.replacements.clear();
  inputs.params = callableOp.getTemplateParamsAttr();
  inputs.explicitParams = !isNullOrEmpty(inputs.params);
  if (isNullOrEmpty(inputs.params)) {
    FailureOr<ArrayAttr> inferredParams =
        getCallSignatureTemplateParams(callableOp, info, targetOp);
    if (failed(inferredParams)) {
      return failure();
    }
    inputs.params = *inferredParams;
    inputs.paramsChanged = true;
  } else {
    inputs.params = expandCurrentTemplateParamsToOriginalOrder(inputs.params, info);
    if (hasResidualFunctionTvarWildcard(inputs.params, info, targetOp)) {
      FailureOr<ArrayAttr> materializedParams =
          materializeResidualFunctionTvarWildcards(callableOp, inputs.params, info, targetOp);
      if (failed(materializedParams)) {
        return failure();
      }
      inputs.params = *materializedParams;
      inputs.paramsChanged = true;
    }
  }

  inputs.replacements = getConcreteCallSiteReplacements(inputs.params, info, targetOp);
  shouldSpecialize = !inputs.replacements.empty();
  return success();
}

/// Build a lossless key for a sibling-template specialized function clone.
///
/// The emitted clone symbol uses `BuildShortTypeString`, which intentionally
/// shortens some structural types. This cache key keeps the same parameter
/// positions but prints full replacement types so two distinct instantiations
/// that prefer the same clone name still get distinct clones.
static std::string buildTemplateLocalFunctionCloneCacheKey(
    StringRef functionName, ArrayRef<StringAttr> oldParamOrder,
    const DenseMap<StringAttr, Type> &replacements
) {
  std::string key;
  llvm::raw_string_ostream os(key);
  os << functionName.size() << ':' << functionName;
  for (StringAttr paramName : oldParamOrder) {
    os << '|';
    os << paramName.getValue().size() << ':' << paramName.getValue() << '=';
    auto replacementIt = replacements.find(paramName);
    if (replacementIt == replacements.end()) {
      os << '_';
      continue;
    }

    std::string typeText;
    llvm::raw_string_ostream typeOs(typeText);
    replacementIt->second.print(typeOs);
    os << typeText.size() << ':' << typeText;
  }
  return key;
}

/// Return the callee path for a clone nested in a sibling specialization template.
static SymbolRefAttr getSpecializedFunctionCloneCallee(
    SymbolRefAttr originalCallee, StringAttr templateName, StringAttr cloneName
) {
  SmallVector<FlatSymbolRefAttr> pieces = getPieces(originalCallee);
  assert(pieces.size() >= 2 && "callee must include at least template and function names");
  pieces.pop_back();
  pieces.pop_back();
  pieces.push_back(FlatSymbolRefAttr::get(templateName));
  pieces.push_back(FlatSymbolRefAttr::get(cloneName));
  return asSymbolRefAttr(pieces);
}

/// Create or reuse a sibling-template clone for a concrete callable-local tvar instantiation.
template <typename CallableOp, typename ConfigureCloneFn>
static FailureOr<SymbolRefAttr> getOrCreateSpecializedCallableClone(
    TemplateOp templateOp, CallableOp callable, SymbolRefAttr originalCallee,
    const TemplateInferenceInfo &info, const DenseMap<StringAttr, Type> &replacements,
    SymbolTableCollection &tables, llvm::StringMap<SymbolRefAttr> &cloneCallees,
    llvm::StringMap<StringAttr> &templateClones, InstantiationLayout &layout, ArrayAttr callParams,
    ConfigureCloneFn configureClone
) {
  DenseMap<Attribute, Attribute> paramNameToConcrete = buildParamNameToConcrete(replacements);
  FailureOr<TemplateOp> newTemplate = getOrCreateSpecializedTemplateClone(
      templateOp, info.oldParamOrder, paramNameToConcrete, callParams, tables, templateClones,
      layout
  );
  if (failed(newTemplate)) {
    return failure();
  }

  std::string cacheKey = buildTemplateLocalFunctionCloneCacheKey(
      callable.getSymName(), info.oldParamOrder, replacements
  );
  auto cachedCallee = cloneCallees.find(cacheKey);
  if (cachedCallee != cloneCallees.end()) {
    return cachedCallee->second;
  }

  SymbolTable &templateSymbols = tables.getSymbolTable(*newTemplate);

  auto clone = llvm::cast<CallableOp>(callable.getOperation()->clone());
  configureClone(clone);
  templateSymbols.insert(clone);

  TypeVarReplacementConverter converter(
      templateOp.getContext(), info.templatePath, info.oldParamOrder, replacements,
      /*trimResolvedParams=*/false
  );
  if (failed(convertTemplateExprTypesIn(*newTemplate, converter))) {
    clone.erase();
    return failure();
  }
  if (failed(convertOperationTypesIn(clone.getOperation(), converter))) {
    clone.erase();
    return failure();
  }
  removeIdentityCasts(clone.getOperation());
  SymbolRefAttr cloneCallee = getSpecializedFunctionCloneCallee(
      originalCallee, newTemplate->getSymNameAttr(), clone.getSymNameAttr()
  );
  cloneCallees.try_emplace(cacheKey, cloneCallee);
  return cloneCallee;
}

/// Create or reuse a sibling-template clone for a concrete function-local tvar instantiation.
static FailureOr<SymbolRefAttr> getOrCreateSpecializedFunctionClone(
    TemplateOp templateOp, FuncDefOp func, SymbolRefAttr originalCallee,
    const TemplateInferenceInfo &info, const DenseMap<StringAttr, Type> &replacements,
    SymbolTableCollection &tables, llvm::StringMap<SymbolRefAttr> &cloneCallees,
    llvm::StringMap<StringAttr> &templateClones, InstantiationLayout &layout, ArrayAttr callParams
) {
  return getOrCreateSpecializedCallableClone(
      templateOp, func, originalCallee, info, replacements, tables, cloneCallees, templateClones,
      layout, callParams, [](FuncDefOp) {}
  );
}

/// Create or reuse a sibling-template clone for a concrete contract-local tvar instantiation.
static FailureOr<SymbolRefAttr> getOrCreateSpecializedContractClone(
    TemplateOp templateOp, verif::ContractOp contract, SymbolRefAttr originalCallee,
    SymbolRefAttr specializedTarget, const TemplateInferenceInfo &info,
    const DenseMap<StringAttr, Type> &replacements, SymbolTableCollection &tables,
    llvm::StringMap<SymbolRefAttr> &cloneCallees, llvm::StringMap<StringAttr> &templateClones,
    InstantiationLayout &layout, ArrayAttr callParams
) {
  return getOrCreateSpecializedCallableClone(
      templateOp, contract, originalCallee, info, replacements, tables, cloneCallees,
      templateClones, layout, callParams,
      [specializedTarget](verif::ContractOp clone) { clone.setTargetAttr(specializedTarget); }
  );
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

/// Verify that a wildcard call/include still matches its target after type-variable inference.
template <typename CallableOp>
static LogicalResult
validateWildcardCallableSignature(CallableOp callableOp, FunctionType targetTy) {
  if (succeeded(callableOp.unifyTypeSignature(targetTy))) {
    return success();
  }
  return callableOp.emitError() << "call signature " << callableOp.getTypeSignature()
                                << " does not match inferred callee signature " << targetTy;
}

/// Update call result types when same-template template symbol arguments were resolved.
static void updateCallableResultTypesIfNeeded(
    CallOp callOp, const TypeVarReplacementConverter &converter, bool &modified
) {
  for (Value result : callOp.getResults()) {
    Type newTy = converter.convertType(result.getType());
    if (newTy != result.getType()) {
      result.setType(newTy);
      modified = true;
    }
  }
}

/// Includes have no SSA results to update.
static void
updateCallableResultTypesIfNeeded(verif::IncludeOp, const TypeVarReplacementConverter &, bool &) {}

/// Update one callable op family that references rewritten templates.
template <typename CallableOp>
static FailureOr<bool> updateCallableTemplateParamsFor(
    ModuleOp module, DenseMap<Operation *, const TypeVarReplacementConverter *> &converters,
    SymbolTableCollection &tables
) {
  bool modified = false;
  bool failedConversion = false;
  module.walk([&](CallableOp callableOp) {
    if (failedConversion) {
      return;
    }
    auto target = callableOp.getCalleeTarget(tables);
    if (failed(target)) {
      return;
    }
    auto targetOp = target->get();
    TemplateOp parentTemplate = getParentOfType<TemplateOp>(targetOp.getOperation());
    if (!parentTemplate) {
      return;
    }
    const TypeVarReplacementConverter *converter = converters.lookup(parentTemplate.getOperation());
    if (!converter) {
      return;
    }

    TemplateOp callableTemplate = getParentOfType<TemplateOp>(callableOp.getOperation());
    bool resolveTemplateSymbolArgs =
        callableTemplate && callableTemplate.getOperation() == parentTemplate.getOperation();
    ArrayAttr oldParams = callableOp.getTemplateParamsAttr();
    FailureOr<ArrayAttr> newParams =
        converter->convertTemplateParams(oldParams, callableOp, resolveTemplateSymbolArgs);
    if (failed(newParams)) {
      failedConversion = true;
      return;
    }
    if (converter->hasWildcardForRemovedParam(oldParams) &&
        failed(validateWildcardCallableSignature(callableOp, targetOp.getFunctionType()))) {
      failedConversion = true;
      return;
    }
    if (oldParams != *newParams) {
      callableOp.setTemplateParamsAttr(*newParams);
      modified = true;
    }

    if (resolveTemplateSymbolArgs) {
      updateCallableResultTypesIfNeeded(callableOp, *converter, modified);
    }
  });
  if (failedConversion) {
    return failure();
  }
  return modified;
}

/// Update references to callable symbols inside rewritten templates.
///
/// Call operations keep their explicit template argument list, if any. When a
/// callee template loses resolved `poly.param`s, call-site argument lists must
/// drop the same positions. Result types are also converted because call results
/// may mention an inferred return type variable. Contract includes use the same
/// template parameter representation and need the same positional trimming.
static FailureOr<bool> updateCallableTemplateParams(
    ModuleOp module, DenseMap<Operation *, const TypeVarReplacementConverter *> &converters
) {
  SymbolTableCollection tables;
  FailureOr<bool> callsModified =
      updateCallableTemplateParamsFor<CallOp>(module, converters, tables);
  if (failed(callsModified)) {
    return failure();
  }
  FailureOr<bool> includesModified =
      updateCallableTemplateParamsFor<verif::IncludeOp>(module, converters, tables);
  if (failed(includesModified)) {
    return failure();
  }
  return *callsModified || *includesModified;
}

/// Converts struct type sites that reference templates whose parameter list is being trimmed.
///
/// Concrete struct instantiation handles fully concrete uses by cloning. Sites that still mention
/// symbols from another template need a checked pass before erased parameters disappear; otherwise
/// a type such as `!struct.type<@TBox::@Box<[@U]>>` can be left with one argument for a
/// zero-parameter owner template.
class ReferencedStructTemplateParamConverter {
  /// Context used to allocate converted aggregate types and attributes.
  MLIRContext *ctx_;
  /// Root operation used for resolving struct definitions.
  ModuleOp module_;
  /// Symbol tables reused for lookup.
  SymbolTableCollection &tables_;
  /// Converters for templates whose resolved parameters will be erased.
  DenseMap<Operation *, const TypeVarReplacementConverter *> &converters_;
  /// Operation used as the diagnostic anchor while converting one op.
  Operation *diagnosticOp_ = nullptr;
  /// Whether conversion of the current operation failed.
  bool hasFailure = false;

public:
  ReferencedStructTemplateParamConverter(
      MLIRContext *ctx, ModuleOp module, SymbolTableCollection &tables,
      DenseMap<Operation *, const TypeVarReplacementConverter *> &converters
  )
      : ctx_(ctx), module_(module), tables_(tables), converters_(converters) {}

  /// Start converting a new operation and use it for diagnostics.
  void startOperation(Operation *op) {
    diagnosticOp_ = op;
    hasFailure = false;
  }

  /// Return whether conversion of the current operation failed.
  bool hadFailure() const { return hasFailure; }

  /// Convert a type by recursively updating struct template parameter lists.
  Type convertType(Type ty) {
    if (!ty || hasFailure) {
      return ty;
    }
    if (auto arrTy = llvm::dyn_cast<ArrayType>(ty)) {
      return convertArrayElementType(arrTy, [this](Type elemTy) { return convertType(elemTy); });
    }
    if (auto structTy = llvm::dyn_cast<StructType>(ty)) {
      return convertStructType(structTy);
    }
    if (auto podTy = llvm::dyn_cast<PodType>(ty)) {
      return convertPodType(podTy, ctx_, *this);
    }
    if (auto funcTy = llvm::dyn_cast<FunctionType>(ty)) {
      return convertFunctionType(funcTy, *this);
    }
    return ty;
  }

  /// Convert an attribute that can contain struct type references.
  Attribute convertAttr(Attribute attr) {
    if (hasFailure) {
      return attr;
    }
    return convertTypeOrArrayAttr(attr, ctx_, [this](Type ty) {
      return convertType(ty);
    }, [this](Attribute nested) { return convertAttr(nested); });
  }

private:
  /// Convert a struct type and checked-trim params when its owner template is being rewritten.
  StructType convertStructType(StructType structTy) {
    ArrayAttr params = structTy.getParams();
    if (!params) {
      return structTy;
    }

    SmallVector<Attribute> newParams;
    bool changed = false;
    for (Attribute attr : params.getValue()) {
      Attribute newAttr = convertAttr(attr);
      newParams.push_back(newAttr);
      changed |= newAttr != attr;
      if (hasFailure) {
        return structTy;
      }
    }
    StructType convertedTy =
        changed ? getStructTypeWithParams(structTy.getNameRef(), ctx_, newParams) : structTy;

    FailureOr<SymbolLookupResult<StructDefOp>> lookup =
        convertedTy.getDefinition(tables_, module_, /*reportMissing=*/false);
    if (failed(lookup)) {
      return convertedTy;
    }
    TemplateOp parentTemplate = getParentOfType<TemplateOp>(lookup->get().getOperation());
    if (!parentTemplate) {
      return convertedTy;
    }
    const TypeVarReplacementConverter *converter =
        converters_.lookup(parentTemplate.getOperation());
    if (!converter) {
      return convertedTy;
    }

    TemplateOp useTemplate = getParentOfType<TemplateOp>(diagnosticOp_);
    bool resolveTemplateSymbolArgs =
        useTemplate && useTemplate.getOperation() == parentTemplate.getOperation();
    FailureOr<ArrayAttr> trimmedParams = converter->convertTemplateParams(
        convertedTy.getParams(), diagnosticOp_, resolveTemplateSymbolArgs
    );
    if (failed(trimmedParams)) {
      hasFailure = true;
      return convertedTy;
    }
    if (convertedTy.getParams() == *trimmedParams) {
      return convertedTy;
    }
    return getStructTypeWithParams(convertedTy.getNameRef(), *trimmedParams);
  }
};

/// Rewrite or reject struct type sites that reference rewritten templates.
static FailureOr<bool> updateStructTemplateParams(
    ModuleOp module, DenseMap<Operation *, const TypeVarReplacementConverter *> &converters
) {
  SymbolTableCollection tables;
  ReferencedStructTemplateParamConverter converter(module.getContext(), module, tables, converters);
  return convertOperationTypesInAndTrack(module.getOperation(), converter);
}

/// Instantiate concrete parameterized struct types exposed by type-variable inference.
static LogicalResult instantiateConcreteStructUses(ModuleOp module) {
  SymbolTableCollection tables;
  ConcreteStructInstantiationConverter converter(module.getContext(), module, tables);
  return convertOperationTypesIn(module.getOperation(), converter);
}

/// Return the template that owns the function or struct targeted by `contract`.
static TemplateOp
getContractTargetTemplate(verif::ContractOp contract, SymbolTableCollection &tables) {
  if (FailureOr<SymbolLookupResult<FuncDefOp>> funcTarget = contract.getFuncTarget(tables);
      succeeded(funcTarget)) {
    return getParentOfType<TemplateOp>(funcTarget->get().getOperation());
  }
  if (FailureOr<SymbolLookupResult<StructDefOp>> structTarget = contract.getStructTarget(tables);
      succeeded(structTarget)) {
    return getParentOfType<TemplateOp>(structTarget->get().getOperation());
  }
  return {};
}

/// Rewrite external contracts that target callables or structs inside rewritten templates.
///
/// A contract can be physically outside the `poly.template` whose parameters it
/// references through its target signature. Template-body conversion does not
/// visit those contracts, so apply the target template's converter here before
/// resolved parameters are erased.
static LogicalResult updateExternalContractTemplateParams(
    ModuleOp module, DenseMap<Operation *, const TypeVarReplacementConverter *> &converters
) {
  bool failedConversion = false;
  SymbolTableCollection tables;
  module.walk([&](verif::ContractOp contract) {
    if (failedConversion) {
      return;
    }
    TemplateOp targetTemplate = getContractTargetTemplate(contract, tables);
    if (!targetTemplate) {
      return;
    }
    if (getParentOfType<TemplateOp>(contract.getOperation()) == targetTemplate) {
      return;
    }
    const TypeVarReplacementConverter *converter = converters.lookup(targetTemplate.getOperation());
    if (!converter) {
      return;
    }

    WalkResult validationResult = contract.walk([converter](Operation *op) {
      return failed(converter->validateOperation(op)) ? WalkResult::interrupt()
                                                      : WalkResult::advance();
    });
    if (validationResult.wasInterrupted() ||
        failed(convertOperationTypesIn(contract.getOperation(), *converter))) {
      failedConversion = true;
      return;
    }
    removeIdentityCasts(contract.getOperation());
  });
  return failure(failedConversion);
}

/// Specialize calls/includes to sibling-template callables with concrete tvar instantiations.
///
/// This intentionally handles only fully-concrete explicit call-site type
/// arguments for callables directly nested in a template. More general partial
/// instantiations should follow the flattening pass' template-cloning machinery,
/// but this covers the tvar-only cases without invoking flattening's
/// struct/array transformations. Function-local inference facts are used to
/// reject call-site instantiations that disagree with a proof in the body.
static LogicalResult specializeFunctionLocalCallables(
    ModuleOp module, DenseMap<Operation *, const TemplateInferenceInfo *> &infoByTemplate,
    SpecializedCallableCloneCache &functionCloneCache,
    SpecializedCallableCloneCache &contractCloneCache,
    SpecializedTemplateCloneCache &templateCloneCache
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
    if (!hasResidualFunctionTvar(*info, targetFunc)) {
      return;
    }

    CallableSpecializationInputs inputs;
    bool shouldSpecialize = false;
    if (failed(
            prepareCallableSpecialization(callOp, *info, targetFunc, inputs, shouldSpecialize)
        )) {
      failedClone = true;
      return;
    }
    if (!shouldSpecialize) {
      return;
    }
    DenseMap<StringAttr, Type> inferredReplacements =
        getFunctionProofReplacements(*info, targetFunc);
    if (failed(diagnoseCallParamsMismatch(
            callOp.getOperation(), inputs.params, info->oldParamOrder, inferredReplacements,
            /*explicitCallParams=*/inputs.explicitParams
        ))) {
      failedClone = true;
      return;
    }

    InstantiationLayout layout;
    FailureOr<SymbolRefAttr> cloneCallee = getOrCreateSpecializedFunctionClone(
        parentTemplate, targetFunc, callOp.getCalleeAttr(), *info, inputs.replacements, tables,
        functionCloneCache[parentTemplate.getOperation()],
        templateCloneCache[parentTemplate.getOperation()], layout, inputs.params
    );
    if (failed(cloneCallee)) {
      failedClone = true;
      return;
    }

    callOp.setCalleeAttr(*cloneCallee);
    callOp.setTemplateParamsAttr(layout.rewrittenCallParams);
  });
  module.walk([&](verif::IncludeOp includeOp) {
    if (failedClone) {
      return;
    }
    FailureOr<SymbolLookupResult<verif::ContractOp>> target = includeOp.getCalleeTarget(tables);
    if (failed(target)) {
      return;
    }
    verif::ContractOp targetContract = target->get();
    auto parentTemplate = llvm::dyn_cast_or_null<TemplateOp>(targetContract->getParentOp());
    if (!parentTemplate) {
      return;
    }
    const TemplateInferenceInfo *info = infoByTemplate.lookup(parentTemplate.getOperation());
    if (!info) {
      return;
    }
    if (!hasResidualFunctionTvar(*info, targetContract)) {
      return;
    }

    FailureOr<SymbolLookupResult<FuncDefOp>> targetFuncResult =
        targetContract.getFuncTarget(tables);
    if (failed(targetFuncResult)) {
      return;
    }
    FuncDefOp targetFunc = targetFuncResult->get();
    if (getParentOfType<TemplateOp>(targetFunc.getOperation()) != parentTemplate) {
      return;
    }

    CallableSpecializationInputs inputs;
    bool shouldSpecialize = false;
    if (failed(prepareCallableSpecialization(
            includeOp, *info, targetContract, inputs, shouldSpecialize
        ))) {
      failedClone = true;
      return;
    }
    if (!shouldSpecialize) {
      return;
    }
    DenseMap<StringAttr, Type> inferredReplacements =
        getFunctionProofReplacements(*info, targetFunc);
    if (failed(diagnoseCallParamsMismatch(
            includeOp.getOperation(), inputs.params, info->oldParamOrder, inferredReplacements,
            /*explicitCallParams=*/inputs.explicitParams
        ))) {
      failedClone = true;
      return;
    }

    InstantiationLayout targetLayout;
    FailureOr<SymbolRefAttr> specializedTarget = getOrCreateSpecializedFunctionClone(
        parentTemplate, targetFunc, targetContract.getTargetAttr(), *info, inputs.replacements,
        tables, functionCloneCache[parentTemplate.getOperation()],
        templateCloneCache[parentTemplate.getOperation()], targetLayout, inputs.params
    );
    if (failed(specializedTarget)) {
      failedClone = true;
      return;
    }
    InstantiationLayout contractLayout;
    FailureOr<SymbolRefAttr> cloneCallee = getOrCreateSpecializedContractClone(
        parentTemplate, targetContract, includeOp.getCalleeAttr(), *specializedTarget, *info,
        inputs.replacements, tables, contractCloneCache[parentTemplate.getOperation()],
        templateCloneCache[parentTemplate.getOperation()], contractLayout, inputs.params
    );
    if (failed(cloneCallee)) {
      failedClone = true;
      return;
    }

    includeOp.setCalleeAttr(*cloneCallee);
    includeOp.setTemplateParamsAttr(contractLayout.rewrittenCallParams);
  });
  return failure(failedClone);
}

/// Return whether two inference maps contain the same concrete replacement types.
static bool sameReplacementTypes(
    const DenseMap<StringAttr, InferredType> &lhs, const DenseMap<StringAttr, InferredType> &rhs
) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (const auto &entry : lhs) {
    auto rhsIt = rhs.find(entry.first);
    if (rhsIt == rhs.end() || rhsIt->second.type != entry.second.type) {
      return false;
    }
  }
  return true;
}

/// Collect proof facts from a contract into the function it verifies.
///
/// A contract must have the same signature as its target, so a contract-local
/// proof about a target type variable is part of that target's compatibility
/// story. Merging the facts catches incompatible contract casts before the
/// target function is rewritten.
static LogicalResult collectContractTargetFunctionInferences(
    TemplateInferenceInfo &info, verif::ContractOp contract, ModuleOp module,
    DenseMap<Operation *, TemplateInferenceInfo *> *infoByTemplate, SymbolTableCollection &tables,
    bool &changed
) {
  FailureOr<SymbolLookupResult<FuncDefOp>> target = contract.getFuncTarget(tables);
  if (failed(target)) {
    return success();
  }
  FuncDefOp targetFunc = target->get();
  if (getParentOfType<TemplateOp>(targetFunc.getOperation()) != info.templateOp) {
    return success();
  }

  DenseMap<StringAttr, InferredType> funcReplacements;
  auto funcIt = info.functionReplacements.find(targetFunc.getOperation());
  if (funcIt != info.functionReplacements.end()) {
    funcReplacements = funcIt->second;
  }
  DenseMap<StringAttr, InferredType> oldFuncReplacements = funcReplacements;

  TypeVarInferenceCollector collector(info, funcReplacements);
  if (failed(collector.collect(contract.getOperation()))) {
    return failure();
  }
  if (infoByTemplate && failed(collector.collectStructTemplateParamInferences(
                            contract.getOperation(), module, *infoByTemplate, tables
                        ))) {
    return failure();
  }

  if (!sameReplacementTypes(oldFuncReplacements, funcReplacements)) {
    changed = true;
  }
  if (funcReplacements.empty()) {
    info.functionReplacements.erase(targetFunc.getOperation());
  } else {
    info.functionReplacements[targetFunc.getOperation()] = std::move(funcReplacements);
  }
  return success();
}

/// Collect proof facts from a contract into the struct it verifies.
///
/// A struct-target contract has the same self signature as the target struct's
/// product/constrain contract type. Proofs in the contract body therefore
/// constrain the target template directly, even when the contract is physically
/// outside that template.
static LogicalResult collectContractTargetStructInferences(
    TemplateInferenceInfo &info, verif::ContractOp contract, ModuleOp module,
    DenseMap<Operation *, TemplateInferenceInfo *> *infoByTemplate, SymbolTableCollection &tables,
    bool &changed
) {
  FailureOr<SymbolLookupResult<StructDefOp>> target = contract.getStructTarget(tables);
  if (failed(target)) {
    return success();
  }
  StructDefOp targetStruct = target->get();
  if (getParentOfType<TemplateOp>(targetStruct.getOperation()) != info.templateOp) {
    return success();
  }

  DenseMap<StringAttr, InferredType> oldTemplateScopeReplacements = info.templateScopeReplacements;

  TypeVarInferenceCollector collector(info, info.templateScopeReplacements);
  FunctionType contractTy = contract.getFunctionType();
  if (contractTy.getNumInputs() > 0) {
    StructType targetSelfTy = targetStruct.getType();
    if (auto contractSelfTy = llvm::dyn_cast<StructType>(contractTy.getInput(0));
        contractSelfTy && contractSelfTy.getNameRef() == targetSelfTy.getNameRef() &&
        failed(collector.collectTypeInferences(contractSelfTy, targetSelfTy, contract.getLoc()))) {
      return failure();
    }
  }
  if (failed(collector.collect(contract.getOperation()))) {
    return failure();
  }
  if (infoByTemplate && failed(collector.collectStructTemplateParamInferences(
                            contract.getOperation(), module, *infoByTemplate, tables
                        ))) {
    return failure();
  }

  if (!sameReplacementTypes(oldTemplateScopeReplacements, info.templateScopeReplacements)) {
    changed = true;
  }
  return success();
}

/// Recompute the template-wide replacements from the per-function proofs.
///
/// Template-scope proofs from `poly.expr` bodies constrain every instantiation
/// of the template, so they are always template-wide. A replacement proven only
/// by function bodies becomes template-wide only when every function that
/// mentions the parameter proves the same concrete type and every non-function
/// type-bearing operation is either covered by a structural function proof or
/// absent. Uncovered non-function uses cannot be cloned per call, so they force
/// the replacement to remain function-local.
static LogicalResult recomputeTemplateWideReplacements(TemplateInferenceInfo &info) {
  DenseMap<StringAttr, unsigned> mentionCounts;
  DenseMap<StringAttr, unsigned> proofCounts;
  DenseMap<StringAttr, InferredType> commonReplacements;
  DenseSet<StringAttr> incompatibleReplacements;

  for (FuncDefOp func : walkCollect<FuncDefOp>(info.templateOp)) {
    for (const auto &entry : info.typeVarParams) {
      if (funcMentionsParam(func, entry.first)) {
        ++mentionCounts[entry.first];
      }
    }

    auto funcIt = info.functionReplacements.find(func.getOperation());
    if (funcIt == info.functionReplacements.end()) {
      continue;
    }
    for (const auto &entry : funcIt->second) {
      ++proofCounts[entry.first];
      auto commonIt = commonReplacements.find(entry.first);
      if (commonIt == commonReplacements.end()) {
        commonReplacements.try_emplace(entry.first, entry.second);
      } else if (commonIt->second.type != entry.second.type) {
        incompatibleReplacements.insert(entry.first);
      }
    }
  }

  for (TemplateExprOp expr : walkCollect<TemplateExprOp>(info.templateOp)) {
    for (const auto &entry : info.typeVarParams) {
      if (exprMentionsParam(expr, entry.first)) {
        ++mentionCounts[entry.first];
      }
    }
  }

  info.replacements = info.templateScopeReplacements;
  for (const auto &entry : info.templateScopeReplacements) {
    auto commonIt = commonReplacements.find(entry.first);
    if (commonIt != commonReplacements.end() && commonIt->second.type != entry.second.type) {
      InFlightDiagnostic diag = emitError(entry.second.loc)
                                << "conflicting inferred type for template parameter @"
                                << entry.first.getValue() << ": " << commonIt->second.type << " vs "
                                << entry.second.type;
      diag.attachNote(commonIt->second.loc) << "previous function-local inference here";
      return diag;
    }
  }

  for (const auto &entry : commonReplacements) {
    StringAttr paramName = entry.first;
    if (info.replacements.contains(paramName)) {
      continue;
    }
    if (!hasUncoveredNonFunctionMention(
            info.templateOp, paramName, entry.second.type, info.functionReplacements
        ) &&
        !incompatibleReplacements.contains(paramName) &&
        proofCounts.lookup(paramName) == mentionCounts.lookup(paramName)) {
      info.replacements.try_emplace(paramName, entry.second);
    }
  }
  return success();
}

/// Infer type-variable replacements from struct type arguments whose target
/// template parameters are already resolved.
static LogicalResult inferStructTemplateParamUses(
    ModuleOp module, MutableArrayRef<TemplateInferenceInfo> templateInfos,
    DenseMap<Operation *, TemplateInferenceInfo *> &infoByTemplate
) {
  bool changed = false;
  do {
    changed = false;
    SymbolTableCollection tables;
    for (TemplateInferenceInfo &info : templateInfos) {
      for (FuncDefOp func : walkCollect<FuncDefOp>(info.templateOp)) {
        DenseMap<StringAttr, InferredType> funcReplacements;
        auto funcIt = info.functionReplacements.find(func.getOperation());
        if (funcIt != info.functionReplacements.end()) {
          funcReplacements = funcIt->second;
        }
        DenseMap<StringAttr, InferredType> oldFuncReplacements = funcReplacements;

        TypeVarInferenceCollector collector(info, funcReplacements);
        if (failed(collector.collect(func.getOperation())) ||
            failed(collector.collectStructTemplateParamInferences(
                func.getOperation(), module, infoByTemplate, tables
            ))) {
          return failure();
        }

        if (!sameReplacementTypes(oldFuncReplacements, funcReplacements)) {
          changed = true;
        }
        if (funcReplacements.empty()) {
          info.functionReplacements.erase(func.getOperation());
        } else {
          info.functionReplacements[func.getOperation()] = std::move(funcReplacements);
        }
      }

      for (verif::ContractOp contract : walkCollect<verif::ContractOp>(info.templateOp)) {
        if (failed(collectContractTargetFunctionInferences(
                info, contract, module, &infoByTemplate, tables, changed
            )) ||
            failed(collectContractTargetStructInferences(
                info, contract, module, &infoByTemplate, tables, changed
            ))) {
          return failure();
        }
      }

      for (TemplateExprOp expr : walkCollect<TemplateExprOp>(info.templateOp)) {
        DenseMap<StringAttr, InferredType> oldTemplateScopeReplacements =
            info.templateScopeReplacements;

        TypeVarInferenceCollector collector(info, info.templateScopeReplacements);
        if (failed(collector.collect(expr.getOperation())) ||
            failed(collector.collectStructTemplateParamInferences(
                expr.getOperation(), module, infoByTemplate, tables
            ))) {
          return failure();
        }

        if (!sameReplacementTypes(oldTemplateScopeReplacements, info.templateScopeReplacements)) {
          changed = true;
        }
      }

      DenseMap<StringAttr, InferredType> oldReplacements = info.replacements;
      if (failed(recomputeTemplateWideReplacements(info))) {
        return failure();
      }
      if (!sameReplacementTypes(oldReplacements, info.replacements)) {
        changed = true;
      }
    }
  } while (changed);
  return success();
}

/// Collect proof facts from contracts that target a rewritten template but are
/// physically outside it.
static LogicalResult collectExternalContractTargetInferences(
    ModuleOp module, MutableArrayRef<TemplateInferenceInfo> templateInfos,
    DenseMap<Operation *, TemplateInferenceInfo *> &infoByTemplate
) {
  bool changed = false;
  do {
    changed = false;
    bool failedCollection = false;
    SymbolTableCollection tables;
    module.walk([&](verif::ContractOp contract) {
      TemplateOp targetTemplate = getContractTargetTemplate(contract, tables);
      if (!targetTemplate) {
        return;
      }
      if (getParentOfType<TemplateOp>(contract.getOperation()) == targetTemplate) {
        return;
      }
      TemplateInferenceInfo *info = infoByTemplate.lookup(targetTemplate.getOperation());
      if (!info) {
        return;
      }
      if (failed(collectContractTargetFunctionInferences(
              *info, contract, module, &infoByTemplate, tables, changed
          )) ||
          failed(collectContractTargetStructInferences(
              *info, contract, module, &infoByTemplate, tables, changed
          ))) {
        failedCollection = true;
      }
    });
    if (failedCollection) {
      return failure();
    }

    for (TemplateInferenceInfo &info : templateInfos) {
      DenseMap<StringAttr, InferredType> oldReplacements = info.replacements;
      if (failed(recomputeTemplateWideReplacements(info))) {
        return failure();
      }
      if (!sameReplacementTypes(oldReplacements, info.replacements)) {
        changed = true;
      }
    }
  } while (changed);
  return success();
}

/// Build all analysis state needed to rewrite one template.
///
/// The function records the original template parameter order before any
/// rewrites, identifies eligible type-variable parameters, and collects concrete
/// replacements. Function-local proofs are kept local unless every relevant
/// function agrees, while proofs from `poly.expr` bodies become template-scope
/// constraints.
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

  for (FuncDefOp func : walkCollect<FuncDefOp>(templateOp)) {
    DenseMap<StringAttr, InferredType> funcReplacements;
    if (failed(TypeVarInferenceCollector(info, funcReplacements).collect(func.getOperation()))) {
      return failure();
    }
    if (!funcReplacements.empty()) {
      info.functionReplacements.try_emplace(func.getOperation(), funcReplacements);
    }
  }

  for (TemplateExprOp expr : walkCollect<TemplateExprOp>(templateOp)) {
    if (failed(TypeVarInferenceCollector(info, info.templateScopeReplacements)
                   .collect(expr.getOperation()))) {
      return failure();
    }
  }

  bool changed = false;
  SymbolTableCollection tables;
  ModuleOp module = templateOp->getParentOfType<ModuleOp>();
  for (verif::ContractOp contract : walkCollect<verif::ContractOp>(templateOp)) {
    if (failed(collectContractTargetFunctionInferences(
            info, contract, module, /*infoByTemplate=*/nullptr, tables, changed
        )) ||
        failed(collectContractTargetStructInferences(
            info, contract, module, /*infoByTemplate=*/nullptr, tables, changed
        ))) {
      return failure();
    }
  }

  if (failed(recomputeTemplateWideReplacements(info))) {
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
    // Note: SmallVector doesn't work because the element size is too large.
    std::vector<TemplateInferenceInfo> templateInfos;
    WalkResult collectResult = module.walk([&templateInfos](TemplateOp templateOp) {
      FailureOr<TemplateInferenceInfo> info = buildInfo(templateOp);
      if (failed(info)) {
        return WalkResult::interrupt();
      }
      templateInfos.push_back(std::move(*info));
      return WalkResult::advance();
    });
    if (collectResult.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    DenseMap<Operation *, TemplateInferenceInfo *> mutableInfoByTemplate;
    for (TemplateInferenceInfo &info : templateInfos) {
      mutableInfoByTemplate.try_emplace(info.templateOp.getOperation(), &info);
    }
    if (failed(
            collectExternalContractTargetInferences(module, templateInfos, mutableInfoByTemplate)
        )) {
      signalPassFailure();
      return;
    }
    if (failed(inferStructTemplateParamUses(module, templateInfos, mutableInfoByTemplate))) {
      signalPassFailure();
      return;
    }

    SmallVector<TemplateInferenceInfo *> rewrites;
    DenseMap<Operation *, const TemplateInferenceInfo *> infoByTemplate;
    for (TemplateInferenceInfo &info : templateInfos) {
      infoByTemplate.try_emplace(info.templateOp.getOperation(), &info);
      if (!info.replacements.empty() || !info.functionReplacements.empty()) {
        rewrites.push_back(&info);
      }
    }

    // Rewrite each affected template in place, but keep the converters alive so
    // call-site rewriting can use the same positional template-parameter map.
    SmallVector<std::unique_ptr<TypeVarReplacementConverter>> converterStorage;
    DenseMap<Operation *, const TypeVarReplacementConverter *> convertersByTemplate;
    for (TemplateInferenceInfo *info : rewrites) {
      DenseMap<StringAttr, Type> replacements;
      for (const auto &entry : info->replacements) {
        replacements.try_emplace(entry.first, entry.second.type);
      }
      auto converter = std::make_unique<TypeVarReplacementConverter>(
          module.getContext(), info->templatePath, info->oldParamOrder, replacements
      );
      convertersByTemplate.try_emplace(info->templateOp.getOperation(), converter.get());
      converterStorage.push_back(std::move(converter));
    }

    // Clone concrete call-site instantiations before trimming template
    // arguments; the clone decision needs the original explicit arguments.
    SpecializedCallableCloneCache functionCloneCache;
    SpecializedCallableCloneCache contractCloneCache;
    SpecializedTemplateCloneCache templateCloneCache;
    if (failed(specializeFunctionLocalCallables(
            module, infoByTemplate, functionCloneCache, contractCloneCache, templateCloneCache
        ))) {
      signalPassFailure();
      return;
    }

    for (TemplateInferenceInfo *info : rewrites) {
      const TypeVarReplacementConverter *converter =
          convertersByTemplate.lookup(info->templateOp.getOperation());
      assert(converter && "rewritten template must have a converter");
      WalkResult validationResult = info->templateOp.walk([converter](Operation *op) {
        return failed(converter->validateOperation(op)) ? WalkResult::interrupt()
                                                        : WalkResult::advance();
      });
      if (validationResult.wasInterrupted()) {
        signalPassFailure();
        return;
      }
      if (failed(convertOperationTypesIn(info->templateOp.getOperation(), *converter))) {
        signalPassFailure();
        return;
      }
      removeIdentityCasts(info->templateOp.getOperation());
    }

    if (failed(updateExternalContractTemplateParams(module, convertersByTemplate))) {
      signalPassFailure();
      return;
    }

    // Calls that still target rewritten templates are updated after all target
    // templates have their converters registered.
    if (failed(updateCallableTemplateParams(module, convertersByTemplate))) {
      signalPassFailure();
      return;
    }
    // Re-run specialization after rewriting because template bodies may now
    // contain concrete forwarded calls into otherwise unconstrained templates.
    if (failed(specializeFunctionLocalCallables(
            module, infoByTemplate, functionCloneCache, contractCloneCache, templateCloneCache
        ))) {
      signalPassFailure();
      return;
    }
    if (failed(updateStructTemplateParams(module, convertersByTemplate))) {
      signalPassFailure();
      return;
    }
    if (failed(instantiateConcreteStructUses(module))) {
      signalPassFailure();
      return;
    }

    // Erase resolved parameters last; before this point, their original order is
    // still useful for template argument list conversion.
    for (TemplateInferenceInfo *info : rewrites) {
      removeResolvedParams(*info);
    }
  }
};

} // namespace
