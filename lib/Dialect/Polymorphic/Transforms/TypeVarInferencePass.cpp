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
#include "llzk/Dialect/Verif/IR/Ops.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolLookup.h"
#include "llzk/Util/SymbolTableLLZK.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>

#include <memory>

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

namespace {

/// A concrete type inferred at a source location.
///
/// The location is retained so a later conflicting inference can point back to
/// the original proof site with a note.
struct InferredType {
  Type type;
  Location loc;
};

/// Template-local clone callees created by this pass, keyed by the clone name that
/// would be preferred before `SymbolTable::insert` uniquifies it.
///
/// The cache deliberately records only pass-created clones. A user-defined symbol
/// may already occupy the preferred name, and `SymbolTable::insert` will rename the
/// clone in that case. Reusing the cached callee avoids confusing such user symbols
/// with generated clones and keeps repeated calls to the same instantiation pointed
/// at the same uniquely-named clone.
using SpecializedFunctionCloneCache = DenseMap<Operation *, llvm::StringMap<SymbolRefAttr>>;

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

/// Convert a function type by recursively converting inputs and results.
template <typename ConverterT>
static Type convertFunctionType(FunctionType funcTy, ConverterT &converter) {
  SmallVector<Type> newInputs;
  SmallVector<Type> newResults;
  bool changed = convertTypeRange(funcTy.getInputs(), newInputs, converter);
  changed |= convertTypeRange(funcTy.getResults(), newResults, converter);
  return changed ? FunctionType::get(funcTy.getContext(), newInputs, newResults) : funcTy;
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
      Type newElemTy = convertType(arrTy.getElementType(), resolvingParams);
      if (newElemTy == arrTy.getElementType()) {
        return ty;
      }
      return llzk::polymorphic::detail::flattenInstantiatedArrayType(arrTy, newElemTy);
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

private:
  Attribute convertAttr(Attribute attr, DenseSet<StringAttr> &resolvingParams) const {
    if (!attr) {
      return attr;
    }
    if (auto tyAttr = llvm::dyn_cast<TypeAttr>(attr)) {
      Type newTy = convertType(tyAttr.getValue(), resolvingParams);
      return newTy == tyAttr.getValue() ? attr : TypeAttr::get(newTy);
    }
    if (auto arrAttr = llvm::dyn_cast<ArrayAttr>(attr)) {
      SmallVector<Attribute> newAttrs;
      bool changed = false;
      for (Attribute nested : arrAttr.getValue()) {
        Attribute newNested = convertTemplateArgAttr(nested, resolvingParams);
        newAttrs.push_back(newNested);
        changed |= newNested != nested;
      }
      return changed ? ArrayAttr::get(ctx_, newAttrs) : attr;
    }
    return attr;
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
    Attribute expectedAttr = TypeAttr::get(convertType(replacementIt->second));
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
    for (auto indexedAttr : llvm::enumerate(params.getValue())) {
      unsigned index = indexedAttr.index();
      Attribute attr = indexedAttr.value();
      if (trimResolvedParams_ && removeOwnedParams &&
          removedParams_.contains(oldParamOrder_[index])) {
        changed = true;
        continue;
      }
      Attribute newAttr = convertTemplateArgAttr(attr, resolvingParams);
      newParams.push_back(newAttr);
      changed |= newAttr != attr;
    }
    return changed ? StructType::get(structTy.getNameRef(), ArrayAttr::get(ctx_, newParams))
                   : structTy;
  }
};

/// Return true if `attr` is a concrete struct-instantiation argument.
static bool isConcreteInstantiationAttr(Attribute attr) {
  if (auto tyAttr = llvm::dyn_cast<TypeAttr>(attr)) {
    return isConcreteType(tyAttr.getValue(), /*allowStructParams=*/false);
  }
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr)) {
    return !isDynamic(intAttr);
  }
  return false;
}

/// Rewrite a function type and keep the entry block arguments in sync.
///
/// `function.def` stores its function signature separately from the entry block
/// argument types, so both surfaces must be updated when an argument slot is
/// rewritten from `!poly.tvar` to a concrete type.
template <typename ConverterT>
static void updateFuncSignature(FuncDefOp func, ConverterT &converter) {
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
template <typename ConverterT>
static bool convertOperationTypes(Operation *op, ConverterT &converter) {
  if (auto readOp = llvm::dyn_cast<ReadArrayOp>(op)) {
    Type newResultTy = converter.convertType(readOp.getResult().getType());
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

/// Convert concrete parameterized struct uses into template-local struct clones.
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
  /// Fully-qualified names of template-local clones created by this converter.
  DenseSet<SymbolRefAttr> instantiatedCloneNames_;
  /// Template-local self types active while rewriting a cloned struct body.
  DenseMap<StructType, StructType> activeLocalStructReplacements_;
  /// Type replacements active while rewriting the body of one cloned struct.
  DenseMap<StringAttr, Type> activeTypeReplacements_;

public:
  /// Create a converter for concrete struct instantiations in `module`.
  ConcreteStructInstantiationConverter(
      MLIRContext *ctx, ModuleOp module, SymbolTableCollection &tables
  )
      : ctx_(ctx), module_(module), tables_(tables) {}

  /// Convert a type by recursively instantiating concrete parameterized structs.
  Type convertType(Type ty) {
    if (!ty) {
      return ty;
    }
    if (auto tvarTy = llvm::dyn_cast<TypeVarType>(ty)) {
      auto it = activeTypeReplacements_.find(tvarTy.getNameRef().getAttr());
      return it == activeTypeReplacements_.end() ? ty : it->second;
    }
    if (auto arrTy = llvm::dyn_cast<ArrayType>(ty)) {
      Type newElemTy = convertType(arrTy.getElementType());
      return newElemTy == arrTy.getElementType()
                 ? ty
                 : llzk::polymorphic::detail::flattenInstantiatedArrayType(arrTy, newElemTy);
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

private:
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

    if (!llvm::all_of(newParams, isConcreteInstantiationAttr)) {
      return convertedTy;
    }

    FailureOr<StructType> cloneTy = getOrCreateStructClone(convertedTy, newParams);
    return succeeded(cloneTy) ? *cloneTy : convertedTy;
  }

  /// Build the symbol name for a template-local specialized struct clone.
  static std::string
  buildTemplateLocalStructCloneName(StringRef structName, ArrayRef<Attribute> concreteParams) {
    std::string cloneName = BuildShortTypeString::from(concreteParams);
    cloneName += "_";
    cloneName += structName;
    return cloneName;
  }

  /// Create or reuse a template-local clone for `concreteStructTy`.
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

    std::string cloneName =
        buildTemplateLocalStructCloneName(origStruct.getSymName(), concreteParams);
    SymbolTable &templateSymbols = tables_.getSymbolTable(parentTemplate);
    if (Operation *existing = templateSymbols.lookup(cloneName)) {
      if (auto existingStruct = llvm::dyn_cast<StructDefOp>(existing)) {
        StructType localTy = existingStruct.getType();
        StructType remoteTy = StructType::get(
            existingStruct.getFullyQualifiedName(), ArrayAttr::get(ctx_, concreteParams)
        );
        instantiations_.try_emplace(concreteStructTy, StructInstantiationTypes {localTy, remoteTy});
        instantiatedCloneNames_.insert(existingStruct.getFullyQualifiedName());
        return remoteTy;
      }
      return failure();
    }

    StructDefOp clone = origStruct.clone();
    clone.setSymName(cloneName);
    templateSymbols.insert(clone, Block::iterator(origStruct));
    StructType localTy = clone.getType();
    StructType remoteTy =
        StructType::get(clone.getFullyQualifiedName(), ArrayAttr::get(ctx_, concreteParams));
    instantiations_.try_emplace(concreteStructTy, StructInstantiationTypes {localTy, remoteTy});
    instantiatedCloneNames_.insert(clone.getFullyQualifiedName());

    DenseMap<StringAttr, Type> previousReplacements = std::move(activeTypeReplacements_);
    DenseMap<StructType, StructType> previousLocalStructReplacements =
        std::move(activeLocalStructReplacements_);
    activeTypeReplacements_.clear();
    activeLocalStructReplacements_.clear();
    activeLocalStructReplacements_.try_emplace(concreteStructTy, localTy);
    activeLocalStructReplacements_.try_emplace(
        StructType::get(localTy.getNameRef(), ArrayAttr::get(ctx_, concreteParams)), localTy
    );
    for (auto [paramName, concreteAttr] : llvm::zip_equal(paramNames.getValue(), concreteParams)) {
      auto paramSym = llvm::dyn_cast<FlatSymbolRefAttr>(paramName);
      auto concreteType = llvm::dyn_cast<TypeAttr>(concreteAttr);
      if (paramSym && concreteType) {
        activeTypeReplacements_.try_emplace(paramSym.getAttr(), concreteType.getValue());
      }
    }
    clone.walk([this](Operation *op) { convertOperationTypes(op, *this); });
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

/// Diagnose explicit call-site template parameters that disagree with concrete
/// replacements proven inside the callee body.
static LogicalResult diagnoseCallParamsMismatch(
    CallOp callOp, ArrayAttr callParams, ArrayRef<StringAttr> oldParamOrder,
    const DenseMap<StringAttr, Type> &replacements
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
    Attribute expectedAttr = TypeAttr::get(entry.second);
    if (attr == expectedAttr) {
      continue;
    }

    InFlightDiagnostic diag = callOp.emitError()
                              << "explicit template argument for inferred parameter @"
                              << entry.first.getValue() << " must match inferred type "
                              << entry.second << ", but found ";
    if (auto typeAttr = llvm::dyn_cast<TypeAttr>(attr)) {
      diag << typeAttr.getValue();
    } else {
      diag << attr;
    }
    return diag;
  }
  return callOp.emitError() << "explicit template arguments do not match inferred callee types";
}

/// Build concrete type-variable replacements from an explicit call instantiation.
///
/// A specialized function clone keeps the original enclosing `poly.template`,
/// so non-type parameters remain available to operations such as
/// `poly.read_const`. Type-variable parameters mentioned by the cloned function
/// are replaced by concrete types from the call's template argument list.
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
    if (!tyAttr) {
      return DenseMap<StringAttr, Type>();
    }
    replacements.try_emplace(paramName, tyAttr.getValue());
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

/// Return true when `func` still needs per-call cloning after template-wide rewrites.
///
/// Parameters in `info.replacements` are safe to rewrite in the template itself.
/// A clone is only necessary when the function mentions an eligible type
/// variable that was not proven for every relevant entry point.
static bool hasResidualFunctionTvar(const TemplateInferenceInfo &info, FuncDefOp func) {
  return llvm::any_of(info.typeVarParams, [&](const auto &entry) {
    return !info.replacements.contains(entry.first) && funcMentionsParam(func, entry.first);
  });
}

/// Build the symbol name for a template-local specialized function clone.
///
/// The containing template already contributes its own name to the fully-qualified
/// callee path, so the clone name only encodes the instantiation layout plus the
/// original function name. Parameters that remain provided by the template are
/// represented with the same placeholder marker used by flattening's partial
/// instantiation names.
static std::string buildTemplateLocalFunctionCloneName(
    StringRef functionName, ArrayRef<StringAttr> oldParamOrder,
    const DenseMap<StringAttr, Type> &replacements
) {
  SmallVector<Attribute> attrsForName;
  attrsForName.reserve(oldParamOrder.size());
  for (StringAttr paramName : oldParamOrder) {
    auto replacementIt = replacements.find(paramName);
    attrsForName.push_back(
        replacementIt == replacements.end() ? Attribute() : TypeAttr::get(replacementIt->second)
    );
  }
  std::string cloneName = BuildShortTypeString::from(attrsForName);
  cloneName += "_";
  cloneName += functionName;
  return cloneName;
}

/// Return the callee path for a clone nested in the same template as the original callee.
static SymbolRefAttr
getTemplateLocalFunctionCloneCallee(SymbolRefAttr originalCallee, StringAttr cloneName) {
  SmallVector<FlatSymbolRefAttr> pieces = getPieces(originalCallee);
  assert(pieces.size() >= 2 && "callee must include at least template and function names");
  pieces.pop_back();
  pieces.push_back(FlatSymbolRefAttr::get(cloneName));
  return asSymbolRefAttr(pieces);
}

/// Create or reuse a template-local clone for a concrete function-local tvar instantiation.
static FailureOr<SymbolRefAttr> getOrCreateSpecializedFunctionClone(
    TemplateOp templateOp, FuncDefOp func, SymbolRefAttr originalCallee,
    const TemplateInferenceInfo &info, const DenseMap<StringAttr, Type> &replacements,
    SymbolTableCollection &tables, llvm::StringMap<SymbolRefAttr> &cloneCallees
) {
  std::string requestedFuncName =
      buildTemplateLocalFunctionCloneName(func.getSymName(), info.oldParamOrder, replacements);
  // `requestedFuncName` is a stable instantiation key; the clone may receive a
  // uniquified symbol name when inserted if user IR already defines that symbol.
  auto cachedCallee = cloneCallees.find(requestedFuncName);
  if (cachedCallee != cloneCallees.end()) {
    return cachedCallee->second;
  }

  SymbolTable &templateSymbols = tables.getSymbolTable(templateOp);

  FuncDefOp clone = func.clone();
  clone.setSymName(requestedFuncName);
  templateSymbols.insert(clone, Block::iterator(func));

  TypeVarReplacementConverter converter(
      templateOp.getContext(), info.templatePath, info.oldParamOrder, replacements,
      /*trimResolvedParams=*/false
  );
  clone.walk([&converter](Operation *op) { convertOperationTypes(op, converter); });
  removeIdentityCasts(clone.getOperation());
  SymbolRefAttr cloneCallee =
      getTemplateLocalFunctionCloneCallee(originalCallee, clone.getSymNameAttr());
  cloneCallees.try_emplace(requestedFuncName, cloneCallee);
  return cloneCallee;
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

    TemplateOp callTemplate = getParentOfType<TemplateOp>(callOp.getOperation());
    bool resolveTemplateSymbolArgs =
        callTemplate && callTemplate.getOperation() == parentTemplate.getOperation();
    ArrayAttr oldParams = callOp.getTemplateParamsAttr();
    FailureOr<ArrayAttr> newParams =
        converter->convertTemplateParams(oldParams, callOp, resolveTemplateSymbolArgs);
    if (failed(newParams)) {
      failedConversion = true;
      return;
    }
    if (oldParams != *newParams) {
      callOp.setTemplateParamsAttr(*newParams);
      modified = true;
    }

    if (resolveTemplateSymbolArgs) {
      for (Value result : callOp.getResults()) {
        Type newTy = converter->convertType(result.getType());
        if (newTy != result.getType()) {
          result.setType(newTy);
          modified = true;
        }
      }
    }
  });
  module.walk([&](verif::IncludeOp includeOp) {
    if (failedConversion) {
      return;
    }
    FailureOr<SymbolLookupResult<verif::ContractOp>> target = includeOp.getCalleeTarget(tables);
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

    TemplateOp includeTemplate = getParentOfType<TemplateOp>(includeOp.getOperation());
    bool resolveTemplateSymbolArgs =
        includeTemplate && includeTemplate.getOperation() == parentTemplate.getOperation();
    ArrayAttr oldParams = includeOp.getTemplateParamsAttr();
    FailureOr<ArrayAttr> newParams =
        converter->convertTemplateParams(oldParams, includeOp, resolveTemplateSymbolArgs);
    if (failed(newParams)) {
      failedConversion = true;
      return;
    }
    if (oldParams != *newParams) {
      includeOp.setTemplateParamsAttr(*newParams);
      modified = true;
    }
  });
  if (failedConversion) {
    return failure();
  }
  return modified;
}

/// Instantiate concrete parameterized struct types exposed by type-variable inference.
static void instantiateConcreteStructUses(ModuleOp module) {
  SymbolTableCollection tables;
  ConcreteStructInstantiationConverter converter(module.getContext(), module, tables);
  module.walk([&converter](Operation *op) { convertOperationTypes(op, converter); });
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
    ModuleOp module, DenseMap<Operation *, const TemplateInferenceInfo *> &infoByTemplate,
    SpecializedFunctionCloneCache &cloneCache
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

    ArrayAttr callParams = callOp.getTemplateParamsAttr();
    DenseMap<StringAttr, Type> replacements =
        getConcreteCallSiteReplacements(callParams, *info, targetFunc);
    if (replacements.empty()) {
      return;
    }
    DenseMap<StringAttr, Type> inferredReplacements =
        getFunctionProofReplacements(*info, targetFunc);
    if (failed(diagnoseCallParamsMismatch(
            callOp, callParams, info->oldParamOrder, inferredReplacements
        ))) {
      failedClone = true;
      return;
    }

    FailureOr<SymbolRefAttr> cloneCallee = getOrCreateSpecializedFunctionClone(
        parentTemplate, targetFunc, callOp.getCalleeAttr(), *info, replacements, tables,
        cloneCache[parentTemplate.getOperation()]
    );
    if (failed(cloneCallee)) {
      failedClone = true;
      return;
    }

    callOp.setCalleeAttr(*cloneCallee);
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
    SmallVector<TemplateInferenceInfo> templateInfos;
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
    SpecializedFunctionCloneCache cloneCache;
    if (failed(specializeFunctionLocalCalls(module, infoByTemplate, cloneCache))) {
      signalPassFailure();
      return;
    }

    for (TemplateInferenceInfo *info : rewrites) {
      const TypeVarReplacementConverter *converter =
          convertersByTemplate.lookup(info->templateOp.getOperation());
      assert(converter && "rewritten template must have a converter");
      info->templateOp.walk([converter](Operation *op) { convertOperationTypes(op, *converter); });
      removeIdentityCasts(info->templateOp.getOperation());
    }

    // Calls that still target rewritten templates are updated after all target
    // templates have their converters registered.
    if (failed(updateCallableTemplateParams(module, convertersByTemplate))) {
      signalPassFailure();
      return;
    }
    // Re-run specialization after rewriting because template bodies may now
    // contain concrete forwarded calls into otherwise unconstrained templates.
    if (failed(specializeFunctionLocalCalls(module, infoByTemplate, cloneCache))) {
      signalPassFailure();
      return;
    }
    instantiateConcreteStructUses(module);

    // Erase resolved parameters last; before this point, their original order is
    // still useful for template argument list conversion.
    for (TemplateInferenceInfo *info : rewrites) {
      removeResolvedParams(*info);
    }
  }
};

} // namespace
