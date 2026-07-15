//===-- SymbolHelper.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Util/SymbolLookup.h"

#include <mlir/IR/Attributes.h>
#include <mlir/Interfaces/CallInterfaces.h>

#include <optional>
#include <ranges>

namespace llzk {

namespace component {
class StructType;
class StructDefOp;
class MemberDefOp;
} // namespace component

namespace function {
class FuncDefOp;
} // namespace function
namespace polymorphic {
class TemplateOp;
} // namespace polymorphic

llvm::SmallVector<mlir::StringRef> getNames(mlir::SymbolRefAttr ref);
llvm::SmallVector<mlir::FlatSymbolRefAttr> getPieces(mlir::SymbolRefAttr ref);

/// Construct a FlatSymbolRefAttr with the given content.
inline mlir::FlatSymbolRefAttr
getFlatSymbolRefAttr(mlir::MLIRContext *context, const mlir::Twine &twine) {
  return mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(context, twine));
}

/// Build a SymbolRefAttr that prepends `tail` with `root`, i.e., `root::tail`.
inline mlir::SymbolRefAttr asSymbolRefAttr(mlir::StringAttr root, mlir::SymbolRefAttr tail) {
  return mlir::SymbolRefAttr::get(root, getPieces(tail));
}

/// Build a SymbolRefAttr from the list of pieces.
inline mlir::SymbolRefAttr asSymbolRefAttr(llvm::ArrayRef<mlir::FlatSymbolRefAttr> path) {
  return mlir::SymbolRefAttr::get(path.front().getAttr(), path.drop_front());
}

/// Build a SymbolRefAttr from the list of pieces.
inline mlir::SymbolRefAttr asSymbolRefAttr(const std::vector<mlir::FlatSymbolRefAttr> &path) {
  return asSymbolRefAttr(llvm::ArrayRef<mlir::FlatSymbolRefAttr>(path));
}

/// Return SymbolRefAttr like the one given but with the root/head element removed.
inline mlir::SymbolRefAttr getTailAsSymbolRefAttr(mlir::SymbolRefAttr symbol) {
  return asSymbolRefAttr(symbol.getNestedReferences());
}

/// Return SymbolRefAttr like the one given but with the leaf/final element removed.
inline mlir::SymbolRefAttr getPrefixAsSymbolRefAttr(mlir::SymbolRefAttr symbol) {
  return mlir::SymbolRefAttr::get(
      symbol.getRootReference(), symbol.getNestedReferences().drop_back()
  );
}

/// Return SymbolRefAttr like the one given but with the leaf (final) element replaced.
mlir::SymbolRefAttr replaceLeaf(mlir::SymbolRefAttr orig, mlir::FlatSymbolRefAttr newLeaf);
inline mlir::SymbolRefAttr replaceLeaf(mlir::SymbolRefAttr orig, mlir::StringAttr newLeaf) {
  return replaceLeaf(orig, mlir::FlatSymbolRefAttr::get(newLeaf));
}
inline mlir::SymbolRefAttr replaceLeaf(mlir::SymbolRefAttr orig, const mlir::Twine &newLeaf) {
  return replaceLeaf(orig, mlir::StringAttr::get(orig.getContext(), newLeaf));
}

/// Return SymbolRefAttr like the one given but with a new leaf (final) element added.
mlir::SymbolRefAttr appendLeaf(mlir::SymbolRefAttr orig, mlir::FlatSymbolRefAttr newLeaf);
inline mlir::SymbolRefAttr appendLeaf(mlir::SymbolRefAttr orig, mlir::StringAttr newLeaf) {
  return appendLeaf(orig, mlir::FlatSymbolRefAttr::get(newLeaf));
}
inline mlir::SymbolRefAttr appendLeaf(mlir::SymbolRefAttr orig, const mlir::Twine &newLeaf) {
  return appendLeaf(orig, mlir::StringAttr::get(orig.getContext(), newLeaf));
}

/// Return SymbolRefAttr like the one given but with the leaf (final) element appended with the
/// given suffix.
mlir::SymbolRefAttr appendLeafName(mlir::SymbolRefAttr orig, const mlir::Twine &newLeafSuffix);

/// Build the module-level symbol name from an instantiated template name and function name.
std::string buildInstantiatedFunctionName(
    llvm::StringRef instantiatedTemplateName, llvm::StringRef functionName
);

/// Build the module-level symbol name for a fully-instantiated template function.
///
/// The generated name encodes the original template name, the concrete template
/// arguments, and the leaf function name. It matches the naming convention used
/// by the flattening pass for full function instantiations.
std::string buildInstantiatedFunctionName(
    llvm::StringRef templateName, llvm::StringRef functionName,
    llvm::ArrayRef<mlir::Attribute> templateArgs
);

/// Return the callee path for a module-level clone of a nested template function.
///
/// Given a callee like `@Template::@f` or `@M::@Template::@f`, this removes the
/// template and function leaves and appends `instantiatedFunctionName`.
mlir::SymbolRefAttr getInstantiatedFunctionCallee(
    mlir::SymbolRefAttr templateFunctionCallee, mlir::StringAttr instantiatedFunctionName
);

/// @brief Starting at the Operation `from`, find the nearest ancestor ModuleOp marked with
/// LANG_ATTR_NAME from which symbol lookups are rooted.
mlir::FailureOr<mlir::ModuleOp> getRootModule(mlir::Operation *from);
mlir::FailureOr<mlir::SymbolRefAttr>
getPathFromRoot(mlir::SymbolOpInterface to, mlir::ModuleOp *foundRoot = nullptr);
mlir::FailureOr<mlir::SymbolRefAttr>
getPathFromRoot(component::StructDefOp &to, mlir::ModuleOp *foundRoot = nullptr);
mlir::FailureOr<mlir::SymbolRefAttr>
getPathFromRoot(component::MemberDefOp &to, mlir::ModuleOp *foundRoot = nullptr);
mlir::FailureOr<mlir::SymbolRefAttr>
getPathFromRoot(function::FuncDefOp &to, mlir::ModuleOp *foundRoot = nullptr);

/// @brief With include statements, there may be root modules nested within
/// other root modules. This function resolves the topmost root module.
mlir::FailureOr<mlir::ModuleOp> getTopRootModule(mlir::Operation *from);
mlir::FailureOr<mlir::SymbolRefAttr>
getPathFromTopRoot(mlir::SymbolOpInterface to, mlir::ModuleOp *foundRoot = nullptr);
mlir::FailureOr<mlir::SymbolRefAttr>
getPathFromTopRoot(component::StructDefOp &to, mlir::ModuleOp *foundRoot = nullptr);
mlir::FailureOr<mlir::SymbolRefAttr>
getPathFromTopRoot(component::MemberDefOp &to, mlir::ModuleOp *foundRoot = nullptr);
mlir::FailureOr<mlir::SymbolRefAttr>
getPathFromTopRoot(function::FuncDefOp &to, mlir::ModuleOp *foundRoot = nullptr);

/// @brief Lookup the `StructType` of the main instance.
///
/// This is specified by a `TypeAttr` on the top-level module with the key `LLZK_MAIN_ATTR_NAME`
/// and is optional, in which case the result will be `success(nullptr)`.
mlir::FailureOr<llzk::component::StructType> getMainInstanceType(mlir::Operation *lookupFrom);

/// @brief Lookup the `StructDefOp` of the main instance.
///
/// This is specified by a `TypeAttr` on the top-level module with the key `LLZK_MAIN_ATTR_NAME`
/// and is optional, in which case the result will be `success(nullptr)`.
mlir::FailureOr<SymbolLookupResult<llzk::component::StructDefOp>>
getMainInstanceDef(mlir::SymbolTableCollection &symbolTable, mlir::Operation *lookupFrom);

/// @brief Based on mlir::CallOpInterface::resolveCallable, but using LLZK lookup helpers
/// @tparam T the type of symbol being resolved (e.g., function::FuncDefOp)
/// @param symbolTable
/// @param call
/// @return the symbol or failure
template <typename T>
inline mlir::FailureOr<SymbolLookupResult<T>>
resolveCallable(mlir::SymbolTableCollection &symbolTable, mlir::CallOpInterface call) {
  mlir::CallInterfaceCallable callable = call.getCallableForCallee();
  if (auto symbolVal = llvm::dyn_cast<mlir::Value>(callable)) {
    return SymbolLookupResult<T>(symbolVal.getDefiningOp());
  }

  // If the callable isn't a value, lookup the symbol reference.
  // We first try to resolve in the nearest symbol table, as per the default
  // MLIR behavior. If the resulting operation is not found, we will then
  // use the LLZK lookup helpers.
  auto symbolRef = llvm::cast<mlir::SymbolRefAttr>(callable);
  mlir::Operation *op = symbolTable.lookupNearestSymbolFrom(call.getOperation(), symbolRef);

  if (op) {
    return SymbolLookupResult<T>(std::move(op));
  }
  // Otherwise, use the top-level lookup.
  return lookupTopLevelSymbol<T>(symbolTable, symbolRef, call.getOperation());
}

/// Resolve a callable without emitting a diagnostic for missing top-level symbols.
///
/// Use this when an unresolved call should make an analysis conservatively skip the call instead of
/// reporting a verifier-style symbol error.
template <typename T>
inline mlir::FailureOr<SymbolLookupResult<T>>
resolveCallableSilently(mlir::SymbolTableCollection &symbolTable, mlir::CallOpInterface call) {
  mlir::CallInterfaceCallable callable = call.getCallableForCallee();
  if (auto symbolVal = llvm::dyn_cast<mlir::Value>(callable)) {
    SymbolLookupResult<T> result(symbolVal.getDefiningOp());
    if (!result) {
      return mlir::failure();
    }
    return result;
  }

  auto symbolRef = llvm::cast<mlir::SymbolRefAttr>(callable);
  if (mlir::Operation *op = symbolTable.lookupNearestSymbolFrom(call.getOperation(), symbolRef)) {
    SymbolLookupResult<T> result(op);
    if (!result) {
      return mlir::failure();
    }
    return result;
  }
  return lookupTopLevelSymbol<T>(
      symbolTable, symbolRef, call.getOperation(), /*reportMissing=*/false
  );
}

template <typename T>
inline mlir::FailureOr<SymbolLookupResult<T>> resolveCallable(mlir::CallOpInterface call) {
  mlir::SymbolTableCollection symbolTable;
  return resolveCallable<T>(symbolTable, call);
}

/// Return the template scope that should be used to resolve flat constant
/// references from the given origin. For ops nested in a `verif.contract`
/// targeting a templated symbol, this is the target's template even when the
/// contract is physically nested elsewhere. Otherwise this is the nearest
/// physical `poly.template`, if any.
mlir::FailureOr<polymorphic::TemplateOp>
getConstResolutionTemplate(mlir::SymbolTableCollection &tables, mlir::Operation *origin);

/// Ensure that the given symbol (that is used as a parameter of the given type) can be resolved.
/// If `requiredParamType` is provided, any resolved template symbol must have exactly that type.
mlir::LogicalResult verifyParamOfType(
    mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr param, mlir::Type structOrArrayType,
    mlir::Operation *origin, std::optional<mlir::Type> requiredParamType = std::nullopt
);

/// Ensure that any symbols that appear within the given attributes (that are parameters of the
/// given type) can be resolved. If `requiredParamType` is provided, any resolved template symbols
/// must have exactly that type.
mlir::LogicalResult verifyParamsOfType(
    mlir::SymbolTableCollection &tables, mlir::ArrayRef<mlir::Attribute> tyParams,
    mlir::Type structOrArrayType, mlir::Operation *origin,
    std::optional<mlir::Type> requiredParamType = std::nullopt
);

/// Ensure that all symbols used within the type can be resolved.
mlir::FailureOr<component::StructDefOp> verifyStructTypeResolution(
    mlir::SymbolTableCollection &tables, component::StructType ty, mlir::Operation *origin
);

/// Ensure that all symbols used within the given Type instance can be resolved.
mlir::LogicalResult
verifyTypeResolution(mlir::SymbolTableCollection &tables, mlir::Operation *origin, mlir::Type type);

/// Ensure that all symbols used within all Type instances can be resolved.
template <std::ranges::input_range Range>
mlir::LogicalResult verifyTypeResolution(
    mlir::SymbolTableCollection &tables, mlir::Operation *origin, const Range &types
) {
  // Check all before returning to present all applicable type errors in one compilation.
  bool failed = false;
  for (const auto &t : types) {
    failed |= mlir::failed(verifyTypeResolution(tables, origin, t));
  }
  return mlir::LogicalResult::failure(failed);
}

} // namespace llzk
