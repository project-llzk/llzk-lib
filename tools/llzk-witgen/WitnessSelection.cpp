//===-- WitnessSelection.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "WitnessSelection.h"

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/POD/IR/Attrs.h"
#include "llzk/Dialect/POD/IR/Types.h"

#include <mlir/IR/Operation.h>

using namespace mlir;

namespace llzk::witgen {
namespace {

/// Return whether the type contains nested signal members.
static FailureOr<bool>
typeContainsSignals(Type type, SymbolTableCollection &tables, Operation *origin);

/// Return whether the struct contains nested signal members.
static FailureOr<bool>
structContainsSignals(component::StructDefOp def, SymbolTableCollection &tables, Operation *origin) {
  for (component::MemberDefOp member : def.getMemberDefs()) {
    if (memberIsSignal(def, member)) {
      return true;
    }
    auto nested = typeContainsSignals(member.getType(), tables, origin);
    if (failed(nested)) {
      return failure();
    }
    if (*nested) {
      return true;
    }
  }
  return false;
}

/// Return whether the type contains nested signal members.
static FailureOr<bool>
typeContainsSignals(Type type, SymbolTableCollection &tables, Operation *origin) {
  if (auto structType = dyn_cast<component::StructType>(type)) {
    auto defLookup = structType.getDefinition(tables, origin);
    if (failed(defLookup)) {
      return failure();
    }
    return structContainsSignals(defLookup->get(), tables, origin);
  }
  return false;
}

/// Append recursively selected signal leaves from one signal aggregate.
static LogicalResult appendSignalLeafBindings(
    Type type, ArrayRef<std::string> prefix, SmallVectorImpl<OutputBinding> &out,
    Operation *origin
) {
  if (isa<felt::FeltType, array::ArrayType>(type)) {
    out.push_back(OutputBinding {llvm::SmallVector<std::string>(prefix.begin(), prefix.end()), type});
    return success();
  }

  if (auto podType = dyn_cast<pod::PodType>(type)) {
    for (pod::RecordAttr record : podType.getRecords()) {
      llvm::SmallVector<std::string> path(prefix.begin(), prefix.end());
      path.push_back(record.getName().getValue().str());
      if (failed(appendSignalLeafBindings(record.getType(), path, out, origin))) {
        return failure();
      }
    }
    return success();
  }

  origin->emitError("signal members in llzk-witgen must be felts, felt arrays, or PODs of felts");
  return failure();
}

/// Append recursively selected signal leaves from one struct container.
static LogicalResult appendStructSignalBindings(
    component::StructDefOp def, SymbolTableCollection &tables, Operation *origin,
    SmallVectorImpl<OutputBinding> &out, ArrayRef<std::string> prefix = {}
) {
  for (component::MemberDefOp member : def.getMemberDefs()) {
    llvm::SmallVector<std::string> path(prefix.begin(), prefix.end());
    path.push_back(member.getSymName().str());

    if (memberIsSignal(def, member)) {
      if (failed(appendSignalLeafBindings(member.getType(), path, out, origin))) {
        return failure();
      }
      continue;
    }

    auto nested = typeContainsSignals(member.getType(), tables, origin);
    if (failed(nested)) {
      return failure();
    }
    if (!*nested) {
      continue;
    }

    auto structType = dyn_cast<component::StructType>(member.getType());
    if (!structType) {
      member.emitError("non-struct signal container is unsupported in llzk-witgen");
      return failure();
    }
    auto defLookup = structType.getDefinition(tables, origin);
    if (failed(defLookup)) {
      return failure();
    }
    if (failed(appendStructSignalBindings(defLookup->get(), tables, origin, out, path))) {
      return failure();
    }
  }
  return success();
}

/// Insert one serialized leaf into the nested JSON object at the given path.
static void insertLeafJSON(
    llvm::json::Object &root, ArrayRef<std::string> path, llvm::json::Value value
) {
  if (path.empty()) {
    return;
  }
  if (path.size() == 1) {
    root[path.front()] = std::move(value);
    return;
  }

  llvm::json::Value *slot = &root[path.front()];
  if (!slot->getAsObject()) {
    *slot = llvm::json::Object();
  }
  insertLeafJSON(*slot->getAsObject(), path.drop_front(), std::move(value));
}

} // namespace

/// Return `true` iff the member is considered a witness signal.
bool memberIsSignal(component::StructDefOp owner, component::MemberDefOp member) {
  return member.getSignal() || (owner.isMainComponent() && member.hasPublicAttr());
}

/// Collect stable JSON bindings for the main compute inputs.
llvm::SmallVector<InputBinding> collectInputBindings(function::FuncDefOp computeFunc) {
  llvm::SmallVector<InputBinding> bindings;
  bindings.reserve(computeFunc.getNumArguments());
  for (unsigned i = 0; i < computeFunc.getNumArguments(); ++i) {
    std::string name;
    if (std::optional<StringAttr> argName = computeFunc.getArgNameAttr(i)) {
      name = argName->getValue().str();
    } else {
      name = "arg" + std::to_string(i);
    }
    bindings.push_back(InputBinding {std::move(name), computeFunc.getArgumentTypes()[i], i});
  }
  return bindings;
}

/// Collect the selected output bindings for the requested scope.
FailureOr<llvm::SmallVector<OutputBinding>> collectOutputBindings(
    component::StructDefOp mainDef, SymbolTableCollection &tables, Operation *origin,
    OutputScope scope
) {
  llvm::SmallVector<OutputBinding> bindings;
  if (scope == OutputScope::Public) {
    for (component::MemberDefOp member : mainDef.getMemberDefs()) {
      if (!member.hasPublicAttr()) {
        continue;
      }
      bindings.push_back(OutputBinding {{member.getSymName().str()}, member.getType()});
    }
    return bindings;
  }

  if (failed(appendStructSignalBindings(mainDef, tables, origin, bindings))) {
    return failure();
  }
  return bindings;
}

/// Assemble a nested JSON object from selected witness leaves.
llvm::json::Value buildSignalsJSONObject(
    ArrayRef<OutputBinding> bindings, ArrayRef<llvm::json::Value> serializedLeaves
) {
  llvm::json::Object result;
  for (auto [binding, leaf] : llvm::zip(bindings, serializedLeaves)) {
    insertLeafJSON(result, binding.path, llvm::json::Value(leaf));
  }
  return llvm::json::Value(std::move(result));
}

} // namespace llzk::witgen
