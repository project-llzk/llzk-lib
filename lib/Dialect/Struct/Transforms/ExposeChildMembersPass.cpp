//===-- ExposeChildMembersPass.cpp -----------------------------------------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Struct/Transforms/ExposeChildMembersPass.h"

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/Struct/Transforms/TransformationPasses.h"
#include "llzk/Util/SymbolHelper.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>

namespace llzk::component {
#define GEN_PASS_DEF_EXPOSECHILDMEMBERSPASS
#include "llzk/Dialect/Struct/Transforms/TransformationPasses.h.inc"
} // namespace llzk::component

using namespace mlir;
using namespace llzk;
using namespace llzk::component;
using namespace llzk::polymorphic;

namespace {
/// Collect child storage definitions before changing their member visibility.
class ChildStorageVisitor {
  /// Caches symbol tables while resolving child types and family source symbols.
  SymbolTableCollection tables;
  /// Outermost LLZK module: specialization IDs and poly.origin paths use this scope.
  ModuleOp rootModule;
  /// Selected traversal root; its own member attributes must remain unchanged.
  StructDefOp storageRoot;
  /// Whether the module-wide specialization-ID index has been built lazily.
  bool specializationsIndexed = false;
  /// All definitions for each valid ID, retaining duplicates to diagnose referenced ambiguity.
  llvm::DenseMap<int64_t, SmallVector<StructDefOp, 1>> specializations;
  /// Definitions already traversed, avoiding repeated work on shared or recursive storage.
  llvm::DenseSet<Operation *> visited;
  /// Reachable children to expose only after the entire traversal succeeds.
  SmallVector<StructDefOp> children;

public:
  ChildStorageVisitor(ModuleOp module, StructDefOp root) : rootModule(module), storageRoot(root) {}

  /// Expose collected children only after traversal succeeds.
  LogicalResult run() {
    if (failed(visitStruct(storageRoot))) {
      return failure();
    }
    for (auto child : children) {
      for (auto member : child.getMemberDefs()) {
        member.setPublicAttr();
      }
    }
    return success();
  }

private:
  /// Visit each concrete storage definition once, preserving the selected root itself.
  LogicalResult visitStruct(StructDefOp definition) {
    if (isInTemplate(definition)) {
      return definition.emitError("child-member visibility requires monomorphized struct types");
    }
    if (!visited.insert(definition).second) {
      return success();
    }
    if (definition != storageRoot) {
      children.push_back(definition);
    }
    for (auto member : definition.getMemberDefs()) {
      if (failed(visitType(member.getType(), member))) {
        return failure();
      }
    }
    return success();
  }

  /// Visit the possible concrete definitions for a rolled storage type. Family
  /// coordinates are irrelevant to visibility; repeated IDs are resolved once.
  LogicalResult visitFamily(StructType structure, Operation *site) {
    Attribute metadata = site->getAttr(FAMILY_ATTR);
    if (!metadata) {
      return site->emitError("missing concrete struct family element for child-member visibility");
    }
    auto records = dyn_cast<ArrayAttr>(metadata);
    if (!records) {
      return site->emitError("invalid poly.family metadata for child-member visibility");
    }
    llvm::DenseSet<int64_t> seenIds;
    for (Attribute record : records) {
      auto dict = dyn_cast<DictionaryAttr>(record);
      auto type = dict ? dict.getAs<TypeAttr>(FAMILY_TYPE_KEY) : TypeAttr();
      if (!type || !isa<StructType>(type.getValue())) {
        return site->emitError("invalid poly.family entry for child-member visibility");
      }
      if (type.getValue() != structure) {
        continue;
      }
      auto id = dict.getAs<IntegerAttr>(FAMILY_SPECIALIZATION_KEY);
      if (!id || !id.getValue().isSignedIntN(64) || id.getInt() < 0) {
        return site->emitError("invalid poly.family entry for child-member visibility");
      }
      if (!seenIds.insert(id.getInt()).second) {
        continue;
      }
      auto definition = resolveSpecialization(structure, id.getInt(), site);
      if (failed(definition) || failed(visitStruct(*definition))) {
        return failure();
      }
    }
    if (seenIds.empty()) {
      return site->emitError("missing concrete struct family element for child-member visibility");
    }
    return success();
  }

  /// Resolve only family specializations needed by the visited storage. Ignore
  /// malformed IDs on unrelated definitions; duplicate IDs matter only when used.
  FailureOr<llzk::component::StructDefOp>
  resolveSpecialization(llzk::component::StructType type, int64_t id, Operation *site) {
    if (!specializationsIndexed) {
      rootModule.walk([this](llzk::component::StructDefOp definition) {
        auto metadata = definition->getAttrOfType<IntegerAttr>(SPECIALIZATION_ID_ATTR);
        if (metadata && metadata.getValue().isSignedIntN(64) && metadata.getInt() >= 0) {
          specializations[metadata.getInt()].push_back(definition);
        }
      });
      specializationsIndexed = true;
    }
    auto target = specializations.find(id);
    if (target == specializations.end()) {
      return site->emitError("unknown struct specialization in child-member visibility family");
    }
    if (target->second.size() != 1) {
      return site->emitError("ambiguous struct specialization in child-member visibility family");
    }
    auto definition = target->second.front();
    auto source = type.getDefinition(tables, site);
    if (failed(source)) {
      return failure();
    }
    if (!*source || source->viaInclude()) {
      return site->emitError(
          "cannot resolve child-member visibility family source; inline includes first"
      );
    }
    auto origin = llzk::getPathRelativeToAncestor(source->get(), rootModule);
    if (failed(origin) || definition->getAttr(SPECIALIZATION_ORIGIN_ATTR) != *origin) {
      return site->emitError(
          "struct specialization origin does not match child-member visibility family"
      );
    }
    return definition;
  }

  /// Follow storage wrappers and resolve concrete or rolled component types.
  LogicalResult visitType(Type type, Operation *site) {
    bool containsStruct = false;
    type.walk([&](StructType) { containsStruct = true; });
    if (!containsStruct) {
      return success();
    }
    if (auto array = dyn_cast<array::ArrayType>(type)) {
      for (int64_t size : array.getShape()) {
        if (size < 0) {
          return site->emitError("child-member visibility requires concrete array dimensions");
        }
      }
      if (llvm::is_contained(array.getShape(), 0)) {
        return success();
      }
      return visitType(array.getElementType(), site);
    }
    if (auto pod = dyn_cast<pod::PodType>(type)) {
      for (auto record : pod.getRecords()) {
        if (failed(visitType(record.getType(), site))) {
          return failure();
        }
      }
      return success();
    }
    auto structure = cast<StructType>(type);
    if (auto params = structure.getParams(); params && !params.empty()) {
      return visitFamily(structure, site);
    }
    auto definition = structure.getDefinition(tables, site);
    if (failed(definition)) {
      return failure();
    }
    if (!*definition || definition->viaInclude()) {
      return site->emitError("cannot resolve child struct storage; inline includes first");
    }
    return visitStruct(definition->get());
  }
};

/// Resolve the selected root and delegate the transformation to the reusable utility.
class ExposeChildMembersPass
    : public llzk::component::impl::ExposeChildMembersPassBase<ExposeChildMembersPass> {
  using Base = llzk::component::impl::ExposeChildMembersPassBase<ExposeChildMembersPass>;
  using Base::Base;

  void runOnOperation() override {
    SymbolTableCollection tables;
    auto root = getRootStructDef(tables, getOperation(), rootStruct);
    if (failed(root)) {
      signalPassFailure();
      return;
    }
    if (!*root || root->viaInclude()) {
      getOperation().emitError(
          "child-member visibility requires a concrete root struct; inline includes first"
      );
      signalPassFailure();
      return;
    }
    if (failed(exposeChildMembers(root->get()))) {
      signalPassFailure();
    }
  }
};
} // namespace

LogicalResult llzk::component::exposeChildMembers(StructDefOp root) {
  auto module = getTopRootModule(root);
  if (failed(module)) {
    return failure();
  }
  return ChildStorageVisitor(*module, root).run();
}
