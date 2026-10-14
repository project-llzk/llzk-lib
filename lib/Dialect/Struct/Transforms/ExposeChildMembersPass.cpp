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
#include "llzk/Dialect/Polymorphic/Util/StructSpecialization.h"
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
  /// Selected traversal root; its own member attributes must remain unchanged.
  StructDefOp storageRoot;
  /// Resolve family IDs in the outermost module without indexing unused families.
  StructSpecializationResolver specializations;
  /// Definitions already traversed, avoiding repeated work on shared or recursive storage.
  llvm::DenseSet<Operation *> visited;
  /// Reachable children to expose only after the entire traversal succeeds.
  SmallVector<StructDefOp> children;
  /// Rolled declarations needed by reads through unchanged family handles.
  llvm::DenseSet<Operation *> rolledMembers;

public:
  ChildStorageVisitor(ModuleOp module, StructDefOp root)
      : storageRoot(root), specializations(module) {}

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
    for (Operation *member : rolledMembers) {
      cast<MemberDefOp>(member).setPublicAttr();
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

  /// Legalize matching declarations without traversing or instantiating templates.
  void collectRolledMembers(StructDefOp rolled, StructDefOp concrete) {
    if (rolled == storageRoot) {
      return;
    }
    for (auto member : concrete.getMemberDefs()) {
      if (auto declaration = rolled.getMemberDef(member.getSymNameAttr())) {
        rolledMembers.insert(declaration);
      }
    }
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
    auto rolled = structure.getDefinition(tables, site);
    if (failed(rolled)) {
      return failure();
    }
    if (!*rolled || rolled->viaInclude()) {
      return site->emitError(
          "cannot resolve child-member visibility family source; inline includes first"
      );
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
      auto definition =
          specializations.resolve(tables, structure, id.getInt(), site, "child-member visibility");
      if (failed(definition) || failed(visitStruct(*definition))) {
        return failure();
      }
      collectRolledMembers(rolled->get(), *definition);
    }
    if (seenIds.empty()) {
      return site->emitError("missing concrete struct family element for child-member visibility");
    }
    return success();
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
class PassImpl : public llzk::component::impl::ExposeChildMembersPassBase<PassImpl> {
  using Base = ExposeChildMembersPassBase<PassImpl>;
  using Base::Base;

  void runOnOperation() override {
    SymbolTableCollection tables;
    auto root = getRootStructDef(tables, getOperation(), rootStruct);
    if (failed(root)) {
      signalPassFailure();
      return;
    }
    if (!*root) {
      getOperation().emitError(
          "child-member visibility requires llzk.main or an explicit root-struct"
      );
      signalPassFailure();
      return;
    }
    if (root->viaInclude()) {
      getOperation().emitError(
          "child-member visibility requires a local root struct; inline includes first"
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
