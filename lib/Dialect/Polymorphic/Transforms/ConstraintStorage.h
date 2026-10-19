//===-- ConstraintStorage.h -------------------------------------*- C++ -*-===//
// Part of the LLZK Project, under the Apache License v2.0.
// SPDX-License-Identifier: Apache-2.0

#ifndef LLZK_CONSTRAINT_STORAGE_H
#define LLZK_CONSTRAINT_STORAGE_H

#include "ConstraintScalarFacts.h"

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/Util/StructSpecialization.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/TypeHelper.h"

namespace llzk::polymorphic::detail {
/// Transient description derived from immutable storage accesses, not an environment.
struct StorageDescription {
  component::StructDefOp definition;
  mlir::ArrayAttr family;
  llvm::SmallVector<int64_t> indices;
  bool signal = false;
};

/// Resolve concrete components and signal provenance directly from original SSA.
class ConstraintStorage {
  component::StructDefOp main;
  mlir::OpBuilder &builder;
  mlir::SymbolTableCollection &tables;
  ConstraintScalarFacts &facts;
  StructSpecializationResolver specializations;

  static mlir::LogicalResult error(mlir::Operation *site, const llvm::Twine &message) {
    return site->emitError("constraint flattening: ") << message;
  }

  /// Resolve a rolled family handle using concrete storage coordinates.
  mlir::LogicalResult
  resolveFamily(component::StructType type, StorageDescription &info, mlir::Operation *site) {
    for (mlir::Attribute record : info.family) {
      auto entry = llvm::dyn_cast<mlir::DictionaryAttr>(record);
      auto origin = entry ? entry.getAs<mlir::TypeAttr>(FAMILY_TYPE_KEY) : mlir::TypeAttr();
      auto indices = entry ? entry.getAs<mlir::ArrayAttr>(FAMILY_INDICES_KEY) : mlir::ArrayAttr();
      if (!origin || origin.getValue() != type || !indices ||
          indices.size() != info.indices.size()) {
        continue;
      }
      bool matches = llvm::all_of(llvm::zip(indices, info.indices), [](auto pair) {
        auto attr = llvm::dyn_cast<mlir::IntegerAttr>(std::get<0>(pair));
        return attr && attr.getValue().isSignedIntN(64) && attr.getInt() == std::get<1>(pair);
      });
      if (!matches) {
        continue;
      }
      auto id = entry.getAs<mlir::IntegerAttr>(FAMILY_SPECIALIZATION_KEY);
      if (!id || !id.getValue().isSignedIntN(64) || id.getInt() < 0) {
        return error(site, "invalid family specialization ID");
      }
      auto target =
          specializations.resolve(tables, type, id.getInt(), site, "constraint flattening");
      if (mlir::failed(target)) {
        return mlir::failure();
      }
      if (info.definition && info.definition != *target) {
        return error(site, "ambiguous concrete struct family element");
      }
      info.definition = *target;
    }
    return mlir::success();
  }

  /// Resolve concrete component handles without interpreting their contents.
  mlir::LogicalResult
  resolveComponent(mlir::Value value, StorageDescription &info, mlir::Operation *site) {
    auto type = llvm::dyn_cast<component::StructType>(value.getType());
    if (!type) {
      return mlir::success();
    }
    if (info.family && mlir::failed(resolveFamily(type, info, site))) {
      return mlir::failure();
    }
    if (info.definition) {
      return mlir::success();
    }
    auto target = lookupTopLevelSymbol<component::StructDefOp>(tables, type.getNameRef(), site);
    if (mlir::failed(target)) {
      return mlir::failure();
    }
    if (isInTemplate(target->get()) || (type.getParams() && !type.getParams().empty())) {
      return error(site, "requires a concrete struct or matching poly.family element");
    }
    info.definition = target->get();
    return mlir::success();
  }

  /// Describe a member read using its actual concrete component definition.
  mlir::LogicalResult
  memberInfo(component::MemberReadOp read, StorageDescription &info, mlir::Operation *site) {
    auto resolved = describe(read.getComponent(), site);
    if (mlir::failed(resolved)) {
      return mlir::failure();
    }
    auto parent = *resolved;
    if (!parent.definition || read.getTableOffset()) {
      return error(site, "member read requires concrete non-column component storage");
    }
    auto member = parent.definition.getMemberDef(builder.getStringAttr(read.getMemberName()));
    if (!member || member.getColumn()) {
      return error(site, "unknown or unsupported column member");
    }
    if (parent.definition != main && !member.hasPublicAttr()) {
      return error(site, "run llzk-expose-child-members before reading child storage");
    }
    auto rolled = lookupTopLevelSymbol<component::StructDefOp>(
        tables, llvm::cast<component::StructType>(read.getComponent().getType()).getNameRef(), site
    );
    if (mlir::failed(rolled)) {
      return mlir::failure();
    }
    if (!typesUnify(read.getType(), member.getType())) {
      return error(site, "member read type does not match the concrete specialization");
    }
    auto declared = rolled->get().getMemberDef(builder.getStringAttr(read.getMemberName()));
    if (rolled->get() != parent.definition &&
        (!declared || !declared.hasPublicAttr() ||
         !typesUnify(read.getType(), declared.getType(), rolled->getNamespace()))) {
      return error(
          site, "concrete family member cannot be represented through the rolled storage type"
      );
    }
    info.family = member->getAttrOfType<mlir::ArrayAttr>(FAMILY_ATTR);
    info.signal = bool(member.getSignal()) || (parent.definition == main && member.hasPublicAttr());
    return mlir::success();
  }

public:
  ConstraintStorage(
      mlir::ModuleOp rootModule, component::StructDefOp mainDefinition,
      mlir::OpBuilder &outputBuilder, mlir::SymbolTableCollection &symbolTables,
      ConstraintScalarFacts &scalarFacts
  )
      : main(mainDefinition), builder(outputBuilder), tables(symbolTables), facts(scalarFacts),
        specializations(rootModule) {}

  /// Resolve component metadata from immutable emitted storage access chains.
  /// Local bindings already resolve to these scalar values, so no parallel
  /// storage-provenance environment is needed.
  mlir::FailureOr<StorageDescription> describe(mlir::Value value, mlir::Operation *site) {
    StorageDescription info;
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value)) {
      info.definition = argument.getArgNumber() == 0 ? main : component::StructDefOp();
      info.signal = argument.getArgNumber() != 0;
    } else if (auto member = value.getDefiningOp<component::MemberReadOp>()) {
      if (mlir::failed(memberInfo(member, info, site))) {
        return mlir::failure();
      }
    } else if (auto element = value.getDefiningOp<array::ReadArrayOp>()) {
      auto parent = describe(element.getArrRef(), site);
      if (mlir::failed(parent)) {
        return mlir::failure();
      }
      info = *parent;
      info.definition = {};
      if (info.family) {
        for (mlir::Value index : element.getIndices()) {
          auto coordinate = facts.integer(index);
          if (!coordinate) {
            return error(site, "array path requires compile-time indices");
          }
          info.indices.push_back(*coordinate);
        }
      }
    } else if (auto record = value.getDefiningOp<pod::ReadPodOp>()) {
      auto parent = describe(record.getPodRef(), site);
      if (mlir::failed(parent)) {
        return mlir::failure();
      }
      info = *parent;
      info.definition = {};
      if (llvm::isa<array::ArrayType>(value.getType())) {
        // Discovery records family coordinates relative to the nearest array.
        info.indices.clear();
      }
    }
    if (mlir::failed(resolveComponent(value, info, site))) {
      return mlir::failure();
    }
    return info;
  }
};

} // namespace llzk::polymorphic::detail
#endif
