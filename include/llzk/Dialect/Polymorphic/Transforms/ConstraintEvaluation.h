//===-- ConstraintEvaluation.h - Logical circuit identities ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/STLExtras.h>

#include <cstdint>

namespace llzk::polymorphic {

/// Evaluation state and provenance shared by lowering and witness generation.
constexpr char EVALUATED_MAIN_ATTR_NAME[] = "poly.evaluated_main";
constexpr char EVALUATED_ATTR_NAME[] = "poly.evaluated";
constexpr char ORIGINAL_PUBLIC_ATTR_NAME[] = "poly.original_public";
constexpr char SIGNAL_BINDING_ATTR_NAME[] = "poly.signal_binding";
constexpr char SIGNAL_BINDINGS_ATTR_NAME[] = "poly.signal_bindings";
constexpr char INSTANCES_ATTR_NAME[] = "poly.instances";
constexpr char DEGREE_LOWERED_ATTR_NAME[] = "poly.degree_lowered";

/// The module marker is authoritative; per-function markers describe provenance.
inline bool isEvaluatedModule(mlir::ModuleOp module) {
  return module && module->hasAttr(EVALUATED_MAIN_ATTR_NAME);
}

/// Read circuit visibility before evaluation relaxed nested member access.
inline bool isOriginallyPublic(mlir::Operation *member) {
  auto original = member->getAttrOfType<mlir::BoolAttr>(ORIGINAL_PUBLIC_ATTR_NAME);
  return original ? original.getValue() : member->hasAttr("llzk.pub");
}

/// Common storage location and visibility carried by signal and wire bindings.
struct StorageBinding {
  mlir::ArrayAttr path;
  bool isPublic;
};

/// Decode serialized storage metadata without assuming that input IR is trusted.
/// The root is a non-negative constrain argument number; subsequent elements are
/// member names or non-negative array indices.
inline mlir::FailureOr<StorageBinding> getStorageBinding(mlir::Attribute attr) {
  auto binding = mlir::dyn_cast_if_present<mlir::DictionaryAttr>(attr);
  if (!binding) {
    return mlir::failure();
  }
  auto path = binding.getAs<mlir::ArrayAttr>("path");
  auto isPublic = binding.getAs<mlir::BoolAttr>("public");
  if (!path || path.empty() || !isPublic) {
    return mlir::failure();
  }
  for (auto [index, segment] : llvm::enumerate(path)) {
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(segment)) {
      if (integer.getValue().isNegative() || integer.getValue().getActiveBits() > 63) {
        return mlir::failure();
      }
    } else if (index == 0 || !mlir::isa<mlir::StringAttr>(segment)) {
      return mlir::failure();
    }
  }
  return StorageBinding {path, isPublic.getValue()};
}

/// Module-local component occurrence identity. A specialization may have many
/// instances. Serialized in poly.instances; independent of physical wire layout.
struct InstanceId {
  uint64_t value;
  bool operator==(const InstanceId &) const = default;
};

/// Interned logical scalar location. Path elements are StringAttr member/record
/// symbols and IntegerAttr array indices, relative to the owning instance.
/// The root input argument number is the first element for main inputs.
struct SignalKey {
  InstanceId instance;
  mlir::ArrayAttr memberPath;
  bool operator==(const SignalKey &) const = default;
};

/// Read a storage binding emitted in poly.signal_bindings and on scalar reads. The parallel
/// absolute path is for diagnostics and source-storage lookup, never a WireId.
inline SignalKey getSignalKey(mlir::DictionaryAttr binding) {
  return {
      {static_cast<uint64_t>(mlir::cast<mlir::IntegerAttr>(binding.get("instance")).getInt())},
      mlir::cast<mlir::ArrayAttr>(binding.get("member_path"))
  };
}
} // namespace llzk::polymorphic
