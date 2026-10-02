//===-- Specialization.h - Definition specialization identity ----*- C++ -*-===//
// Part of the LLZK Project, under the Apache License v2.0.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>

#include <optional>

namespace llzk::polymorphic {

/// Module-local definition identity, independent of generated symbol spelling.
/// IDs survive serialization via poly.specialization_id. They are not instance
/// identities and must be remapped when independently specialized modules merge.
struct SpecializationId {
  uint64_t value;
  bool operator==(const SpecializationId &) const = default;
};

/// Read the identity attached by llzk-monomorphize to a concrete definition.
inline std::optional<SpecializationId> getSpecializationId(mlir::Operation *definition) {
  if (auto id = definition->getAttrOfType<mlir::IntegerAttr>("poly.specialization_id")) {
    return SpecializationId {static_cast<uint64_t>(id.getInt())};
  }
  return std::nullopt;
}

/// Read the original symbol retained for diagnostics and registry reconstruction.
inline mlir::SymbolRefAttr getSpecializationOrigin(mlir::Operation *definition) {
  return definition->getAttrOfType<mlir::SymbolRefAttr>("poly.origin");
}

/// Read the ordered concrete tuple, including TypeAttr type arguments.
inline mlir::ArrayAttr getSpecializationArguments(mlir::Operation *definition) {
  return definition->getAttrOfType<mlir::ArrayAttr>("poly.arguments");
}

} // namespace llzk::polymorphic
