//===-- ConstraintEvaluation.h - Logical circuit identities -------*- C++ -*-===//
// Part of the LLZK Project, under the Apache License v2.0.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <mlir/IR/BuiltinAttributes.h>

#include <cstdint>

namespace llzk::polymorphic {
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

/// Read an argument binding emitted in poly.signal_bindings. The parallel
/// absolute path is for diagnostics and source-storage lookup, never a WireId.
inline SignalKey getSignalKey(mlir::DictionaryAttr binding) {
  return {
      {static_cast<uint64_t>(mlir::cast<mlir::IntegerAttr>(binding.get("instance")).getInt())},
      mlir::cast<mlir::ArrayAttr>(binding.get("member_path"))
  };
}
} // namespace llzk::polymorphic
