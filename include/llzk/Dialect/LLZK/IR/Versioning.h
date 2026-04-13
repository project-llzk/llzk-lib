//===-- Versioning.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <mlir/Bytecode/BytecodeImplementation.h>

namespace llzk {

/// Temporary attribute name used during v1→v2 bytecode upgrade to carry the old
/// `const_params` value from `StructDefOp::readProperties` to `upgradeFromVersion`.
static constexpr auto kV1ConstParamsAttr = "llzk.v1_const_params";

struct LLZKDialectVersion : public mlir::DialectVersion {
  static const LLZKDialectVersion &CurrentVersion();

  static mlir::FailureOr<LLZKDialectVersion> read(mlir::DialectBytecodeReader &reader);

  LLZKDialectVersion() : LLZKDialectVersion(0, 0, 0) {}
  LLZKDialectVersion(uint64_t majorV, uint64_t minorV, uint64_t patchV)
      : majorVersion(majorV), minorVersion(minorV), patchVersion(patchV) {}

  void write(mlir::DialectBytecodeWriter &writer) const;

  std::string str() const;

  std::strong_ordering operator<=>(const LLZKDialectVersion &other) const;

  bool operator==(const LLZKDialectVersion &other) const { return std::is_eq(operator<=>(other)); }

  uint64_t majorVersion, minorVersion, patchVersion;
};

/// @brief This implements the bytecode interface for the LLZK dialect.
template <typename DialectTy>
struct LLZKDialectBytecodeInterface : public mlir::BytecodeDialectInterface {

  LLZKDialectBytecodeInterface(mlir::Dialect *dia) : mlir::BytecodeDialectInterface(dia) {}

  /// @brief Writes the current version of the LLZK-lib to the given writer.
  void writeVersion(mlir::DialectBytecodeWriter &writer) const override {
    auto versionOr = writer.getDialectVersion<DialectTy>();
    // Check if a target version to emit was specified on the writer configs.
    if (mlir::succeeded(versionOr)) {
      reinterpret_cast<const LLZKDialectVersion *>(*versionOr)->write(writer);
    } else {
      // Otherwise, write the current version
      LLZKDialectVersion::CurrentVersion().write(writer);
    }
  }

  /// @brief Read the version of this dialect from the provided reader and return it as
  /// a `unique_ptr` to a dialect version object (or nullptr on failure).
  std::unique_ptr<mlir::DialectVersion>
  readVersion(mlir::DialectBytecodeReader &reader) const override {
    auto versionOr = LLZKDialectVersion::read(reader);
    if (mlir::failed(versionOr)) {
      return nullptr;
    }
    return std::make_unique<LLZKDialectVersion>(std::move(*versionOr));
  }

  /// Hook invoked for each custom dialect after parsing is completed if a version directive was
  /// present and included an entry for the current dialect. This hook offers the opportunity for
  /// the dialect to visit the IR and upgrade constructs emitted by the provided version of the
  /// dialect to the current version.
  mlir::LogicalResult
  upgradeFromVersion(mlir::Operation *rootOp, const mlir::DialectVersion &version) const final {
    const auto &requested = static_cast<const LLZKDialectVersion &>(version);
    const auto &current = LLZKDialectVersion::CurrentVersion();
    if (requested == current) {
      return mlir::success(); // versions match, nothing to do
    }
    if (requested > current) {
      return rootOp->emitError().append(
          "Cannot upgrade from current version ", current.str(), " to future version ",
          requested.str()
      );
    }
    return upgradeFromVersion(rootOp, current, requested);
  }

  virtual mlir::LogicalResult upgradeFromVersion(
      mlir::Operation *rootOp, const LLZKDialectVersion &current,
      const LLZKDialectVersion &requested
  ) const {
    assert(requested < current && "pre-condition of upgradeFromVersion not met");
    // Default implementation does nothing. Dialects that have breaking changes between versions
    // should override this method to perform the necessary upgrade transformations.
    return mlir::success();
  }
};

} // namespace llzk
