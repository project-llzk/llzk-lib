//===-- EffectHelper.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/ADT/STLExtras.h>

namespace llzk {

/// Returns true when \p op has a memory read effect. Unknown effects are
/// handled separately by hasUnknownOrNonReadEffect().
inline bool hasReadEffect(mlir::Operation *op) {
  auto effects = mlir::getEffectsRecursively(op);
  return effects && llvm::any_of(*effects, [](const mlir::MemoryEffects::EffectInstance &effect) {
    return llvm::isa<mlir::MemoryEffects::Read>(effect.getEffect());
  });
}

/// Returns true when \p op may have an unknown effect or any effect other than
/// memory read.
inline bool hasUnknownOrNonReadEffect(mlir::Operation *op) {
  auto effects = mlir::getEffectsRecursively(op);
  return !effects || llvm::any_of(*effects, [](const mlir::MemoryEffects::EffectInstance &effect) {
    return !llvm::isa<mlir::MemoryEffects::Read>(effect.getEffect());
  });
}

} // namespace llzk
