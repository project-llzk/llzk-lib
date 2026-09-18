//===-- OverflowSemantics.h - Cast overflow helpers -------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Cast/IR/Enums.h"
#include "llzk/Util/Field.h"

#include <llvm/ADT/DynamicAPInt.h>

#include <optional>

namespace llzk::cast {

/// Apply overflow semantics for an integer-to-felt conversion. Returns nullopt
/// when assert semantics reject the input.
std::optional<llvm::DynamicAPInt> applyIntToFeltOverflow(
    const llvm::DynamicAPInt &value, const Field &field, OverflowSemantics overflow
);

/// Apply overflow semantics for a felt-to-index conversion. `indexBitWidth`
/// is the storage width; results are always in its nonnegative safe range.
/// Returns nullopt when assert semantics reject the input.
std::optional<llvm::DynamicAPInt> applyFeltToIndexOverflow(
    const llvm::DynamicAPInt &value, unsigned indexBitWidth, OverflowSemantics overflow
);

} // namespace llzk::cast
