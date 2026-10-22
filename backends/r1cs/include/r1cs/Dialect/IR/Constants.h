//===-- Constants.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/StringRef.h>

#include <cstdint>

namespace r1cs {

/// Wire zero represents the constant one in the R1CS binary format.
inline constexpr std::uint32_t CONSTANT_ONE_WIRE_ID = 0;
/// First ID available to declared signals and circuit inputs.
inline constexpr std::uint32_t FIRST_SIGNAL_WIRE_ID = CONSTANT_ONE_WIRE_ID + 1;

/// Wire categories in the R1CS binary format.
enum class WireKind : std::uint8_t { PublicOutput, PublicInput, PrivateInput, Internal };

/// Required order of nonconstant wire categories in the binary format.
inline constexpr WireKind WIRE_KIND_ORDER[] = {
    WireKind::PublicOutput, WireKind::PublicInput, WireKind::PrivateInput, WireKind::Internal
};

/// Explicit final wire IDs for circuit block arguments, in block-argument order.
inline constexpr llvm::StringLiteral INPUT_LABELS_ATTR = "r1cs.input_labels";

} // namespace r1cs
