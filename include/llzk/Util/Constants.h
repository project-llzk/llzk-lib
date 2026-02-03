//===-- Constants.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

namespace llzk {

/// Symbol name for the struct/component representing a signal. A "signal" has direct correspondence
/// to a circom signal or AIR/PLONK column, opposed to intermediate values or other expressions.
constexpr char COMPONENT_NAME_SIGNAL[] = "Signal";

/// Symbol name for the main entry point struct/component (if any). There are additional
/// restrictions on the struct with this name:
/// 1. It cannot have struct parameters.
/// 2. The parameter types of its functions (besides the required "self" parameter) can
///     only be `struct<Signal>` or `array<.. x struct<Signal>>`.
constexpr char COMPONENT_NAME_MAIN[] = "Main";

/// Symbol name for the witness generation (and resp. constraint generation) functions within a
/// component.
constexpr char FUNC_NAME_COMPUTE[] = "compute";
constexpr char FUNC_NAME_CONSTRAIN[] = "constrain";
constexpr char FUNC_NAME_PRODUCT[] = "product";

/// Name of the attribute on the top-level ModuleOp that specifies the IR language name.
constexpr char LANG_ATTR_NAME[] = "veridise.lang";

/// Name of the attribute on aligned product program ops that specifies where they came from.
constexpr char PRODUCT_SOURCE[] = "product_source";

} // namespace llzk
