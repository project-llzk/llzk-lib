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

/// Symbol name for the witness generation (and resp. constraint generation) functions within a
/// component.
constexpr char FUNC_NAME_COMPUTE[] = "compute";
constexpr char FUNC_NAME_CONSTRAIN[] = "constrain";
constexpr char FUNC_NAME_PRODUCT[] = "product";

/// Name of the attribute on the top-level ModuleOp that identifies the ModuleOp as the
/// root module and specifies the frontend language name that the IR was compiled from, if
/// available.
constexpr char LANG_ATTR_NAME[] = "llzk.lang";

/// Name of the attribute on the top-level ModuleOp that specifies the type of the main struct.
/// This attribute can appear zero or one times on the top-level ModuleOp and is associated with
/// a `TypeAttr` specifying the `StructType` of the main struct.
constexpr char MAIN_ATTR_NAME[] = "llzk.main";

} // namespace llzk
