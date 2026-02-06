//===-- Constants.h - LLZK constants ------------------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares string constants used by LLZK. The actual values are defined in
// llzk/Utils/Constants.h
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_CONSTANTS_H
#define LLZK_C_CONSTANTS_H

#ifdef __cplusplus
extern "C" {
#endif

/// Symbol name for the witness generation (and resp. constraint generation) functions within a
/// component.
extern const char *LLZK_FUNC_NAME_COMPUTE;
extern const char *LLZK_FUNC_NAME_CONSTRAIN;

/// Name of the attribute on the top-level ModuleOp that identifies the ModuleOp as the
/// root module and specifies the frontend language name that the IR was compiled from, if
/// available.
extern const char *LLZK_LANG_ATTR_NAME;

/// Name of the attribute on the top-level ModuleOp that defines prime fields
/// used in the circuit.
extern const char *FIELD_ATTR_NAME;

/// Name of the attribute on the top-level ModuleOp that specifies the type of the main struct.
/// This attribute can appear zero or one times on the top-level ModuleOp and is associated with
/// a `TypeAttr` specifying the `StructType` of the main struct.
extern const char *LLZK_MAIN_ATTR_NAME;

#ifdef __cplusplus
}
#endif

#endif
