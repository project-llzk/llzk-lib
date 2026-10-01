//===-- DialectRegistration.h -------------------------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Declares dialect and pass registration functions for core LLZK dialects.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_DIALECTREGISTRATION_H
#define LLZK_C_DIALECTREGISTRATION_H

#include <mlir-c/IR.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Registers core LLZK dialects in the given registry.
MLIR_CAPI_EXPORTED void llzkRegisterCoreDialects(MlirDialectRegistry registry);

/// Registers core LLZK passes in the given registry.
MLIR_CAPI_EXPORTED void llzkRegisterCorePasses(MlirDialectRegistry registry);

#ifdef __cplusplus
}
#endif

#endif // LLZK_C_DIALECTREGISTRATION_H
