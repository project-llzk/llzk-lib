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
/// Declares dialect and pass registration functions for core LLZK dialects and backends.
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

/// Registers core LLZK passes.
MLIR_CAPI_EXPORTED void llzkRegisterCorePasses(MlirDialectRegistry registry);

/// Registers PCL dialects in the given registry.
/// Does nothing if LLZK was compiled without the PCL backend.
MLIR_CAPI_EXPORTED void llzkRegisterPCLDialects(MlirDialectRegistry registry);

/// Registers PCL passes.
/// Does nothing if LLZK was compiled without the PCL backend.
MLIR_CAPI_EXPORTED void llzkRegisterPCLPasses(MlirDialectRegistry registry);

/// Registers R1CS dialects in the given registry.
MLIR_CAPI_EXPORTED void llzkRegisterR1CSDialects(MlirDialectRegistry registry);

/// Registers R1CS passes.
MLIR_CAPI_EXPORTED void llzkRegisterR1CSPasses(MlirDialectRegistry registry);

/// Registers the MLIR smt and LLZK smt_info metadata dialects in the given registry.
MLIR_CAPI_EXPORTED void llzkRegisterSMTDialects(MlirDialectRegistry registry);

/// Registers SMT conversion passes.
MLIR_CAPI_EXPORTED void llzkRegisterSMTPasses(MlirDialectRegistry registry);

/// Registers ZKLean dialects in the given registry.
MLIR_CAPI_EXPORTED void llzkRegisterZKLeanDialects(MlirDialectRegistry registry);

/// Registers ZKLean passes.
MLIR_CAPI_EXPORTED void llzkRegisterZKLeanPasses(MlirDialectRegistry registry);

#ifdef __cplusplus
}
#endif

#endif // LLZK_C_DIALECTREGISTRATION_H
