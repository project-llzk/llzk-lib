//===-- PCL.h - C API for Picus PCL Target ------------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface to target Picus PCL.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_TARGET_PCL_H
#define MLIR_C_TARGET_PCL_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Translate an operation that satisfies the PCL dialect (i.e. previously converted with
/// the `-llzk-to-pcl` pass). The PCL lisp code is written into the provided callback.
///
/// If LLZK was compiled without the PCL backend this operation does nothing but return failure.
///
/// \returns A logical result indicating if the operation was successful or not.
MLIR_CAPI_EXPORTED MlirLogicalResult
llzkTranslateModuleToPCL(MlirOperation module, MlirStringCallback callback, void *userData);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_TARGET_PCL_H
