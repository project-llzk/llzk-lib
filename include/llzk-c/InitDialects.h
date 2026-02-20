//===-- InitDialects.h - C API for dialect registration -----------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares the registration method for
// all LLZK dialects.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_INITDIALECTS_H
#define LLZK_C_INITDIALECTS_H

#include <mlir-c/IR.h>

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void llzkRegisterAllDialects(MlirDialectRegistry registry);

#ifdef __cplusplus
}
#endif

#endif // LLZK_C_INITDIALECTS_H
