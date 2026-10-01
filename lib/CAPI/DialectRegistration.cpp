//===-- DialectRegistration.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines dialect and pass registration functions for core LLZK dialects and backends.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/DialectRegistration.h"

#include "r1cs/DialectRegistration.h"
#include "zklean/DialectRegistration.h"

#include "llzk/Config/Config.h"

#if LLZK_WITH_PCL
#include "pcl/DialectRegistration.h"
#endif

#include "llzk-c/DialectRegistration.h"

#include <mlir-c/IR.h>

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Wrap.h>

void llzkRegisterCoreDialects(MlirDialectRegistry registry) {
  llzk::registerDialects(*unwrap(registry));
}

void llzkRegisterCorePasses(MlirDialectRegistry registry) {
  llzk::registerPasses(*unwrap(registry));
}

void llzkRegisterPCLDialects(MlirDialectRegistry registry) {
#if LLZK_WITH_PCL
  pcl::registerDialects(*unwrap(registry));
#else
  (void)registry;
#endif
}

void llzkRegisterPCLPasses(MlirDialectRegistry registry) {
#if LLZK_WITH_PCL
  pcl::registerPasses(*unwrap(registry));
#else
  (void)registry;
#endif
}

void llzkRegisterR1CSDialects(MlirDialectRegistry registry) {
  r1cs::registerDialects(*unwrap(registry));
}

void llzkRegisterR1CSPasses(MlirDialectRegistry registry) {
  r1cs::registerPasses(*unwrap(registry));
}

void llzkRegisterZKLeanDialects(MlirDialectRegistry registry) {
  zklean::registerDialects(*unwrap(registry));
}

void llzkRegisterZKLeanPasses(MlirDialectRegistry registry) {
  zklean::registerPasses(*unwrap(registry));
}
