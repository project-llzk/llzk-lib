//===-- LLZKRegistration.cpp - LLZK Python Registration ---------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This defines a module that can be used to register the LLZK dialects.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/DialectRegistration.h"

#include <mlir/Bindings/Python/PybindAdaptors.h>
#include <mlir/CAPI/IR.h>

PYBIND11_MODULE(_llzkRegistration, m) {
  m.doc() = "LLZK dialect registration";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    llzk::registerDialects(*unwrap(registry));
  });
}
