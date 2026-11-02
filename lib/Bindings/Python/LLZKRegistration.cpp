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

#include "llzk-c/DialectRegistration.h"

#include <mlir/Bindings/Python/NanobindAdaptors.h>

NB_MODULE(_llzkRegistration, m) {
  m.doc() = "LLZK dialect and pass registration";

  m.def(
      "register_dialects", &llzkRegisterCoreDialects,
      "Register core LLZK dialects in an MLIR dialect registry."
  );
  m.def(
      "register_passes", &llzkRegisterCorePasses,
      "Register core LLZK passes and add their extensions to a dialect registry."
  );
}
