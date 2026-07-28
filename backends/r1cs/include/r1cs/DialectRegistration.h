//===-- DialectRegistration.h -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/DialectRegistry.h>

namespace r1cs {
void registerAllDialects(mlir::DialectRegistry &registry);
}
