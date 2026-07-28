//===-- Ops.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "llzk/Dialect/LLZK/IR/Ops.h.inc"
