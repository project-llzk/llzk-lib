//===-- Ops.cpp - RAM operation implementations -----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/RAM/IR/Ops.h"

#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/Builders.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/RAM/IR/Ops.cpp.inc"
