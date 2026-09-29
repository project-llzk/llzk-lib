//===-- Ops.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Channel/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/OpTraits.h"

#include <mlir/IR/Builders.h>

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "llzk/Dialect/Channel/IR/Ops.h.inc"
