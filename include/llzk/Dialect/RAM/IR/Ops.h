//===-- Ops.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/OpTraits.h"
#include "llzk/Dialect/RAM/IR/Dialect.h"

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "llzk/Dialect/RAM/IR/Ops.h.inc"
