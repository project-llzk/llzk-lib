//===-- Ops.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/OpTraits.h"

// MLIR interfaces used by generated ops
#include <mlir/Interfaces/InferTypeOpInterface.h>

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "llzk/Dialect/Cast/IR/Ops.h.inc"
