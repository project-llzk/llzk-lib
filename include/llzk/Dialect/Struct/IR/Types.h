//===-- Types.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Struct/IR/Dialect.h"
#include "llzk/Util/SymbolLookup.h"
#include "llzk/Util/TypeHelper.h"

// Forward-declare ops since StructDefOp is used within type definitions
#include "llzk/Dialect/Struct/IR/Ops.h.inc"

// Include TableGen'd declarations
#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/Struct/IR/Types.h.inc"
