//===-- Ops.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Shared/OpHelpers.h"
#include "llzk/Dialect/Verif/IR/Dialect.h"
#include "llzk/Util/TypeHelper.h"
#include "llzk/Dialect/Verif/IR/OpInterfaces.h"
#include "llzk/Dialect/Felt/IR/Types.h"

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "llzk/Dialect/Verif/IR/Ops.h.inc"
