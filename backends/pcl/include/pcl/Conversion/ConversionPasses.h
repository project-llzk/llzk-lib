//===-- ConversionPasses.h --------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Config/Config.h"
#include "llzk/Pass/PassBase.h"
#include "llzk/Transforms/Parsers.h"

namespace pcl {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "pcl/Conversion/ConversionPasses.h.inc"

} // namespace pcl::conversion
