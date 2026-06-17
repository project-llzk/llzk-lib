//===-- SMTPasses.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Pass/Pass.h>

#include <memory>

namespace llzk::smt {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "smt/Transforms/SMTPasses.h.inc"

} // namespace llzk::smt
