//===-- Errors.h - llzk-witgen error helpers --------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/Twine.h>
#include <llvm/Support/Error.h>

namespace llzk::witgen {

/// Build a string-backed error for user-facing witgen failures.
inline llvm::Error makeError(const llvm::Twine &msg) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), msg.str());
}

} // namespace llzk::witgen
