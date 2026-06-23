//===-- InlineIncludesPass.td ------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_INCLUDE_INLINE_PASS_TD
#define LLZK_INCLUDE_INLINE_PASS_TD

include "llzk/Pass/PassBase.td"

def InlineIncludesPass : LLZKPass<"llzk-inline-includes"> {
  let summary = "Replace all IncludeOp with contents of the referenced file.";
}

#endif // LLZK_INCLUDE_INLINE_PASS_TD
