//===-- LLZKValidationPasses.td - Validation Passes --------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the `-llzk-validate-member-writes` pass.
///
//===----------------------------------------------------------------------===//

#ifndef LLZK_VALIDATION_PASSES_TD
#define LLZK_VALIDATION_PASSES_TD

include "llzk/Pass/PassBase.td"

def MemberWriteValidatorPass : LLZKPass<"llzk-validate-member-writes",
                                        "::llzk::component::StructDefOp"> {
  let summary =
      "Detect multiple and missing writes to the same member of a component.";
  let constructor = "llzk::createMemberWriteValidatorPass()";
}

#endif // LLZK_VALIDATION_PASSES_TD
