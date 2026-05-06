//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_CONVERSION_PASSES_TD
#define LLZK_CONVERSION_PASSES_TD

include "llzk/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// FeltToLLVM
//===----------------------------------------------------------------------===//

def FeltToLLVMConversionPass : Pass<"convert-felt-to-llvm"> {
  let summary = "Convert felt dialect to LLVM dialect";
  let description = [{
    This pass converts supported felt ops to LLVM dialect instructions.
  }];
  let dependentDialects = ["LLVM::LLVMDialect"];
  let constructor = "llzk::createFeltToLLVMConversionPass()";
}

#endif // LLZK_CONVERSION_PASSES_TD
