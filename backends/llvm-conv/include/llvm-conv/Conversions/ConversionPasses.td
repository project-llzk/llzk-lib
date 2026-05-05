//===-- ConversionPasses.td --------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZk
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CONV_CONVERSION_PASSES_TD
#define LLVM_CONV_CONVERSION_PASSES_TD

include "llzk/Pass/PassBase.td"

def LLVMLoweringPass : LLZKPass<"llzk-to-llvm"> {
  let summary = "Lower LLZK compute logic to the LLVM MLIR dialect.";
  let description = [{
      Transforms LLZK compute logic into the LLVM MLIR dialect.
      This pass expects all felts to have a specified prime field and for there to be a
      main entry point.
    }];
  let constructor = "llzk::llvm_conv::createLLVMLoweringPass()";
}

#endif // LLVM_CONV_CONVERSION_PASSES_TD