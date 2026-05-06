//===- FeltToLLVM.h - felt to LLVM dialect conversion -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_CONVERSION_FELTTOLLVM_H
#define LLZK_CONVERSION_FELTTOLLVM_H

#include <memory>

namespace llzk {

class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_FELTTOLLVMCONVERSIONPASS
#include "llzk/Conversions/Passes.h.inc"

namespace felt {
void populateArithToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns
);

void registerConvertArithToLLVMInterface(DialectRegistry &registry);
} // namespace felt
} // namespace llzk

#endif // LLZK_CONVERSION_FELTTOLLVM_H
