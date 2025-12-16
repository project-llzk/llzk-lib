//===-- Passes.h -------------------------------------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/Pass.h"

namespace llzk {

std::unique_ptr<mlir::Pass> createConvertLLZKToZKLeanPass();
std::unique_ptr<mlir::Pass> createConvertZKLeanToLLZKPass();

/// Registers all conversion passes defined in this directory.
void registerConversionPasses();

} // namespace llzk

#define GEN_PASS_DECL_CONVERTLLZKTOZKLEANPASS
#define GEN_PASS_DECL_CONVERTZKLEANTOLLZKPASS
#include "llzk/Conversions/LLZKConversionPasses.h.inc"
