//===-- FeltToLLVM.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements conversions from the `felt` dialect to the LLVM
/// IR dialect.
///
//===----------------------------------------------------------------------===//

#include "llzk/Conversions/FeltToLLVM.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"

#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/VectorPattern.h>

using namespace mlir;

//===----------------------------------------------------------------------===//
// ConvertToLLVMPatternInterface implementation
//===----------------------------------------------------------------------===//

struct ArithToLLVMDialectInterface : public ConvertToLLVMPatternInterface {};
