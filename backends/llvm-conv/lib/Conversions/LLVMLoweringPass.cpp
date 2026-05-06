//===-- LLVMLoweringPass.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-to-llvm` pass.
///
//===----------------------------------------------------------------------===//

#include "llvm-conv/Conversions/ConversionPasses.h"

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Bool/IR/Enums.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/Include/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/String/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/Field.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

#include <algorithm>
#include <utility>

namespace llzk {

namespace llvm_conv {
#define GEN_PASS_DECL_LLVMLOWERINGPASS
#define GEN_PASS_DEF_LLVMLOWERINGPASS
#include "llvm-conv/Conversions/ConversionPasses.h.inc"
} // namespace llvm_conv

using namespace mlir;

class LLVMLoweringPass : public llvm_conv::impl::LLVMLoweringPassBase<LLVMLoweringPass> {
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override { llvm::errs() << "hello!\n"; }
};

namespace llvm_conv {
std::unique_ptr<mlir::Pass> createLLVMLoweringPass() {
  return std::make_unique<LLVMLoweringPass>();
}
} // namespace llvm_conv

} // namespace llzk
