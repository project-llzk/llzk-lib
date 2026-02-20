//===-- LLZKInliningExtensions.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Dialect.h"
#include "llzk/Dialect/Bool/IR/Dialect.h"
#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Dialect.h"
#include "llzk/Dialect/Include/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/POD/IR/Dialect.h"
#include "llzk/Dialect/Polymorphic/IR/Dialect.h"
#include "llzk/Dialect/String/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Dialect.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Transforms/InliningUtils.h>

using namespace mlir;
using namespace llzk;

namespace {

template <typename InlinerImpl, typename DialectImpl, typename... RequiredDialects>
struct BaseInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  static void registrationHook(MLIRContext *ctx, DialectImpl *dialect) {
    dialect->template addInterfaces<InlinerImpl>();
    if constexpr (sizeof...(RequiredDialects) != 0) {
      ctx->loadDialect<RequiredDialects...>();
    }
  }
};

// Adapted from `mlir/lib/Dialect/Func/Extensions/InlinerExtension.cpp`
struct FuncInlinerInterface
    : public BaseInlinerInterface<
          FuncInlinerInterface, function::FunctionDialect, cf::ControlFlowDialect> {
  using BaseInlinerInterface::BaseInlinerInterface;

  /// All calls, operations, and functions can be inlined.
  bool isLegalToInline(Operation *, Operation *, bool) const final { return true; }
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final { return true; }
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final { return true; }

  void handleTerminator(Operation *op, Block *newDest) const final {
    // Only return needs to be handled here. Replace the return with a branch to the dest.
    // Note: This function is only called when there are multiple blocks in the region being
    // inlined. In LLZK IR, that would only occur when the `cf` dialect is already used (since no
    // LLZK dialect defines any kind of cross-block branching ops) so it's fine to add a
    // `cf::BranchOp` here.
    if (auto returnOp = llvm::dyn_cast<function::ReturnOp>(op)) {
      OpBuilder builder(op);
      builder.create<cf::BranchOp>(op->getLoc(), newDest, returnOp.getOperands());
      op->erase();
    }
  }

  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    // ASSERT: when region contains a single block, terminator must be ReturnOp
    assert(llvm::isa<function::ReturnOp>(op));

    // Replace the values directly with the return operands.
    auto returnOp = llvm::cast<function::ReturnOp>(op);
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands())) {
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }
  }
};

template <typename DialectImpl>
struct FullyLegalForInlining
    : public BaseInlinerInterface<FullyLegalForInlining<DialectImpl>, DialectImpl> {
  using BaseInlinerInterface<FullyLegalForInlining<DialectImpl>, DialectImpl>::BaseInlinerInterface;

  bool isLegalToInline(Operation *, Operation *, bool) const override { return true; }
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const override { return true; }
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const override { return true; }
};

} // namespace

namespace llzk {

void registerInliningExtensions(DialectRegistry &registry) {
  registry.addExtension(FuncInlinerInterface::registrationHook);
  registry.addExtension(FullyLegalForInlining<component::StructDialect>::registrationHook);
  registry.addExtension(FullyLegalForInlining<constrain::ConstrainDialect>::registrationHook);
  registry.addExtension(FullyLegalForInlining<string::StringDialect>::registrationHook);
  registry.addExtension(FullyLegalForInlining<polymorphic::PolymorphicDialect>::registrationHook);
  registry.addExtension(FullyLegalForInlining<felt::FeltDialect>::registrationHook);
  registry.addExtension(FullyLegalForInlining<global::GlobalDialect>::registrationHook);
  registry.addExtension(FullyLegalForInlining<boolean::BoolDialect>::registrationHook);
  registry.addExtension(FullyLegalForInlining<array::ArrayDialect>::registrationHook);
  registry.addExtension(FullyLegalForInlining<cast::CastDialect>::registrationHook);
  registry.addExtension(FullyLegalForInlining<include::IncludeDialect>::registrationHook);
  registry.addExtension(FullyLegalForInlining<llzk::LLZKDialect>::registrationHook);
  registry.addExtension(FullyLegalForInlining<pod::PODDialect>::registrationHook);
}

} // namespace llzk
