//===-- LLZKInferDeterminismAttributePass.cpp --------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides the `-infer-determinism-attribute` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/RAM/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/WalkResult.h>

#include <llvm/ADT/StringRef.h>

namespace llzk {
#define GEN_PASS_DEF_INFERDETERMINISMATTRIBUTEPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace llzk;
namespace {

bool isLocalRead(component::MemberReadOp read) {
  auto component = read.getComponent();
  if (auto nestedRead = component.getDefiningOp<component::MemberReadOp>()) {
    return isLocalRead(nestedRead);
  }
  if (component.getDefiningOp<component::CreateStructOp>()) {
    return true;
  }
  return false;
}

/// Infer function determinism attributes and remove invalid annotations.
class PassImpl : public llzk::impl::InferDeterminismAttributePassBase<PassImpl> {
  using Base = InferDeterminismAttributePassBase<PassImpl>;
  using Base::Base;

  void runOnOperation() override {
    getOperation().walk([](function::FuncDefOp funcDef) {
      if (funcDef.isDeclaration()) {
        return;
      }

      bool hasAttr = funcDef->hasAttr(DeterministicAttr::name);
      bool shouldHaveAttr = !funcDef
                                 .walk([&hasAttr, &funcDef](mlir::Operation *op) {
        auto reportBadAttr = [&hasAttr, op, funcDef](llvm::StringRef reason) {
          if (!hasAttr) {
            return;
          }
          funcDef->emitWarning(
                     "removing " + DeterministicAttr::name + " applied to non-deterministic function"
          )
                  .attachNote(op->getLoc())
              << reason;
          funcDef->removeAttr(DeterministicAttr::name);
        };
        if (auto globalRead = dyn_cast<global::GlobalReadOp>(op);
            globalRead && !globalRead.isConstant()) {
          reportBadAttr("non-constant global read");
          return mlir::WalkResult::interrupt();
        }
        if (auto structRead = dyn_cast<component::MemberReadOp>(op);
            structRead && !isLocalRead(structRead)) {
          reportBadAttr("non-self struct member read");
          return mlir::WalkResult::interrupt();
        }
        if (isa<llzk::ram::LoadOp>(op)) {
          reportBadAttr("RAM load");
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      }).wasInterrupted();

      if (shouldHaveAttr) {
        funcDef->setAttr(DeterministicAttr::name, mlir::UnitAttr::get(funcDef.getContext()));
      }
    });
  }
};

} // namespace
