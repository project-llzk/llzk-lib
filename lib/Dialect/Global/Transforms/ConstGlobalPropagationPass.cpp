//===-- ConstGlobalPropagationPass.cpp --------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-const-global-propagation` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/SymbolUseGraph.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/Global/Transforms/TransformationPasses.h"
#include "llzk/Util/Debug.h"
#include "llzk/Util/SymbolTableLLZK.h"
#include "llzk/Util/Walk.h"

// Include the generated base pass class definitions.
namespace llzk::global {
#define GEN_PASS_DEF_CONSTGLOBALPROPAGATIONPASS
#include "llzk/Dialect/Global/Transforms/TransformationPasses.h.inc"
} // namespace llzk::global

#define DEBUG_TYPE "llzk-const-global-propagation"

using namespace mlir;
using namespace llzk;
using namespace llzk::global;

namespace {

class PassImpl : public llzk::global::impl::ConstGlobalPropagationPassBase<PassImpl> {
  using Base = ConstGlobalPropagationPassBase<PassImpl>;

public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp root = getOperation();
    auto constGlobals = walkCollect<GlobalDefOp>(root, [](auto g) { return g.isConstant(); });
    if (constGlobals.empty()) {
      return;
    }

    SymbolUseGraph &useGraph = getAnalysis<SymbolUseGraph>();
    for (GlobalDefOp globalDef : constGlobals) {
      if (const SymbolUseGraphNode *node = useGraph.lookupNode(globalDef)) {
        SymbolRefAttr symbolAttr = node->getSymbolPath();
        Attribute constValue = globalDef.getInitialValueAttr();
        for (Operation *userOp : node->getUserOps()) {
          LLVM_DEBUG(
              llvm::outs() << "Replacing '" << symbolAttr << "' with '" << constValue << "' in "
                           << *userOp << '\n'
          );

          AttrTypeReplacer replacer;
          replacer.addReplacement([symbolAttr, constValue](SymbolRefAttr a) {
            return (a == symbolAttr) ? std::make_optional(constValue) : std::nullopt;
          });
          replacer.replaceElementsIn(
              userOp,
              /*replaceAttrs=*/true,
              /*replaceLocs=*/false,
              /*replaceTypes=*/true
          );
        }
      }
      // Delete the const global since there are no (remaining) uses.
      globalDef.erase();
    }
  }
};

} // namespace
