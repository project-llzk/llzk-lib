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
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/Global/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/String/IR/Ops.h"
#include "llzk/Util/Debug.h"
#include "llzk/Util/SymbolTableLLZK.h"
#include "llzk/Util/Walk.h"

#include <mlir/Dialect/Arith/IR/Arith.h>

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

static Value convertToConstantValue(GlobalReadOp readOp, Type globalType, Attribute attr) {
  OpBuilder bldr(readOp);
  Location loc = readOp.getLoc();
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr)) {
    if (auto feltTy = llvm::dyn_cast<felt::FeltType>(globalType)) {
      auto asFeltAttr = felt::FeltConstAttr::get(readOp.getContext(), intAttr.getValue(), feltTy);
      Value constant = bldr.create<felt::FeltConstantOp>(loc, asFeltAttr).getResult();
      if (constant.getType() != readOp.getType()) {
        return bldr.create<polymorphic::UnifiableCastOp>(loc, readOp.getType(), constant)
            .getResult();
      }
      return constant;
    } else {
      // Generic construction can provide an IntegerAttr whose storage type does
      // not match the global type. Materialize the constant with the declared
      // type so replacing the read preserves SSA type correctness.
      auto typedAttr = IntegerAttr::get(globalType, intAttr.getValue());
      return bldr.create<arith::ConstantOp>(loc, typedAttr).getResult();
    }
  } else if (auto feltAttr = llvm::dyn_cast<felt::FeltConstAttr>(attr)) {
    auto feltTy = llvm::cast<felt::FeltType>(globalType);
    // An unspecified initializer adopts the field declared by its global.
    if (!feltAttr.getType().hasField() && feltTy.hasField()) {
      feltAttr = felt::FeltConstAttr::get(readOp.getContext(), feltAttr.getValue(), feltTy);
    }
    Value constant = bldr.create<felt::FeltConstantOp>(loc, feltAttr).getResult();
    if (constant.getType() == readOp.getType()) {
      return constant;
    }
    // When the read uses an unspecified field type, add a unifiable cast to bridge
    // the specified field type to the unspecified one.
    return bldr.create<polymorphic::UnifiableCastOp>(loc, readOp.getType(), constant).getResult();
  } else if (auto strAttr = llvm::dyn_cast<StringAttr>(attr)) {
    return bldr.create<string::LitStringOp>(loc, readOp.getType(), strAttr).getResult();
  } else {
    llvm::outs() << "Encountered: " << attr.getAbstractAttribute().getName() << '\n';
    llvm_unreachable("Unsupported constant attribute type");
  }
}

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

        // Array constants cannot yet be materialized as operations. Leave both
        // the global and all of its uses unchanged until they are supported.
        if (llvm::isa<ArrayAttr>(constValue)) {
          continue;
        }

        for (Operation *userOp : node->getUserOps()) {
          LLVM_DEBUG(
              llvm::outs() << "Replacing '" << symbolAttr << "' with '" << constValue << "' in "
                           << *userOp << '\n'
          );

          // Special handling to fully replace GlobalReadOp with the appropriate constant op.
          if (auto readOp = llvm::dyn_cast<GlobalReadOp>(userOp);
              readOp && readOp.getNameRef() == symbolAttr) {
            Value constantResult = convertToConstantValue(readOp, globalDef.getType(), constValue);
            readOp.getResult().replaceAllUsesWith(constantResult);
            readOp.erase();
            continue;
          }

          // Standard case that just replaces all uses of the symbol with the constant value.
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
