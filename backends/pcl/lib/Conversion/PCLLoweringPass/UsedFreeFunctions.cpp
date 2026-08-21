//===-- UsedFreeFunctions.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "UsedFreeFunctions.h"

using namespace llzk::function;
using namespace llzk::component;
using namespace mlir;
using namespace pcl::lowering;

void UsedFreeFunctions::insertCallees(
    SmallVectorImpl<FuncDefOp> &WL, SymbolTableCollection &tables, FuncDefOp f,
    function_ref<bool(FuncDefOp)> P
) {
  f.walk([&WL, this, &tables, P](CallOp callOp) {
    auto calleeLookup = callOp.getCalleeTarget(tables);
    if (failed(calleeLookup)) {
      return;
    }

    auto callee = calleeLookup->get();
    // If function already in set, or predicate is true (if available), skip the op.
    if (contains(callee) || (P && P(callee))) {
      return;
    }

    insert(callee);
    WL.push_back(callee);
  });
}

UsedFreeFunctions::UsedFreeFunctions(ModuleOp op) {
  SmallVector<FuncDefOp> WL;
  SymbolTableCollection tables;

  // Fill the worklist with the initial set of functions.
  op->walk([&WL, this, &tables](StructDefOp structOp) {
    if (auto f = structOp.getConstrainFuncOp()) {
      insertCallees(WL, tables, f, [](FuncDefOp callee) { return callee.isStructConstrain(); });
    }
  });

  // Complete the graph using the worklist.
  while (!WL.empty()) {
    auto next = WL.back();
    WL.pop_back();
    insertCallees(WL, tables, next);
  }
}

void UsedFreeFunctions::erase(FuncDefOp op) {
  auto fqn = op.getFullyQualifiedName();
  if (auto mappedOp = funcs[fqn]; mappedOp == op) {
    funcs.erase(fqn);
  }
}

bool pcl::lowering::operator!=(
    const UsedFreeFunctions::iterator &LHS, const UsedFreeFunctions::iterator &RHS
) {
  return LHS.iter != RHS.iter;
}
