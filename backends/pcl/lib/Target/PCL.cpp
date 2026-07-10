//===-- PCL.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the PCL Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "pcl/Target/PCL.h"

#include "pcl/Dialect/IR/Attrs.h"
#include "pcl/Dialect/IR/Dialect.h"
#include "pcl/Dialect/IR/Ops.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;

namespace {
struct NameState {
  DenseMap<Value, std::string> names;
  unsigned nextId = 0;
  std::string get(Value v, const std::string &prefix = "v") {
    if (auto it = names.find(v); it != names.end()) {
      return it->second;
    }
    std::string s = prefix + std::to_string(nextId++);
    names[v] = s;
    return s;
  }
};

static void emitExpr(Value v, raw_ostream &os, NameState &ns) {
  // Constants
  if (auto c = v.getDefiningOp<pcl::ConstOp>()) {
    c.getValue().getValue().print(os, true);
    return;
  }

  // Variable
  if (auto c = v.getDefiningOp<pcl::VarOp>()) {
    auto name = c.getName();
    ns.names[v] = name.str();
    os << name;
    return;
  }
  // Binary arithmetic
  if (auto add = v.getDefiningOp<pcl::AddOp>()) {
    os << "(+ ";
    emitExpr(add.getLhs(), os, ns);
    os << " ";
    emitExpr(add.getRhs(), os, ns);
    os << ")";
    return;
  }
  if (auto sub = v.getDefiningOp<pcl::SubOp>()) {
    os << "(- ";
    emitExpr(sub.getLhs(), os, ns);
    os << " ";
    emitExpr(sub.getRhs(), os, ns);
    os << ")";
    return;
  }
  if (auto mul = v.getDefiningOp<pcl::MulOp>()) {
    os << "(* ";
    emitExpr(mul.getLhs(), os, ns);
    os << " ";
    emitExpr(mul.getRhs(), os, ns);
    os << ")";
    return;
  }
  if (auto neg = v.getDefiningOp<pcl::NegOp>()) {
    os << "(- ";
    emitExpr(neg.getVal(), os, ns);
    os << ")";
    return;
  }
  if (auto asFelt = v.getDefiningOp<pcl::AsFeltOp>()) {
    emitExpr(asFelt.getVal(), os, ns);
    return;
  }
  // Variables (block args / unmatched defs)
  os << ns.get(v);
}

// emitBool emits a PCL constraint for boolean predicates such as
// (= e_1 e_2), (< e_1, e_2),
static void emitBool(Value v, raw_ostream &os, NameState &ns) {
  if (auto eq = v.getDefiningOp<pcl::CmpEqOp>()) {
    os << "(= ";
    emitExpr(eq.getLhs(), os, ns);
    os << " ";
    emitExpr(eq.getRhs(), os, ns);
    os << ")";
    return;
  }
  if (auto lt = v.getDefiningOp<pcl::CmpLtOp>()) {
    os << "(< ";
    emitExpr(lt.getLhs(), os, ns);
    os << " ";
    emitExpr(lt.getRhs(), os, ns);
    os << ")";
    return;
  }
  if (auto le = v.getDefiningOp<pcl::CmpLeOp>()) {
    os << "(<= ";
    emitExpr(le.getLhs(), os, ns);
    os << " ";
    emitExpr(le.getRhs(), os, ns);
    os << ")";
    return;
  }
  if (auto gt = v.getDefiningOp<pcl::CmpGtOp>()) {
    os << "(> ";
    emitExpr(gt.getLhs(), os, ns);
    os << " ";
    emitExpr(gt.getRhs(), os, ns);
    os << ")";
    return;
  }
  if (auto ge = v.getDefiningOp<pcl::CmpGeOp>()) {
    os << "(>= ";
    emitExpr(ge.getLhs(), os, ns);
    os << " ";
    emitExpr(ge.getRhs(), os, ns);
    os << ")";
    return;
  }
  if (auto det = v.getDefiningOp<pcl::DetOp>()) {
    os << "(det ";
    emitExpr(det.getA(), os, ns);
    os << ")";
    return;
  }
  // Fallback: treat as variable
  os << ns.get(v);
}
}; // namespace

LogicalResult pcl::moduleToPcl(ModuleOp mod, raw_ostream &os) {
  // (prime-number …)
  auto prime = mod->getAttrOfType<pcl::PrimeAttr>("pcl.prime");
  if (!prime) {
    return mod.emitError("missing 'pcl.prime'"), failure();
  }
  os << "(prime-number " << prime.getValue().getValue() << ")\n";

  for (auto fn : mod.getOps<func::FuncOp>()) {
    NameState ns;
    std::string modName = fn.getSymName().str();
    os << "(begin-module " << modName << ")\n";

    // Inputs = function args
    auto &entry = fn.getBody().front();
    for (BlockArgument arg : entry.getArguments()) {
      os << "(input " << ns.get(arg, "in") << ")\n";
    }

    // Walk ops: print assertions and collects outputs if you want
    fn.walk([&](Operation *op) {
      if (auto a = dyn_cast<pcl::AssertOp>(op)) {
        os << "(assert ";
        emitBool(a.getCond(), os, ns);
        os << ")\n";
      } else if (auto post = dyn_cast<pcl::PostOp>(op)) {
        os << "(post-condition ";
        emitBool(post.getCond(), os, ns);
        os << ")\n";
      }
    });

    // Outputs = return operands
    if (auto ret = dyn_cast_or_null<func::ReturnOp>(entry.getTerminator())) {
      for (Value v : ret.getOperands()) {
        os << "(output " << ns.get(v, "out") << ")\n";
      }
    }

    os << "(end-module)\n\n";
  }
  return success();
}
