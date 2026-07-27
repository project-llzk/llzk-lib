//===-- PCL.cpp -------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "pcl/Target/PCL.h"

#include "Sexp.h"
#include "pcl/Dialect/IR/Attrs.h"
#include "pcl/Dialect/IR/Dialect.h"
#include "pcl/Dialect/IR/Ops.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/raw_ostream.h>

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

  /// Strict version of `get` that fails if the name is not defined.
  FailureOr<llvm::StringRef> getOrFail(Value v) {
    auto it = names.find(v);
    if (it == names.end()) {
      return failure();
    }
    return StringRef(it->second);
  }

  void set(Value v, std::string name) { names[v] = std::move(name); }
};

LogicalResult prologue(ModuleOp mod, pcl::Sexps &S) {
  // (prime-number …)
  auto prime = mod->getAttrOfType<pcl::PrimeAttr>("pcl.prime");
  if (!prime) {
    return mod.emitError("missing 'pcl.prime'");
  }
  S.push({S.atom("prime-number"), S.atom(prime.getValue())});
  return success();
}

class ModuleEmitter {
  NameState ns;
  func::FuncOp func;

  ArrayRef<BlockArgument> inputs() { return func.getBody().front().getArguments(); }

  SmallVector<Value> outputs() {
    if (auto ret = dyn_cast_or_null<func::ReturnOp>(func.getBody().front().getTerminator())) {
      return llvm::map_to_vector(ret.getOperands(), [](auto v) { return v; });
    }
    return {};
  }

  void prologue(pcl::Sexps &S) {
    S.push({S.atom("begin-module"), S.atom(func.getSymName())});
    for (auto arg : inputs()) {
      S.push({S.atom("input"), S.atom(ns.get(arg, "in"))});
    }
  }

  LogicalResult epilogue(pcl::Sexps &S) {
    if (auto ret = dyn_cast_or_null<func::ReturnOp>(func.getBody().front().getTerminator())) {
      for (Value v : ret.getOperands()) {
        // Map the output to either a previously assigned map (i.e. from a `pcl.var` op) or to
        // "outN" by default.
        auto name = ns.get(v, "out");
        // If the value does not come from a non-op source (i.e. arguments), `pcl.var`, or
        // `func.call` then we need to emit `(assert (= outN {emitFormula(v)}))`. Otherwise that
        // slice gets lost in translation. The conversion pass avoids generating IR like this but,
        // as a precaution, we handle it here.
        auto *defOp = v.getDefiningOp();
        if (defOp && !mlir::isa<pcl::VarOp, func::CallOp>(defOp)) {
          auto vSexp = emitFormula(v, S);
          if (failed(vSexp)) {
            return failure();
          }
          S.push({S.atom("assert"), S.sexp({S.atom("="), S.atom(name), *vSexp})});
        }
        S.push({S.atom("output"), S.atom(name)});
      }
    }
    S.push(S.sexp({S.atom("end-module")}));
    return success();
  }

  template <typename Op>
  FailureOr<pcl::Sexp> emitBinaryExpr(llvm::StringLiteral sym, Op op, pcl::Sexps &S) {
    auto lhs = emitExpr(op.getLhs(), S);
    if (failed(lhs)) {
      return failure();
    }
    auto rhs = emitExpr(op.getRhs(), S);
    if (failed(rhs)) {
      return failure();
    }
    return S.sexp({S.atom(sym), *lhs, *rhs});
  }

  FailureOr<pcl::Sexp> emitUnaryExpr(llvm::StringLiteral sym, Value v, pcl::Sexps &S) {
    auto vs = emitExpr(v, S);
    if (failed(vs)) {
      return failure();
    }
    return S.sexp({S.atom(sym), *vs});
  }

  template <typename Op>
  FailureOr<pcl::Sexp> emitBinaryFormula(llvm::StringLiteral sym, Op op, pcl::Sexps &S) {
    auto lhs = emitFormula(op.getLhs(), S);
    if (failed(lhs)) {
      return failure();
    }
    auto rhs = emitFormula(op.getRhs(), S);
    if (failed(rhs)) {
      return failure();
    }
    return S.sexp({S.atom(sym), *lhs, *rhs});
  }

  FailureOr<pcl::Sexp> emitUnaryFormula(llvm::StringLiteral sym, Value v, pcl::Sexps &S) {
    auto vs = emitFormula(v, S);
    if (failed(vs)) {
      return failure();
    }
    return S.sexp({S.atom(sym), *vs});
  }

  FailureOr<pcl::Sexp> emitVar(Value v, pcl::Sexps &S) {
    auto name = ns.getOrFail(v);
    if (failed(name)) {
      return func->emitOpError() << ", value " << v << " could not be emitted";
    }
    return S.atom(*name);
  }

  FailureOr<pcl::Sexp> emitExpr(Value v, pcl::Sexps &S) {
    auto *defOp = v.getDefiningOp();
    if (!defOp) {
      return emitVar(v, S);
    }
    return llvm::TypeSwitch<Operation *, FailureOr<pcl::Sexp>>(defOp)
        .Case<pcl::AddOp>([this, &S](auto op) { return emitBinaryExpr("+", op, S); })
        .Case<pcl::MulOp>([this, &S](auto op) { return emitBinaryExpr("*", op, S); })
        .Case<pcl::SubOp>([this, &S](auto op) { return emitBinaryExpr("-", op, S); })
        .Case<pcl::NegOp>([this, &S](auto op) { return emitUnaryExpr("-", op.getVal(), S); })
        .Case<pcl::AsFeltOp>([this, &S](auto op) { return emitFormula(op.getVal(), S); })
        .Case<pcl::VarOp>([this, &S](auto op) { return S.atom(ns.get(op)); })
        .Case<pcl::ConstOp>([&S](auto op) { return S.atom(op.getValueAPInt()); })
        .Case<func::CallOp>([this, &S, v](auto) {
      // If we encounter a call we need to emit the value as a variable,
      // since we preload all the call outputs into the environment.
      return emitVar(v, S);
    }).Default([v](auto op) {
      return op->emitOpError() << ", value " << v << " could not be emitted";
    });
  }

  FailureOr<pcl::Sexp> emitFormula(Value v, pcl::Sexps &S) {
    auto *defOp = v.getDefiningOp();
    if (!defOp) {
      return emitVar(v, S);
    }
    return llvm::TypeSwitch<Operation *, FailureOr<pcl::Sexp>>(defOp)
        .Case<pcl::CmpEqOp>([this, &S](auto op) { return emitBinaryExpr("=", op, S); })
        .Case<pcl::CmpLtOp>([this, &S](auto op) { return emitBinaryExpr("<", op, S); })
        .Case<pcl::CmpLeOp>([this, &S](auto op) { return emitBinaryExpr("<=", op, S); })
        .Case<pcl::CmpGtOp>([this, &S](auto op) { return emitBinaryExpr(">", op, S); })
        .Case<pcl::CmpGeOp>([this, &S](auto op) { return emitBinaryExpr(">=", op, S); })
        .Case<pcl::AndOp>([this, &S](auto op) { return emitBinaryFormula("&&", op, S); })
        .Case<pcl::OrOp>([this, &S](auto op) { return emitBinaryFormula("||", op, S); })
        .Case<pcl::ImpliesOp>([this, &S](auto op) { return emitBinaryFormula("=>", op, S); })
        .Case<pcl::IffOp>([this, &S](auto op) { return emitBinaryFormula("<=>", op, S); })
        .Case<pcl::NotOp>([this, &S](auto op) { return emitUnaryFormula("!", op.getCond(), S); })
        .Case<pcl::DetOp>([this, &S](auto op) { return emitUnaryExpr("det", op.getExpr(), S); })
        .Case<pcl::TrueOp>([&S](auto) { return S.atom<unsigned>(1); })
        .Case<pcl::FalseOp>([&S](auto) {
      return S.atom<unsigned>(0);
    }).Default([this, &S, v](auto) { return emitExpr(v, S); });
  }

  LogicalResult body(pcl::Sexps &S) {
    return failure(func.walk([this, &S](Operation *op) {
      if (failed(
              llvm::TypeSwitch<Operation *, LogicalResult>(op)
                  .Case<pcl::AssertOp>([this, &S](auto assertOp) {
        auto cond = emitFormula(assertOp.getCond(), S);
        if (failed(cond)) {
          return failure();
        }
        S.push({S.atom("assert"), *cond});
        return success();
      })
                  .Case<pcl::PostOp>([this, &S](auto postOp) {
        auto cond = emitFormula(postOp.getCond(), S);
        if (failed(cond)) {
          return failure();
        }
        S.push({S.atom("post-condition"), *cond});
        return success();
      })
                  .Case<pcl::AssumeDeterministicOp>([this, &S](auto assumeOp) {
        auto expr = emitExpr(assumeOp.getV(), S);
        if (failed(expr)) {
          return failure();
        }
        S.push({S.atom("assume-deterministic"), *expr});
        return success();
      })
                  .Case<func::CallOp>([this, &S](auto callOp) {
        SmallVector<pcl::Sexp> outputs, inputs;
        outputs.reserve(callOp.getResults().size());
        inputs.reserve(callOp.getOperands().size());

        for (Value v : callOp.getResults()) {
          auto name = ns.getOrFail(v);
          if (failed(name)) {
            return failure();
          }
          outputs.push_back(S.atom(*name));
        }
        for (Value v : callOp.getOperands()) {
          auto expr = emitExpr(v, S);
          if (failed(expr)) {
            return failure();
          }
          inputs.push_back(*expr);
        }
        auto outputsExpr = S.sexp(outputs).withSquareBrackets();
        auto inputsExpr = S.sexp(inputs).withSquareBrackets();
        S.push({S.atom("call"), outputsExpr, S.atom(callOp.getCallee()), inputsExpr});
        return success();
      }).Default([](auto) { return success(); })
          )) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }).wasInterrupted());
  }

  void fillNames() {
    for (auto input : inputs()) {
      (void)ns.get(input, "in");
    }
    for (auto output : outputs()) {
      (void)ns.get(output, "out");
    }
    func.walk([this](pcl::VarOp varOp) { ns.set(varOp, varOp.getName().str()); });
    unsigned callNo = 0;
    func.walk([this, &callNo](func::CallOp callOp) {
      for (auto [n, v] : llvm::enumerate(callOp.getResults())) {
        std::string name = (callOp.getCallee() + "_call" + Twine(callNo) + "_out" + Twine(n)).str();
        ns.set(v, name);
      }
      callNo++;
    });
  }

public:
  ModuleEmitter(func::FuncOp F) : func(F) {}

  LogicalResult emit(pcl::Sexps &S) {
    fillNames();
    prologue(S);
    if (failed(body(S))) {
      return failure();
    }
    if (failed(epilogue(S))) {
      return failure();
    }

    return success();
  }
};

static SmallVector<ModuleEmitter> findModules(ModuleOp op) {
  return llvm::map_to_vector(op.getOps<func::FuncOp>(), [](func::FuncOp f) {
    return ModuleEmitter(f);
  });
}

void nl(raw_ostream &os) { os << '\n'; }

} // namespace

LogicalResult pcl::moduleToPcl(ModuleOp mod, raw_ostream &os, PCLTargetConfig config) {
  pcl::Sexps S;

  if (failed(prologue(mod, S))) {
    return failure();
  }

  S.dump(os);
  if (!config.compressedLines) {
    nl(os);
  }

  for (auto &emitter : findModules(mod)) {
    if (failed(emitter.emit(S))) {
      return failure();
    }
    S.dump(os);
    if (!config.compressedLines) {
      nl(os);
    }
  }

  return success();
}
