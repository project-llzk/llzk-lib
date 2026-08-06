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

/// Environment for keeping track of the mapping between MLIR values and variable names.
struct NameState {
  DenseMap<Value, std::string> names;
  unsigned nextId = 0;

  /// Returns the name the given value maps into.
  ///
  /// If the name is not already in the environment creates one using the given prefix.
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

  /// Sets the name of the given value.
  ///
  /// If the value already had a mapping, it is overriden.
  void set(Value v, std::string name) { names[v] = std::move(name); }
};

/// Emits the s-expressions for the beginning of the PCL file.
LogicalResult prologue(ModuleOp mod, pcl::Sexps &S) {
  // (prime-number …)
  auto prime = mod->getAttrOfType<pcl::PrimeAttr>("pcl.prime");
  if (!prime) {
    return mod.emitError("missing 'pcl.prime'");
  }
  S.push({S.atom("prime-number"), S.atom(prime.getValue())});
  return success();
}

/// Handles emission of the s-expressions representing a PCL module by a `func.func` operation.
class ModuleEmitter {
  NameState ns;
  func::FuncOp func;

  /// Helper for getting the inputs of the module.
  ArrayRef<BlockArgument> inputs() { return func.getBody().front().getArguments(); }

  /// Helper for getting the outputs of the module.
  SmallVector<Value> outputs() {
    if (auto ret = dyn_cast_or_null<func::ReturnOp>(func.getBody().front().getTerminator())) {
      return llvm::map_to_vector(ret.getOperands(), [](auto v) { return v; });
    }
    return {};
  }

  /// Emits the s-expressions in the prologue of the module.
  ///
  /// The prologue comprises the `(begin-module ...)` declaration followed by
  /// the input declarations.
  void prologue(pcl::Sexps &S) {
    S.push({S.atom("begin-module"), S.atom(func.getSymName())});
    for (auto arg : inputs()) {
      S.push({S.atom("input"), S.atom(ns.get(arg, "in"))});
    }
  }

  /// Emits the s-expressions in the epilogue of the module.
  ///
  /// The epilogue comprises the output declarations followed by the `(end-module)` closer.
  /// Outputs could be anonymous, meaning that the SSA value passed to the `func.return` op does
  /// not come from an operation that defines a name. In that case, in addition to the `(output
  /// ...)` declaration, a small piece of code is added that emits the s-expression of the output
  /// and asserts that is equal to the name given to the output.
  ///
  /// For example, the following IR:
  ///
  /// ```
  /// %0 = pcl.var "a" true   // Named output.
  /// %1 = pcl.const 3
  /// %2 = pcl.var "b" false
  /// %3 = pcl.mul %1, %2     // Anonymous output.
  /// return %0, %3 : !pcl.felt, !pcl.felt
  /// ```
  ///
  /// Will emit s-expressions similar to the following:
  ///
  /// ```
  /// (output a)                  ; The named output.
  /// (assert (= out0 (* 3 b)))   ; Assert the anon output.
  /// (output out0)               ; Declare the anon output.
  /// (end-module)                ; Close the module.
  /// ```
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

  /// Helper for emitting a binary expression's s-expressions.
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

  /// Helper for emitting an unary expression's s-expressions.
  FailureOr<pcl::Sexp> emitUnaryExpr(llvm::StringLiteral sym, Value v, pcl::Sexps &S) {
    auto vs = emitExpr(v, S);
    if (failed(vs)) {
      return failure();
    }
    return S.sexp({S.atom(sym), *vs});
  }

  /// Helper for emitting a binary formula's s-expressions.
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

  /// Helper for emitting an unary formula's s-expressions.
  FailureOr<pcl::Sexp> emitUnaryFormula(llvm::StringLiteral sym, Value v, pcl::Sexps &S) {
    auto vs = emitFormula(v, S);
    if (failed(vs)) {
      return failure();
    }
    return S.sexp({S.atom(sym), *vs});
  }

  /// Helper for emitting the s-expression of a variable name.
  ///
  /// The value must have a mapping in the environment to an existing name.
  FailureOr<pcl::Sexp> emitVar(Value v, pcl::Sexps &S) {
    auto name = ns.getOrFail(v);
    if (failed(name)) {
      return func->emitOpError() << ", value " << v << " could not be emitted";
    }
    return S.atom(*name);
  }

  /// Emits the s-expressions for the given expression represented by the value.
  FailureOr<pcl::Sexp> emitExpr(Value v, pcl::Sexps &S) {
    auto *defOp = v.getDefiningOp();
    if (!defOp) {
      return emitVar(v, S);
    }
    return llvm::TypeSwitch<Operation *, FailureOr<pcl::Sexp>>(defOp)
        .Case<pcl::AddOp>([this, &S](auto op) { return emitBinaryExpr("+", op, S); })
        .Case<pcl::MulOp>([this, &S](auto op) { return emitBinaryExpr("*", op, S); })
        .Case<pcl::SubOp>([this, &S](auto op) { return emitBinaryExpr("-", op, S); })
        .Case<pcl::NegOp>([this, &S](auto op) { return emitUnaryExpr("-", op, S); })
        .Case<pcl::AsFeltOp>([this, &S](auto op) { return emitFormula(op, S); })
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

  /// Emits the s-expressions for the given formula represented by the value.
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
        .Case<pcl::NotOp>([this, &S](auto op) { return emitUnaryFormula("!", op, S); })
        .Case<pcl::DetOp>([this, &S](auto op) { return emitUnaryExpr("det", op, S); })
        .Case<pcl::TrueOp>([&S](auto) { return S.atom<unsigned>(1); })
        .Case<pcl::FalseOp>([&S](auto) {
      return S.atom<unsigned>(0);
    }).Default([this, &S, v](auto) { return emitExpr(v, S); });
  }

  /// Emits the body of the module.
  ///
  /// The body of the module comprises root s-expressions: `assert`, `post-condition`, `call`, and
  /// `assume-deterministic`.
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

  /// Fills the environment with values that declare a name.
  ///
  /// Values that can declare names are the inputs and outputs of the module, `pcl.var` ops, and
  /// `func.call` ops.
  ///
  /// In the case of `func.call` ops, the op "declares" one name per result.
  /// Let that op be the N-th `func.call` op in the body and the value the `M-th` result of the op,
  /// the value will declare a name using the pattern `{callee name}_call{N}_out{M}`.
  ///
  /// For example, given the following IR and assuming N = 0:
  ///
  /// ```
  /// %X:3 = func.call @Foo(%0, %1) : (!F, !F) -> (!F, !F, !F)
  /// ```
  ///
  /// The environment will map the SSA values:
  /// - `%X:0` to `Foo_call0_out0`
  /// - `%X:1` to `Foo_call0_out1`
  /// - `%X:2` to `Foo_call0_out2`
  ///
  /// The order in which these names are added to the environment matters. We add the inputs and
  /// the outputs first since those will have autogenerated names (`in0`, `in1`, `out2`, etc.).
  /// Then, if an output's SSA value comes from either a `pcl.var` or `func.call` we override
  /// the autogenerated name with the name declared by that operation.
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

  /// Emits the complete sequence of s-expressions representing the module.
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

/// Locates all the `func.func` ops in the MLIR module that represent a PCL module.
static SmallVector<ModuleEmitter> findModules(ModuleOp op) {
  return llvm::map_to_vector(op.getOps<func::FuncOp>(), [](auto f) { return ModuleEmitter(f); });
}

static void nl(raw_ostream &os, const pcl::PCLTargetConfig &config) {
  if (!config.compressedLines) {
    os << '\n';
  }
}

} // namespace

LogicalResult pcl::moduleToPcl(ModuleOp mod, raw_ostream &os, PCLTargetConfig config) {
  pcl::Sexps S;

  if (failed(prologue(mod, S))) {
    return failure();
  }

  S.dump(os);
  nl(os, config);

  for (auto &emitter : findModules(mod)) {
    if (failed(emitter.emit(S))) {
      return failure();
    }
    S.dump(os);
    nl(os, config);
  }

  return success();
}
