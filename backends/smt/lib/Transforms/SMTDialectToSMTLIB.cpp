//===-- SMTDialectToSMTLIB.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "smt/Transforms/SMTLIBEmitter.h"
#include "smt/Transforms/SMTPasses.h"

#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/SMT/IR/SMTAttributes.h"
#include "llzk/Dialect/SMT/IR/SMTDialect.h"
#include "llzk/Dialect/SMT/IR/SMTOps.h"
#include "llzk/Dialect/SMT/IR/SMTTypes.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/FileUtilities.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/raw_ostream.h>

#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace llzk::smt {
#define GEN_PASS_DEF_SMTDIALECTTOSMTLIBPASS
#include "smt/Transforms/SMTPasses.h.inc"
} // namespace llzk::smt

using namespace mlir;

namespace {

/// Sanitize a symbol or user-provided identifier into a bare SMT-LIB symbol.
static std::string sanitizeSymbol(StringRef name) {
  std::string out;
  out.reserve(name.size());
  for (char ch : name) {
    if (llvm::isAlnum(ch) || ch == '_' || ch == '.' || ch == '$') {
      out.push_back(ch);
    } else {
      out.push_back('_');
    }
  }
  if (out.empty()) {
    out = "tmp";
  }
  return out;
}

/// Convert an SMT dialect sort to its SMT-LIB textual form.
static std::string sortForType(Type type) {
  return TypeSwitch<Type, std::string>(type)
      .Case<llzk::smt::IntType>([](auto) { return "Int"; })
      .Case<llzk::smt::BoolType>([](auto) { return "Bool"; })
      .Case<llzk::smt::SMTFuncType>([](auto type) {
    std::string sort = "((";
    llvm::interleave(type.getDomainTypes(), [&](Type domainType) {
      sort += sortForType(domainType);
    }, [&] { sort.push_back(' '); });
    sort += ") ";
    sort += sortForType(type.getRangeType());
    sort.push_back(')');
    return sort;
  })
      .Case<llzk::smt::BitVectorType>([](auto type) {
    return "(_ BitVec " + std::to_string(type.getWidth()) + ")";
  })
      .Case<llzk::smt::ArrayType>([](auto type) {
    return "(Array " + sortForType(type.getDomainType()) + " " + sortForType(type.getRangeType()) +
           ")";
  }).Default([&](Type) -> std::string {
    llvm::report_fatal_error("unsupported SMTLIB sort in smt-to-smtlib");
  });
}

struct CheckSuccessInfo {
  StringRef expectedOutcome;
  Region *successRegion;
};

enum class HelperMode {
  PureFunction,
  InlineScript,
};

/// Stateful SMT-LIB emitter for one module-level solver export.
///
/// The emitter linearizes the selected root `smt.solver`, emits reusable pure
/// helpers as `define-fun`, and inlines script-style helpers at call sites.
class SMTLIBEmitter {
public:
  SMTLIBEmitter(ModuleOp module, llvm::raw_ostream &os) : module(module), os(os) {}

  LogicalResult emit() {
    auto root = collectRoot();
    if (failed(root)) {
      return failure();
    }
    return emitRoot(*root, /*emitReset=*/false);
  }

private:
  struct EvalContext {
    DenseMap<Value, std::string> values;
    std::optional<std::string> pendingStageLabel;
    SmallVector<std::pair<std::string, std::string>> letBindings;
    bool preserveSharing = false;
  };

  ModuleOp getEffectiveRootModule() {
    ModuleOp nestedModule;
    for (Operation &op : module.getBody()->getOperations()) {
      auto childModule = dyn_cast<ModuleOp>(op);
      if (!childModule) {
        return module;
      }
      if (nestedModule) {
        return module;
      }
      nestedModule = childModule;
    }
    return nestedModule ? nestedModule : module;
  }

  FailureOr<llzk::smt::SolverOp> collectRoot() {
    ModuleOp rootModule = getEffectiveRootModule();
    llzk::smt::SolverOp solver;
    for (Operation &op : rootModule.getBody()->getOperations()) {
      auto solverOp = dyn_cast<llzk::smt::SolverOp>(op);
      if (!solverOp) {
        continue;
      }
      if (solver) {
        rootModule.emitError(
            "smt-to-smtlib requires exactly one top-level smt.solver in the root module"
        );
        return failure();
      }
      solver = solverOp;
    }
    if (!solver) {
      rootModule.emitError(
          "smt-to-smtlib could not find a top-level smt.solver in the root "
          "module"
      );
      return failure();
    }
    return solver;
  }

  LogicalResult emitRoot(llzk::smt::SolverOp solver, bool emitReset) {
    return emitSolverRoot(solver, emitReset);
  }

  static bool solverHasExplicitSetLogic(llzk::smt::SolverOp solver) {
    return llvm::any_of(solver.getBodyRegion().front().without_terminator(), [](Operation &op) {
      return isa<llzk::smt::SetLogicOp>(op);
    });
  }

  void resetScriptState() {
    nextTempId = 0;
    nextTargetStageIndex = 0;
    nextPostStageIndex = 0;
    pushDepth = 0;
    emittedSymbolCounts.clear();
    emittedAssertions.clear();
    helperSymbols.clear();
    emittedPureHelpers.clear();
  }

  LogicalResult emitRootPreamble(bool emitReset, bool emitDefaultLogic) {
    if (emitReset) {
      os << "\n(reset)\n";
    }
    resetScriptState();
    if (emitDefaultLogic) {
      os << "(set-logic ALL)\n";
    }
    os << "(set-info :llzk-root \"module\")\n";
    return success();
  }

  LogicalResult emitSolverRoot(llzk::smt::SolverOp solver, bool emitReset) {
    if (solver.getNumOperands() != 0 || solver.getBodyRegion().front().getNumArguments() != 0) {
      return solver.emitError(
          "SMT-LIB scripts have no standard parameter channel; the selected top-level smt.solver "
          "must be a closed script"
      );
    }
    if (solver.getNumResults() != 0 ||
        !llvm::all_of(solver.getResults(), [](Value result) { return result.use_empty(); })) {
      return solver.emitError(
          "SMT-LIB scripts have no standard return channel; the selected top-level smt.solver "
          "must not return results"
      );
    }

    bool emitDefaultLogic = !solverHasExplicitSetLogic(solver);
    if (failed(emitRootPreamble(emitReset, emitDefaultLogic))) {
      return failure();
    }

    EvalContext ctx;
    return emitBlock(solver.getBodyRegion().front(), ctx);
  }

  LogicalResult emitBlock(Block &block, EvalContext &ctx) {
    for (Operation &op : block.without_terminator()) {
      if (failed(emitOperation(&op, ctx))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult emitOperation(Operation *op, EvalContext &ctx) {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<llzk::smt::SetLogicOp>([&](auto setLogicOp) {
      os << "(set-logic " << setLogicOp.getLogic() << ")\n";
      return success();
    })
        .Case<llzk::smt::DeclareFunOp>([&](auto declareOp) { return emitDeclare(declareOp, ctx); })
        .Case<llzk::smt::AssertOp>([&](auto assertOp) { return emitAssert(assertOp, ctx); })
        .Case<llzk::smt::ResetOp>([&](auto) {
      os << "(reset)\n";
      resetScriptState();
      return success();
    })
        .Case<llzk::smt::PushOp>([&](auto pushOp) {
      pushDepth += pushOp.getCount();
      os << "(push " << pushOp.getCount() << ")\n";
      return success();
    })
        .Case<llzk::smt::PopOp>([&](auto popOp) {
      pushDepth -= popOp.getCount();
      os << "(pop " << popOp.getCount() << ")\n";
      return success();
    })
        .Case<llzk::smt::CheckOp>([&](auto checkOp) { return emitCheck(checkOp, ctx); })
        .Case<llzk::smt::SolverOp>([&](auto solverOp) {
      return solverOp.emitError(
          "nested smt.solver is not exportable to SMT-LIB; SMT-LIB has a single script context, "
          "so use push/pop if same-solver nesting was intended"
      );
    })
        .Case<func::CallOp>([&](auto callOp) { return emitCall(callOp, ctx); })
        .Case<UnrealizedConversionCastOp>([&](auto castOp) {
      return emitUnrealizedCast(castOp, ctx);
    })
        .Case<arith::ConstantOp>([&](auto constOp) { return emitArithConstant(constOp, ctx); })
        .Case<llzk::smt::BoolConstantOp>([&](auto constOp) { return bindExpr(constOp, ctx); })
        .Case<llzk::smt::IntConstantOp>([&](auto constOp) { return bindExpr(constOp, ctx); })
        .Case<llzk::smt::BVConstantOp>([&](auto constOp) { return bindExpr(constOp, ctx); })
        .Case<llzk::smt::EqOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::NotOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::AndOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::OrOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::XOrOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::ImpliesOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::IteOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::IntNegOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::IntAddOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::IntMulOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::IntSubOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::IntDivOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::IntModOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::IntCmpOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::Int2BVOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BV2IntOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::DistinctOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::IntAbsOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BVNegOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BVAndOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BVAddOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BVMulOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BVUDivOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BVSDivOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BVURemOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BVSRemOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BVSModOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BVOrOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BVXOrOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BVNotOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BVShlOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BVLShrOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BVAShrOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BVCmpOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::ConcatOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::ExtractOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::RepeatOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::ApplyFuncOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::ArraySelectOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::ArrayStoreOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::ArrayBroadcastOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::ForallOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::ExistsOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::boolean::AssertOp>([&](auto assertOp) {
      return assertOp.emitError(
          "boolean.assert is only supported inside smt.check failure "
          "regions and should not appear on the linearized path"
      );
    }).Default([&](Operation *unknownOp) {
      unknownOp->emitError("unsupported operation in smt-to-smtlib");
      return failure();
    });
  }

  template <typename OpTy> LogicalResult bindExpr(OpTy op, EvalContext &ctx) {
    auto expr = buildExpr(op.getOperation(), ctx);
    if (failed(expr)) {
      return failure();
    }
    if (op->getNumResults() != 1) {
      op.emitError("smt-to-smtlib only supports single-result expression ops");
      return failure();
    }
    if (ctx.preserveSharing && !expr->starts_with("(")) {
      ctx.values[op->getResult(0)] = std::move(*expr);
      return success();
    }
    if (ctx.preserveSharing) {
      std::string name = makeLetName();
      ctx.letBindings.emplace_back(name, std::move(*expr));
      ctx.values[op->getResult(0)] = std::move(name);
      return success();
    }
    ctx.values[op->getResult(0)] = std::move(*expr);
    return success();
  }

  LogicalResult emitDeclare(llzk::smt::DeclareFunOp declareOp, EvalContext &ctx) {
    std::string symbol;
    if (auto prefix = declareOp.getNamePrefix()) {
      symbol = sanitizeSymbol(*prefix);
    } else {
      symbol = "tmp" + std::to_string(nextTempId++);
    }
    if (unsigned &count = emittedSymbolCounts[symbol]; count++ != 0) {
      symbol += "_" + std::to_string(count);
    }
    ctx.values[declareOp.getResult()] = symbol;
    if (auto funcType = dyn_cast<llzk::smt::SMTFuncType>(declareOp.getType())) {
      os << "(declare-fun " << symbol << " (";
      llvm::interleave(funcType.getDomainTypes(), [&](Type domainType) {
        os << sortForType(domainType);
      }, [&] { os << ' '; });
      os << ") " << sortForType(funcType.getRangeType()) << ")\n";
      return success();
    }
    os << "(declare-fun " << symbol << " () " << sortForType(declareOp.getType()) << ")\n";
    return success();
  }

  LogicalResult emitAssert(llzk::smt::AssertOp assertOp, EvalContext &ctx) {
    auto expr = lookup(assertOp.getInput(), ctx);
    if (failed(expr)) {
      return assertOp.emitError("missing SMTLIB expression for assertion input");
    }
    std::string rendered = wrapWithLets(*expr, ctx.letBindings);
    ctx.letBindings.clear();
    if (pushDepth == 0 && !emittedAssertions.insert(rendered).second) {
      return success();
    }
    os << "(assert " << rendered << ")\n";
    return success();
  }

  LogicalResult emitArithConstant(arith::ConstantOp constOp, EvalContext &ctx) {
    if (constOp->getNumResults() != 1) {
      return constOp.emitOpError("smt-to-smtlib only supports single-result arith.constant");
    }
    if (auto boolAttr = dyn_cast<BoolAttr>(constOp.getValue())) {
      ctx.values[constOp.getResult()] = boolAttr.getValue() ? "true" : "false";
      return success();
    }
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      SmallString<32> value;
      intAttr.getValue().toStringSigned(value);
      ctx.values[constOp.getResult()] = value.str().str();
      return success();
    }
    return constOp.emitOpError("unsupported arith.constant for smt-to-smtlib");
  }

  LogicalResult emitUnrealizedCast(UnrealizedConversionCastOp castOp, EvalContext &ctx) {
    if (castOp->getNumOperands() != 1 || castOp->getNumResults() != 1) {
      return castOp.emitError("smt-to-smtlib only supports one-to-one unrealized casts");
    }
    auto input = lookup(castOp.getOperand(0), ctx);
    if (failed(input)) {
      return castOp.emitError("missing SMTLIB expression for cast input");
    }
    ctx.values[castOp.getResult(0)] = *input;
    return success();
  }

  LogicalResult emitCall(func::CallOp callOp, EvalContext &ctx) {
    auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
    if (!callee) {
      return callOp.emitOpError("smt-to-smtlib could not resolve callee");
    }

    SmallVector<std::string> argExprs;
    argExprs.reserve(callOp.getNumOperands());
    for (Value operand : callOp.getOperands()) {
      auto arg = lookup(operand, ctx);
      if (failed(arg)) {
        return callOp.emitOpError("missing SMTLIB expression for call operand");
      }
      argExprs.push_back(std::move(*arg));
    }

    std::optional<std::string> label = getStageLabel(callOp.getCallee());
    auto helperMode = classifyHelperMode(callee);
    if (failed(helperMode)) {
      return failure();
    }

    if (*helperMode == HelperMode::PureFunction) {
      if (callOp.getNumResults() != 1) {
        callOp.emitError("pure SMT helper calls must return exactly one value");
        return failure();
      }
      if (failed(ensurePureHelperEmitted(callee))) {
        callOp.emitError("smt-to-smtlib requires an emit-compatible helper callee");
        return failure();
      }
      ctx.values[callOp.getResult(0)] = buildHelperApplication(callee, argExprs);
      if (label) {
        ctx.pendingStageLabel = *label;
      }
      return success();
    }

    auto results = inlineHelper(callee, argExprs, label);
    if (failed(results)) {
      callOp.emitError("smt-to-smtlib requires an emit-compatible helper callee");
      return failure();
    }

    if (results->size() != callOp.getNumResults()) {
      return callOp.emitOpError("helper result arity mismatch during SMTLIB export");
    }
    for (auto [result, expr] : llvm::zip(callOp.getResults(), *results)) {
      ctx.values[result] = expr;
    }

    if (label && callOp.getNumResults() == 1) {
      ctx.pendingStageLabel = *label;
    }
    return success();
  }

  /// Classify a helper as either a real SMT-LIB function or an inline script.
  ///
  /// `PureFunction` helpers are emitted once as `define-fun` and referenced from
  /// callers using ordinary SMT-LIB application syntax. A helper stays pure only
  /// if it returns exactly one value, contains no script-level SMT operations,
  /// makes no zero-result calls, and only depends on other pure helpers.
  ///
  /// Everything else becomes `InlineScript`, meaning the callee body is expanded
  /// at each call site so commands such as `assert`, `push`, `pop`, or `check`
  /// remain in the surrounding linear SMT-LIB script.
  FailureOr<HelperMode> classifyHelperMode(func::FuncOp func) {
    if (!func || func.empty()) {
      func.emitError("smt-to-smtlib requires non-empty helper funcs");
      return failure();
    }
    if (func.getFunctionType().getNumResults() != 1) {
      return HelperMode::InlineScript;
    }

    auto it = helperModes.find(func.getOperation());
    if (it != helperModes.end()) {
      return it->second;
    }

    auto inserted = helperModesInProgress.insert(func.getOperation());
    if (!inserted.second) {
      return HelperMode::PureFunction;
    }
    auto cleanup = llvm::make_scope_exit([&] { helperModesInProgress.erase(func.getOperation()); });

    for (Operation &op : func.getBody().front().without_terminator()) {
      if (isa<llzk::smt::SetLogicOp, llzk::smt::DeclareFunOp, llzk::smt::AssertOp,
              llzk::smt::ResetOp, llzk::smt::PushOp, llzk::smt::PopOp, llzk::smt::CheckOp,
              llzk::smt::SolverOp>(op)) {
        helperModes[func.getOperation()] = HelperMode::InlineScript;
        return HelperMode::InlineScript;
      }
      if (auto callOp = dyn_cast<func::CallOp>(op)) {
        if (callOp.getNumResults() == 0) {
          helperModes[func.getOperation()] = HelperMode::InlineScript;
          return HelperMode::InlineScript;
        }
        auto nestedFunc = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
        if (!nestedFunc) {
          callOp.emitOpError("smt-to-smtlib could not resolve callee");
          return failure();
        }
        auto nestedMode = classifyHelperMode(nestedFunc);
        if (failed(nestedMode)) {
          return failure();
        }
        if (*nestedMode == HelperMode::InlineScript) {
          helperModes[func.getOperation()] = HelperMode::InlineScript;
          return HelperMode::InlineScript;
        }
      }
    }

    helperModes[func.getOperation()] = HelperMode::PureFunction;
    return HelperMode::PureFunction;
  }

  std::string getHelperSymbol(func::FuncOp func) {
    auto it = helperSymbols.find(func.getOperation());
    if (it != helperSymbols.end()) {
      return it->second;
    }

    std::string symbol = sanitizeSymbol(func.getSymName());
    if (unsigned &count = emittedSymbolCounts[symbol]; count++ != 0) {
      symbol += "_" + std::to_string(count);
    }
    helperSymbols[func.getOperation()] = symbol;
    return symbol;
  }

  std::string buildHelperApplication(func::FuncOp func, ArrayRef<std::string> argExprs) {
    std::string expr = "(" + getHelperSymbol(func);
    for (const std::string &argExpr : argExprs) {
      expr.push_back(' ');
      expr += argExpr;
    }
    expr.push_back(')');
    return expr;
  }

  LogicalResult ensurePureHelperEmitted(func::FuncOp func) {
    if (emittedPureHelpers.contains(func.getOperation())) {
      return success();
    }
    if (!func || func.empty()) {
      func.emitError("smt-to-smtlib requires non-empty helper funcs");
      return failure();
    }
    if (func.getFunctionType().getNumResults() != 1) {
      func.emitError("pure SMT helper definitions require exactly one return value");
      return failure();
    }
    if (!activePureHelpers.insert(func.getOperation()).second) {
      func.emitError("recursive pure helper calls are not supported by smt-to-smtlib");
      return failure();
    }
    auto cleanup = llvm::make_scope_exit([&] { activePureHelpers.erase(func.getOperation()); });

    for (Operation &op : func.getBody().front().without_terminator()) {
      if (auto callOp = dyn_cast<func::CallOp>(op); callOp && callOp.getNumResults() != 0) {
        auto nestedFunc = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
        if (!nestedFunc) {
          callOp.emitOpError("smt-to-smtlib could not resolve callee");
          return failure();
        }
        auto nestedMode = classifyHelperMode(nestedFunc);
        if (failed(nestedMode)) {
          return failure();
        }
        if (*nestedMode != HelperMode::PureFunction) {
          func.emitError("pure helper depends on a script-style helper");
          return failure();
        }
        if (failed(ensurePureHelperEmitted(nestedFunc))) {
          return failure();
        }
      }
    }

    SmallVector<std::string> argExprs;
    argExprs.reserve(func.getNumArguments());
    std::string funcName = getHelperSymbol(func);
    os << "(define-fun " << funcName << " (";
    for (auto [index, arg] : llvm::enumerate(func.getArguments())) {
      if (index != 0) {
        os << ' ';
      }
      std::string argName = funcName + "__arg" + std::to_string(index);
      argExprs.push_back(argName);
      os << "(" << argName << " " << sortForType(arg.getType()) << ")";
    }
    os << ") " << sortForType(func.getResultTypes().front()) << " ";

    auto results = evalHelper(func, argExprs);
    if (failed(results) || results->size() != 1) {
      func.emitError("smt-to-smtlib requires an emit-compatible pure helper");
      return failure();
    }
    os << results->front() << ")\n";
    emittedPureHelpers.insert(func.getOperation());
    return success();
  }

  /// Expand a script-style helper directly into the caller's linearized script.
  ///
  /// This binds callee block arguments to already-rendered caller expressions,
  /// emits the helper body in source order, and then returns the rendered
  /// `func.return` operands back to the caller for subsequent substitution.
  FailureOr<SmallVector<std::string>> inlineHelper(
      func::FuncOp func, ArrayRef<std::string> argExprs,
      std::optional<std::string> initialStageLabel
  ) {
    if (!func || func.empty()) {
      func.emitError("smt-to-smtlib requires non-empty helper funcs");
      return failure();
    }
    if (!activeInlineHelpers.insert(func.getOperation()).second) {
      func.emitError("recursive helper calls are not supported by smt-to-smtlib");
      return failure();
    }
    auto cleanup = llvm::make_scope_exit([&] { activeInlineHelpers.erase(func.getOperation()); });

    EvalContext helperCtx;
    helperCtx.preserveSharing = helperIsPurelyExpressionBased(func);
    for (auto [arg, expr] : llvm::zip(func.getArguments(), argExprs)) {
      helperCtx.values[arg] = expr;
    }
    if (initialStageLabel) {
      helperCtx.pendingStageLabel = *initialStageLabel;
    }

    if (failed(emitBlock(func.getBody().front(), helperCtx))) {
      return failure();
    }

    auto returnOp = dyn_cast<func::ReturnOp>(func.getBody().front().getTerminator());
    if (!returnOp) {
      func.emitError("helper func must terminate with func.return");
      return failure();
    }

    SmallVector<std::string> results;
    results.reserve(returnOp.getNumOperands());
    for (Value operand : returnOp.getOperands()) {
      auto expr = lookup(operand, helperCtx);
      if (failed(expr)) {
        return failure();
      }
      results.push_back(wrapWithLets(*expr, helperCtx.letBindings));
    }
    return results;
  }

  std::optional<std::string> getStageLabel(StringRef callee) const {
    auto makeIndexedLabel = [&](StringRef stem) -> std::optional<std::string> {
      std::string marker = (Twine("_") + stem + "_part_").str();
      size_t markerPos = callee.rfind(marker);
      if (markerPos == StringRef::npos) {
        return std::nullopt;
      }
      StringRef suffix = callee.drop_front(markerPos + marker.size());
      unsigned index = 0;
      if (suffix.empty() || suffix.getAsInteger(10, index)) {
        return std::nullopt;
      }
      return (Twine(stem) + "[" + Twine(index) + "]").str();
    };
    if (auto indexed = makeIndexedLabel("target")) {
      return indexed;
    }
    if (auto indexed = makeIndexedLabel("post")) {
      return indexed;
    }
    if (callee.ends_with("_pre")) {
      return "pre";
    }
    if (callee.ends_with("_target")) {
      return "target";
    }
    if (callee.ends_with("_post")) {
      return "post";
    }
    return std::nullopt;
  }

  FailureOr<CheckSuccessInfo> identifySuccessRegion(llzk::smt::CheckOp checkOp) {
    SmallVector<std::pair<StringRef, Region *>> regions = {
        {"sat", &checkOp.getSatRegion()},
        {"unknown", &checkOp.getUnknownRegion()},
        {"unsat", &checkOp.getUnsatRegion()},
    };

    Region *successRegion = nullptr;
    StringRef outcome;
    for (auto [candidateOutcome, region] : regions) {
      if (isSuccessRegion(*region)) {
        if (successRegion != nullptr) {
          checkOp.emitError("smt-to-smtlib requires exactly one non-failing smt.check region");
          return failure();
        }
        successRegion = region;
        outcome = candidateOutcome;
      }
    }

    if (successRegion == nullptr) {
      checkOp.emitError("smt-to-smtlib could not determine successful smt.check path");
      return failure();
    }
    return CheckSuccessInfo {outcome, successRegion};
  }

  bool isSuccessRegion(Region &region) const {
    if (!llvm::hasSingleElement(region)) {
      return false;
    }
    for (Operation &op : region.front().without_terminator()) {
      if (isa<llzk::boolean::AssertOp>(op)) {
        return false;
      }
    }
    return true;
  }

  LogicalResult emitCheck(llzk::smt::CheckOp checkOp, EvalContext &ctx) {
    if (checkOp.getNumResults() != 0) {
      return checkOp.emitOpError("smt-to-smtlib does not support result-producing smt.check");
    }
    auto successInfo = identifySuccessRegion(checkOp);
    if (failed(successInfo)) {
      return failure();
    }

    auto formatStageLabel = [&](StringRef rawLabel) -> std::string {
      if (rawLabel == "target" || rawLabel.starts_with("target[")) {
        return (Twine("target[") + Twine(nextTargetStageIndex++) + "]").str();
      }
      if (rawLabel == "post" || rawLabel.starts_with("post[")) {
        return (Twine("post[") + Twine(nextPostStageIndex++) + "]").str();
      }
      return rawLabel.str();
    };

    if (ctx.pendingStageLabel) {
      os << "(set-info :llzk-stage \"" << formatStageLabel(*ctx.pendingStageLabel) << "\")\n";
    }
    os << "(set-info :status " << successInfo->expectedOutcome << ")\n";
    os << "(check-sat)\n";
    ctx.pendingStageLabel.reset();
    return emitBlock(successInfo->successRegion->front(), ctx);
  }

  LogicalResult emitSuccessEffects(llzk::smt::CheckOp checkOp, EvalContext &ctx) {
    auto successInfo = identifySuccessRegion(checkOp);
    if (failed(successInfo)) {
      return failure();
    }
    for (Operation &op : successInfo->successRegion->front().without_terminator()) {
      if (isa<llzk::smt::PushOp, llzk::smt::PopOp>(op)) {
        continue;
      }
      if (failed(emitOperation(&op, ctx))) {
        return failure();
      }
    }
    return success();
  }

  FailureOr<std::string> lookup(Value value, EvalContext &ctx) {
    auto it = ctx.values.find(value);
    if (it == ctx.values.end()) {
      if (Operation *def = value.getDefiningOp()) {
        def->emitError("missing SMTLIB value binding");
      } else {
        module.emitError() << "missing SMTLIB value binding for block argument";
      }
      return failure();
    }
    return it->second;
  }

  FailureOr<SmallVector<std::string>>
  evalHelper(func::FuncOp func, ArrayRef<std::string> argExprs) {
    if (!func || func.empty()) {
      func.emitError("smt-to-smtlib requires non-empty helper funcs");
      return failure();
    }

    EvalContext ctx;
    ctx.preserveSharing = helperIsPurelyExpressionBased(func);
    if (func.getNumArguments() != argExprs.size()) {
      func.emitError("helper argument arity mismatch during SMTLIB export");
      return failure();
    }
    for (auto [arg, expr] : llvm::zip(func.getArguments(), argExprs)) {
      ctx.values[arg] = expr;
    }

    Block &block = func.getBody().front();
    for (Operation &op : block.without_terminator()) {
      if (isa<llzk::smt::DeclareFunOp, llzk::smt::AssertOp, llzk::smt::PushOp, llzk::smt::PopOp,
              llzk::smt::CheckOp>(op)) {
        op.emitError("script-style SMT ops cannot appear in pure helper definitions");
        return failure();
      }
      if (failed(emitOperation(&op, ctx))) {
        return failure();
      }
    }

    auto returnOp = dyn_cast<func::ReturnOp>(block.getTerminator());
    if (!returnOp) {
      func.emitError("helper func must terminate with func.return");
      return failure();
    }

    SmallVector<std::string> results;
    results.reserve(returnOp.getNumOperands());
    for (Value operand : returnOp.getOperands()) {
      auto expr = lookup(operand, ctx);
      if (failed(expr)) {
        return failure();
      }
      results.push_back(wrapWithLets(*expr, ctx.letBindings));
    }
    return results;
  }

  bool helperIsPurelyExpressionBased(func::FuncOp func) const {
    for (Operation &op : func.getBody().front().without_terminator()) {
      if (isa<llzk::smt::DeclareFunOp, llzk::smt::AssertOp, llzk::smt::PushOp, llzk::smt::PopOp,
              llzk::smt::CheckOp>(op)) {
        return false;
      }
      if (auto callOp = dyn_cast<func::CallOp>(op); callOp && callOp.getNumResults() == 0) {
        return false;
      }
    }
    return true;
  }

  FailureOr<std::string> buildExpr(Operation *op, EvalContext &ctx) {
    return TypeSwitch<Operation *, FailureOr<std::string>>(op)
        .Case<llzk::smt::BoolConstantOp>([](auto constOp) {
      return std::string(constOp.getValue() ? "true" : "false");
    })
        .Case<llzk::smt::IntConstantOp>([](auto constOp) {
      SmallString<32> str;
      constOp.getValue().toStringSigned(str);
      return str.str().str();
    })
        .Case<llzk::smt::BVConstantOp>([](auto constOp) {
      return constOp.getValue().getValueAsString();
    })
        .Case<llzk::smt::EqOp>([&](auto exprOp) { return buildSExpr("=", exprOp.getInputs(), ctx); }
        )
        .Case<llzk::smt::DistinctOp>([&](auto exprOp) {
      return buildSExpr("distinct", exprOp.getInputs(), ctx);
    })
        .Case<llzk::smt::NotOp>([&](auto exprOp) {
      return buildSExpr("not", ValueRange {exprOp.getInput()}, ctx);
    })
        .Case<llzk::smt::AndOp>([&](auto exprOp) {
      return buildSExpr("and", exprOp.getInputs(), ctx);
    })
        .Case<llzk::smt::OrOp>([&](auto exprOp) {
      return buildSExpr("or", exprOp.getInputs(), ctx);
    })
        .Case<llzk::smt::XOrOp>([&](auto exprOp) {
      return buildSExpr("xor", exprOp.getInputs(), ctx);
    })
        .Case<llzk::smt::ImpliesOp>([&](auto exprOp) {
      return buildSExpr("=>", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<llzk::smt::IteOp>([&](auto exprOp) {
      return buildSExpr(
          "ite", ValueRange {exprOp.getCond(), exprOp.getThenValue(), exprOp.getElseValue()}, ctx
      );
    })
        .Case<llzk::smt::IntNegOp>([&](auto exprOp) {
      return buildSExpr("-", ValueRange {exprOp.getInput()}, ctx);
    })
        .Case<llzk::smt::IntAbsOp>([&](auto exprOp) {
      return buildSExpr("abs", ValueRange {exprOp.getInput()}, ctx);
    })
        .Case<llzk::smt::IntAddOp>([&](auto exprOp) {
      return buildSExpr("+", exprOp.getInputs(), ctx);
    })
        .Case<llzk::smt::IntMulOp>([&](auto exprOp) {
      return buildSExpr("*", exprOp.getInputs(), ctx);
    })
        .Case<llzk::smt::IntSubOp>([&](auto exprOp) {
      return buildSExpr("-", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<llzk::smt::IntDivOp>([&](auto exprOp) {
      return buildSExpr("div", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<llzk::smt::IntModOp>([&](auto exprOp) {
      return buildSExpr("mod", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<llzk::smt::IntCmpOp>([&](auto cmpOp) { return buildCmpExpr(cmpOp, ctx); })
        .Case<llzk::smt::Int2BVOp>([&](auto exprOp) { return buildInt2BVExpr(exprOp, ctx); })
        .Case<llzk::smt::BV2IntOp>([&](auto exprOp) { return buildBV2IntExpr(exprOp, ctx); })
        .Case<llzk::smt::BVNegOp>([&](auto exprOp) {
      return buildSExpr("bvneg", ValueRange {exprOp.getInput()}, ctx);
    })
        .Case<llzk::smt::BVAndOp>([&](auto exprOp) {
      return buildSExpr("bvand", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<llzk::smt::BVAddOp>([&](auto exprOp) {
      return buildSExpr("bvadd", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<llzk::smt::BVMulOp>([&](auto exprOp) {
      return buildSExpr("bvmul", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<llzk::smt::BVUDivOp>([&](auto exprOp) {
      return buildSExpr("bvudiv", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<llzk::smt::BVSDivOp>([&](auto exprOp) {
      return buildSExpr("bvsdiv", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<llzk::smt::BVURemOp>([&](auto exprOp) {
      return buildSExpr("bvurem", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<llzk::smt::BVSRemOp>([&](auto exprOp) {
      return buildSExpr("bvsrem", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<llzk::smt::BVSModOp>([&](auto exprOp) {
      return buildSExpr("bvsmod", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<llzk::smt::BVOrOp>([&](auto exprOp) {
      return buildSExpr("bvor", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<llzk::smt::BVXOrOp>([&](auto exprOp) {
      return buildSExpr("bvxor", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<llzk::smt::BVNotOp>([&](auto exprOp) {
      return buildSExpr("bvnot", ValueRange {exprOp.getInput()}, ctx);
    })
        .Case<llzk::smt::BVShlOp>([&](auto exprOp) {
      return buildSExpr("bvshl", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<llzk::smt::BVLShrOp>([&](auto exprOp) {
      return buildSExpr("bvlshr", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<llzk::smt::BVAShrOp>([&](auto exprOp) {
      return buildSExpr("bvashr", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<llzk::smt::BVCmpOp>([&](auto exprOp) { return buildBVCmpExpr(exprOp, ctx); })
        .Case<llzk::smt::ConcatOp>([&](auto exprOp) {
      return buildSExpr("concat", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<llzk::smt::ExtractOp>([&](auto exprOp) { return buildExtractExpr(exprOp, ctx); })
        .Case<llzk::smt::RepeatOp>([&](auto exprOp) { return buildRepeatExpr(exprOp, ctx); })
        .Case<llzk::smt::ArraySelectOp>([&](auto exprOp) {
      return buildSExpr("select", ValueRange {exprOp.getArray(), exprOp.getIndex()}, ctx);
    })
        .Case<llzk::smt::ArrayStoreOp>([&](auto exprOp) {
      return buildSExpr(
          "store", ValueRange {exprOp.getArray(), exprOp.getIndex(), exprOp.getValue()}, ctx
      );
    })
        .Case<llzk::smt::ArrayBroadcastOp>([&](auto exprOp) -> FailureOr<std::string> {
      auto valueExpr = lookup(exprOp.getValue(), ctx);
      if (failed(valueExpr)) {
        return failure();
      }
      return "((as const " + sortForType(exprOp.getType()) + ") " + *valueExpr + ")";
    })
        .Case<llzk::smt::ApplyFuncOp>([&](auto exprOp) -> FailureOr<std::string> {
      auto funcExpr = lookup(exprOp.getFunc(), ctx);
      if (failed(funcExpr)) {
        return failure();
      }
      if (exprOp.getArgs().empty()) {
        return *funcExpr;
      }
      std::string expr = "(" + *funcExpr;
      for (Value arg : exprOp.getArgs()) {
        auto argExpr = lookup(arg, ctx);
        if (failed(argExpr)) {
          return failure();
        }
        expr.push_back(' ');
        expr += *argExpr;
      }
      expr.push_back(')');
      return expr;
    })
        .Case<llzk::smt::ForallOp>([&](auto exprOp) {
      return buildQuantifierExpr("forall", exprOp, ctx);
    })
        .Case<llzk::smt::ExistsOp>([&](auto exprOp) {
      return buildQuantifierExpr("exists", exprOp, ctx);
    }).Case<arith::ConstantOp>([&](auto constOp) {
      return buildArithConstantExpr(constOp);
    }).Default([&](Operation *unknownOp) {
      unknownOp->emitError("unsupported expression op in smt-to-smtlib");
      return failure();
    });
  }

  FailureOr<std::string> buildArithConstantExpr(arith::ConstantOp constOp) {
    if (auto boolAttr = dyn_cast<BoolAttr>(constOp.getValue())) {
      return std::string(boolAttr.getValue() ? "true" : "false");
    }
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      SmallString<32> value;
      intAttr.getValue().toStringSigned(value);
      return value.str().str();
    }
    constOp.emitOpError("unsupported arith.constant expression");
    return failure();
  }

  FailureOr<std::string> buildCmpExpr(llzk::smt::IntCmpOp cmpOp, EvalContext &ctx) {
    StringRef pred;
    switch (cmpOp.getPred()) {
    case llzk::smt::IntPredicate::lt:
      pred = "<";
      break;
    case llzk::smt::IntPredicate::le:
      pred = "<=";
      break;
    case llzk::smt::IntPredicate::gt:
      pred = ">";
      break;
    case llzk::smt::IntPredicate::ge:
      pred = ">=";
      break;
    }
    return buildSExpr(pred, ValueRange {cmpOp.getLhs(), cmpOp.getRhs()}, ctx);
  }

  FailureOr<std::string> buildInt2BVExpr(llzk::smt::Int2BVOp op, EvalContext &ctx) {
    auto input = lookup(op.getInput(), ctx);
    if (failed(input)) {
      return failure();
    }
    auto resultType = cast<llzk::smt::BitVectorType>(op.getResult().getType());
    return "((_ int_to_bv " + std::to_string(resultType.getWidth()) + ") " + *input + ")";
  }

  FailureOr<std::string> buildBV2IntExpr(llzk::smt::BV2IntOp op, EvalContext &ctx) {
    auto input = lookup(op.getInput(), ctx);
    if (failed(input)) {
      return failure();
    }
    return "(" + std::string(op.getIsSigned() ? "sbv_to_int" : "ubv_to_int") + " " + *input + ")";
  }

  FailureOr<std::string> buildBVCmpExpr(llzk::smt::BVCmpOp cmpOp, EvalContext &ctx) {
    StringRef pred;
    switch (cmpOp.getPred()) {
    case llzk::smt::BVCmpPredicate::slt:
      pred = "bvslt";
      break;
    case llzk::smt::BVCmpPredicate::sle:
      pred = "bvsle";
      break;
    case llzk::smt::BVCmpPredicate::sgt:
      pred = "bvsgt";
      break;
    case llzk::smt::BVCmpPredicate::sge:
      pred = "bvsge";
      break;
    case llzk::smt::BVCmpPredicate::ult:
      pred = "bvult";
      break;
    case llzk::smt::BVCmpPredicate::ule:
      pred = "bvule";
      break;
    case llzk::smt::BVCmpPredicate::ugt:
      pred = "bvugt";
      break;
    case llzk::smt::BVCmpPredicate::uge:
      pred = "bvuge";
      break;
    }
    return buildSExpr(pred, ValueRange {cmpOp.getLhs(), cmpOp.getRhs()}, ctx);
  }

  FailureOr<std::string> buildExtractExpr(llzk::smt::ExtractOp op, EvalContext &ctx) {
    auto input = lookup(op.getInput(), ctx);
    if (failed(input)) {
      return failure();
    }
    unsigned lowBit = op.getLowBit();
    unsigned highBit = lowBit + cast<llzk::smt::BitVectorType>(op.getType()).getWidth() - 1;
    return "((_ extract " + std::to_string(highBit) + " " + std::to_string(lowBit) + ") " + *input +
           ")";
  }

  FailureOr<std::string> buildRepeatExpr(llzk::smt::RepeatOp op, EvalContext &ctx) {
    auto input = lookup(op.getInput(), ctx);
    if (failed(input)) {
      return failure();
    }
    return "((_ repeat " + std::to_string(op.getCount()) + ") " + *input + ")";
  }

  FailureOr<std::string> buildSExpr(StringRef opName, ValueRange operands, EvalContext &ctx) {
    if (opName == "and") {
      SmallVector<std::string> renderedOperands;
      renderedOperands.reserve(operands.size());
      for (Value operand : operands) {
        auto value = lookup(operand, ctx);
        if (failed(value)) {
          return failure();
        }
        if (*value == "false") {
          return std::string("false");
        }
        if (*value == "true") {
          continue;
        }
        renderedOperands.push_back(std::move(*value));
      }
      if (renderedOperands.empty()) {
        return std::string("true");
      }
      if (renderedOperands.size() == 1) {
        return renderedOperands.front();
      }
      std::string expr = "(and";
      for (const std::string &operand : renderedOperands) {
        expr.push_back(' ');
        expr += operand;
      }
      expr.push_back(')');
      return expr;
    }

    std::string expr = "(" + opName.str();
    for (Value operand : operands) {
      auto value = lookup(operand, ctx);
      if (failed(value)) {
        return failure();
      }
      expr.push_back(' ');
      expr += *value;
    }
    expr.push_back(')');
    return expr;
  }

  template <typename QuantifierOpTy>
  FailureOr<std::string>
  buildQuantifierExpr(StringRef quantifierName, QuantifierOpTy op, EvalContext &ctx) {
    if (!op.getPatterns().empty()) {
      op.emitError("smt-to-smtlib does not yet support quantified pattern emission");
      return failure();
    }
    if (!llvm::hasSingleElement(op.getBody())) {
      op.emitError("smt-to-smtlib requires quantifier bodies with a single block");
      return failure();
    }

    EvalContext bodyCtx = ctx;
    bodyCtx.letBindings.clear();
    Block &body = op.getBody().front();
    std::string expr = "(" + quantifierName.str() + " (";
    auto namesAttr = op.getBoundVarNames();
    for (auto [index, arg] : llvm::enumerate(body.getArguments())) {
      std::string name = namesAttr && index < namesAttr->size()
                             ? sanitizeSymbol(cast<StringAttr>((*namesAttr)[index]).getValue())
                             : "q" + std::to_string(nextTempId++);
      bodyCtx.values[arg] = name;
      if (index != 0) {
        expr.push_back(' ');
      }
      expr += "(" + name + " " + sortForType(arg.getType()) + ")";
    }
    expr += ") ";

    for (Operation &nestedOp : body.without_terminator()) {
      if (failed(emitOperation(&nestedOp, bodyCtx))) {
        return failure();
      }
    }

    auto yieldOp = dyn_cast<llzk::smt::YieldOp>(body.getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
      op.emitError("smt-to-smtlib requires quantifier bodies to yield exactly one value");
      return failure();
    }
    auto yielded = lookup(yieldOp.getOperand(0), bodyCtx);
    if (failed(yielded)) {
      return failure();
    }
    expr += wrapWithLets(*yielded, bodyCtx.letBindings);
    expr += ")";
    return expr;
  }

  std::string makeLetName() { return "__let" + std::to_string(nextTempId++); }

  static std::string
  wrapWithLets(std::string expr, ArrayRef<std::pair<std::string, std::string>> letBindings) {
    for (const auto &binding : llvm::reverse(letBindings)) {
      expr = "(let ((" + binding.first + " " + binding.second + ")) " + expr + ")";
    }
    return expr;
  }

  ModuleOp module;
  llvm::raw_ostream &os;
  unsigned nextTempId = 0;
  unsigned nextTargetStageIndex = 0;
  unsigned nextPostStageIndex = 0;
  unsigned pushDepth = 0;
  DenseSet<Operation *> activePureHelpers;
  DenseSet<Operation *> activeInlineHelpers;
  DenseSet<Operation *> helperModesInProgress;
  DenseMap<Operation *, HelperMode> helperModes;
  llvm::StringMap<unsigned> emittedSymbolCounts;
  llvm::StringSet<> emittedAssertions;
  DenseMap<Operation *, std::string> helperSymbols;
  DenseSet<Operation *> emittedPureHelpers;
};

class SMTDialectToSMTLIBPass
    : public llzk::smt::impl::SMTDialectToSMTLIBPassBase<SMTDialectToSMTLIBPass> {
  using Base = llzk::smt::impl::SMTDialectToSMTLIBPassBase<SMTDialectToSMTLIBPass>;
  using Base::Base;

  void runOnOperation() override {
    raw_ostream *stream = &llvm::outs();
    std::unique_ptr<llvm::ToolOutputFile> outputFile;
    if (!outputFilename.empty() && outputFilename != "-") {
      outputFile = openOutputFile(outputFilename);
      if (!outputFile) {
        signalPassFailure();
        return;
      }
      stream = &outputFile->os();
    }
    if (failed(llzk::smt::emitSMTLIBModule(getOperation(), *stream))) {
      signalPassFailure();
      return;
    }

    if (outputFile) {
      outputFile->keep();
    }
  }
};

} // namespace

/// Export the selected SMT solver rooted in `module` to SMT-LIB text.
LogicalResult llzk::smt::emitSMTLIBModule(ModuleOp module, llvm::raw_ostream &os) {
  return SMTLIBEmitter(module, os).emit();
}
