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

#include <functional>
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
  if (name.empty()) {
    return "tmp";
  }

  std::string out;
  out.reserve(name.size());
  for (char ch : name) {
    if (llvm::isAlnum(ch) || ch == '_' || ch == '.' || ch == '$') {
      out.push_back(ch);
    } else {
      out.push_back('_');
    }
  }
  if (llvm::isDigit(out.front())) {
    out.insert(out.begin(), '_');
  }
  return out;
}

/// Append an SMT dialect sort using SMT-LIB textual syntax.
static void printSortForType(llvm::raw_ostream &os, Type type) {
  TypeSwitch<Type>(type)
      .Case<llzk::smt::IntType>([&os](auto) { os << "Int"; })
      .Case<llzk::smt::BoolType>([&os](auto) { os << "Bool"; })
      .Case<llzk::smt::SMTFuncType>([&os](auto funcType) {
    os << "((";
    llvm::interleave(funcType.getDomainTypes(), [&os](Type domainType) {
      printSortForType(os, domainType);
    }, [&os] { os << ' '; });
    os << ") ";
    printSortForType(os, funcType.getRangeType());
    os << ')';
  })
      .Case<llzk::smt::BitVectorType>([&os](auto bvType) {
    os << "(_ BitVec " << bvType.getWidth() << ')';
  })
      .Case<llzk::smt::ArrayType>([&os](auto arrayType) {
    os << "(Array ";
    printSortForType(os, arrayType.getDomainType());
    os << ' ';
    printSortForType(os, arrayType.getRangeType());
    os << ')';
  }).Default([](Type) { llvm::report_fatal_error("unsupported SMTLIB sort in smt-to-smtlib"); });
}

/// Convert an SMT dialect sort to its SMT-LIB textual form.
static std::string sortForType(Type type) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  printSortForType(os, type);
  return storage;
}

/// Print a structured SMT-LIB `set-info` value attribute.
static void printSetInfoValue(llvm::raw_ostream &os, Attribute value) {
  TypeSwitch<Attribute>(value)
      .Case<llzk::smt::KeywordAttr>([&os](auto keywordAttr) { os << keywordAttr.getValue(); })
      .Case<llzk::smt::SymbolAttr>([&os](auto symbolAttr) { os << symbolAttr.getValue(); })
      .Case<StringAttr>([&os](auto strAttr) { strAttr.print(os); })
      .Case<BoolAttr>([&os](auto boolAttr) { os << (boolAttr.getValue() ? "true" : "false"); })
      .Case<IntegerAttr>([&os](auto intAttr) {
    SmallString<32> valueText;
    intAttr.getValue().toStringSigned(valueText);
    os << valueText;
  }).Case<ArrayAttr>([&os](auto arrayAttr) {
    os << '(';
    llvm::interleave(arrayAttr, [&os](Attribute element) {
      printSetInfoValue(os, element);
    }, [&os] { os << ' '; });
    os << ')';
  });
}

/// Lowering mode for a helper `func.func` referenced from the script root.
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
  SMTLIBEmitter(ModuleOp moduleOp, llvm::raw_ostream &outputStream)
      : module(moduleOp), os(outputStream) {}

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
    SmallVector<std::pair<std::string, std::string>> letBindings;
    bool preserveSharing = false;
  };

  struct PureHelperDefinition {
    std::string symbol;
    SmallVector<std::string> argNames;
    SmallVector<std::string> argSorts;
    std::string resultSort;
    std::string bodyExpr;
  };

  ModuleOp getEffectiveRootModule() {
    ModuleOp nestedModule;
    for (Operation &op : module.getBody()->getOperations()) {
      auto childModule = dyn_cast<ModuleOp>(op);
      if (!childModule || nestedModule) {
        return module;
      }
      nestedModule = childModule;
    }
    return nestedModule ? nestedModule : module;
  }

  FailureOr<llzk::smt::SolverOp> collectRoot() {
    selectedRootModule = getEffectiveRootModule();
    llzk::smt::SolverOp solver;
    for (Operation &op : selectedRootModule.getBody()->getOperations()) {
      auto solverOp = dyn_cast<llzk::smt::SolverOp>(op);
      if (!solverOp) {
        continue;
      }
      if (solver) {
        return selectedRootModule.emitError(
            "smt-to-smtlib requires exactly one top-level smt.solver in the root module"
        );
      }
      solver = solverOp;
    }
    if (!solver) {
      return selectedRootModule.emitError(
          "smt-to-smtlib could not find a top-level smt.solver in the root "
          "module"
      );
    }
    return solver;
  }

  ModuleOp getSelectedRootModule() {
    return selectedRootModule ? selectedRootModule : getEffectiveRootModule();
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
    pushDepth = 0;
    emittedSymbolCounts.clear();
    emittedAssertions.clear();
    helperSymbols.clear();
    emittedPureHelpers.clear();
    emittedPureHelperSCCs.clear();
  }

  LogicalResult emitRootPreamble(bool emitReset, bool emitDefaultLogic) {
    if (emitReset) {
      os << "\n(reset)\n";
    }
    resetScriptState();
    if (emitDefaultLogic) {
      os << "(set-logic ALL)\n";
    }
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
        .Case<llzk::smt::SetLogicOp>([this](auto setLogicOp) {
      os << "(set-logic " << setLogicOp.getLogic() << ")\n";
      return success();
    })
        .Case<llzk::smt::SetInfoOp>([this](auto setInfoOp) {
      os << "(set-info " << setInfoOp.getKey().getValue() << ' ';
      printSetInfoValue(os, setInfoOp.getValueAttr());
      os << ")\n";
      return success();
    })
        .Case<llzk::smt::DeclareFunOp>([&](auto declareOp) { return emitDeclare(declareOp, ctx); })
        .Case<llzk::smt::AssertOp>([&](auto assertOp) { return emitAssert(assertOp, ctx); })
        .Case<llzk::smt::ResetOp>([this](auto) {
      os << "(reset)\n";
      resetScriptState();
      return success();
    })
        .Case<llzk::smt::PushOp>([this](auto pushOp) {
      pushDepth += pushOp.getCount();
      os << "(push " << pushOp.getCount() << ")\n";
      return success();
    })
        .Case<llzk::smt::PopOp>([this](auto popOp) {
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
      return assertOp.emitError("boolean.assert is not serializable to SMT-LIB");
    }).Default([&](Operation *unknownOp) {
      return unknownOp->emitError("unsupported operation in smt-to-smtlib");
    });
  }

  template <typename OpTy> LogicalResult bindExpr(OpTy op, EvalContext &ctx) {
    auto expr = buildExpr(op.getOperation(), ctx);
    if (failed(expr)) {
      return failure();
    }
    if (op->getNumResults() != 1) {
      return op.emitError("smt-to-smtlib only supports single-result expression ops");
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
      llvm::interleave(funcType.getDomainTypes(), [this](Type domainType) {
        printSortForType(os, domainType);
      }, [this] { os << ' '; });
      os << ") ";
      printSortForType(os, funcType.getRangeType());
      os << ")\n";
      return success();
    }
    os << "(declare-fun " << symbol << " () ";
    printSortForType(os, declareOp.getType());
    os << ")\n";
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
    auto callee = getSelectedRootModule().lookupSymbol<func::FuncOp>(callOp.getCallee());
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

    auto helperMode = classifyHelperMode(callee);
    if (failed(helperMode)) {
      return failure();
    }

    if (*helperMode == HelperMode::PureFunction) {
      if (callOp.getNumResults() != 1) {
        return callOp.emitError("pure SMT helper calls must return exactly one value");
      }
      if (failed(ensurePureHelperEmitted(callee))) {
        return callOp.emitError("smt-to-smtlib requires an emit-compatible helper callee");
      }
      ctx.values[callOp.getResult(0)] = buildHelperApplication(callee, argExprs);
      return success();
    }

    auto results = inlineHelper(callee, argExprs);
    if (failed(results)) {
      return callOp.emitError("smt-to-smtlib requires an emit-compatible helper callee");
    }

    if (results->size() != callOp.getNumResults()) {
      return callOp.emitOpError("helper result arity mismatch during SMTLIB export");
    }
    for (auto [result, expr] : llvm::zip(callOp.getResults(), *results)) {
      ctx.values[result] = expr;
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
      return func.emitError("smt-to-smtlib requires non-empty helper funcs");
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
    Operation *funcOp = func.getOperation();
    auto cleanup = llvm::make_scope_exit([this, funcOp] { helperModesInProgress.erase(funcOp); });

    for (Operation &op : func.getBody().front().without_terminator()) {
      if (isa<llzk::smt::SetLogicOp, llzk::smt::SetInfoOp, llzk::smt::DeclareFunOp,
              llzk::smt::AssertOp, llzk::smt::ResetOp, llzk::smt::PushOp, llzk::smt::PopOp,
              llzk::smt::CheckOp, llzk::smt::SolverOp>(op)) {
        helperModes[func.getOperation()] = HelperMode::InlineScript;
        return HelperMode::InlineScript;
      }
      if (auto callOp = dyn_cast<func::CallOp>(op)) {
        if (callOp.getNumResults() == 0) {
          helperModes[func.getOperation()] = HelperMode::InlineScript;
          return HelperMode::InlineScript;
        }
        auto nestedFunc = getSelectedRootModule().lookupSymbol<func::FuncOp>(callOp.getCallee());
        if (!nestedFunc) {
          return callOp.emitOpError("smt-to-smtlib could not resolve callee");
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

  /// Return the unique emitted SMT-LIB symbol name for a helper function.
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

  /// Build an SMT-LIB application of an emitted helper symbol.
  std::string buildHelperApplication(func::FuncOp func, ArrayRef<std::string> argExprs) {
    std::string expr;
    llvm::raw_string_ostream exprStream(expr);
    exprStream << '(' << getHelperSymbol(func);
    for (const std::string &argExpr : argExprs) {
      exprStream << ' ' << argExpr;
    }
    exprStream << ')';
    exprStream.flush();
    return expr;
  }

  LogicalResult initializePureHelperSCCs() {
    if (pureHelperSCCsInitialized) {
      return success();
    }
    pureHelperSCCsInitialized = true;

    DenseMap<Operation *, unsigned> indexByOp;
    DenseMap<Operation *, unsigned> lowLinkByOp;
    SmallVector<func::FuncOp> stack;
    DenseSet<Operation *> onStack;
    unsigned nextIndex = 0;

    std::function<LogicalResult(func::FuncOp)> strongConnect = [&](func::FuncOp currentFunc) {
      Operation *funcOp = currentFunc.getOperation();
      indexByOp[funcOp] = nextIndex;
      lowLinkByOp[funcOp] = nextIndex;
      ++nextIndex;
      stack.push_back(currentFunc);
      onStack.insert(funcOp);

      auto callees = collectPureHelperCallees(currentFunc);
      if (failed(callees)) {
        return failure();
      }
      pureHelperCallees.try_emplace(funcOp, callees->begin(), callees->end());

      for (func::FuncOp callee : *callees) {
        Operation *calleeOp = callee.getOperation();
        auto indexIt = indexByOp.find(calleeOp);
        if (indexIt == indexByOp.end()) {
          if (failed(strongConnect(callee))) {
            return failure();
          }
          lowLinkByOp[funcOp] = std::min(lowLinkByOp[funcOp], lowLinkByOp[calleeOp]);
          continue;
        }
        if (onStack.contains(calleeOp)) {
          lowLinkByOp[funcOp] = std::min(lowLinkByOp[funcOp], indexIt->second);
        }
      }

      if (lowLinkByOp[funcOp] != indexByOp[funcOp]) {
        return success();
      }

      SmallVector<func::FuncOp> scc;
      bool hasSelfRecursion = false;
      while (!stack.empty()) {
        func::FuncOp current = stack.pop_back_val();
        Operation *currentOp = current.getOperation();
        onStack.erase(currentOp);
        scc.push_back(current);
        if (current == currentFunc) {
          break;
        }
      }
      llvm::sort(scc, [](func::FuncOp lhs, func::FuncOp rhs) {
        return lhs.getSymName() < rhs.getSymName();
      });
      for (func::FuncOp member : scc) {
        pureHelperSCCId[member.getOperation()] = pureHelperSCCs.size();
      }

      if (scc.size() == 1) {
        auto it = pureHelperCallees.find(funcOp);
        hasSelfRecursion = it != pureHelperCallees.end() &&
                           llvm::any_of(it->second, [currentFunc](func::FuncOp callee) {
          return callee == currentFunc;
        });
      }
      if (hasSelfRecursion || scc.size() > 1) {
        recursivePureHelperSCCs.insert(pureHelperSCCs.size());
      }
      pureHelperSCCs.push_back(std::move(scc));
      return success();
    };

    for (auto helperFunc : getSelectedRootModule().getOps<func::FuncOp>()) {
      auto helperMode = classifyHelperMode(helperFunc);
      if (failed(helperMode)) {
        return failure();
      }
      if (*helperMode != HelperMode::PureFunction ||
          pureHelperSCCId.contains(helperFunc.getOperation()) ||
          indexByOp.contains(helperFunc.getOperation())) {
        continue;
      }
      if (failed(strongConnect(helperFunc))) {
        return failure();
      }
    }

    return success();
  }

  FailureOr<SmallVector<func::FuncOp>> collectPureHelperCallees(func::FuncOp func) {
    SmallVector<func::FuncOp> callees;
    for (Operation &op : func.getBody().front().without_terminator()) {
      auto callOp = dyn_cast<func::CallOp>(op);
      if (!callOp || callOp.getNumResults() == 0) {
        continue;
      }
      auto callee = getSelectedRootModule().lookupSymbol<func::FuncOp>(callOp.getCallee());
      if (!callee) {
        return callOp.emitOpError("smt-to-smtlib could not resolve callee");
      }
      auto helperMode = classifyHelperMode(callee);
      if (failed(helperMode)) {
        return failure();
      }
      if (*helperMode != HelperMode::PureFunction) {
        return func.emitError("pure helper depends on a script-style helper");
      }
      callees.push_back(callee);
    }
    return callees;
  }

  FailureOr<PureHelperDefinition> buildPureHelperDefinition(func::FuncOp func) {
    if (!func || func.empty()) {
      return func.emitError("smt-to-smtlib requires non-empty helper funcs");
    }
    if (func.getFunctionType().getNumResults() != 1) {
      return func.emitError("pure SMT helper definitions require exactly one return value");
    }

    SmallVector<std::string> argExprs;
    PureHelperDefinition definition;
    definition.symbol = getHelperSymbol(func);
    definition.resultSort = sortForType(func.getResultTypes().front());
    argExprs.reserve(func.getNumArguments());
    definition.argNames.reserve(func.getNumArguments());
    definition.argSorts.reserve(func.getNumArguments());
    for (auto [index, arg] : llvm::enumerate(func.getArguments())) {
      std::string argName = definition.symbol + "__arg" + std::to_string(index);
      argExprs.push_back(argName);
      definition.argNames.push_back(std::move(argName));
      definition.argSorts.push_back(sortForType(arg.getType()));
    }

    auto results = evalHelper(func, argExprs);
    if (failed(results) || results->size() != 1) {
      return func.emitError("smt-to-smtlib requires an emit-compatible pure helper");
    }
    definition.bodyExpr = std::move(results->front());
    return definition;
  }

  LogicalResult emitPureHelperDefinition(const PureHelperDefinition &definition) {
    os << "(define-fun " << definition.symbol << " (";
    for (auto [index, argName] : llvm::enumerate(definition.argNames)) {
      if (index != 0) {
        os << ' ';
      }
      os << '(' << argName << ' ' << definition.argSorts[index] << ')';
    }
    os << ") " << definition.resultSort << ' ' << definition.bodyExpr << ")\n";
    return success();
  }

  LogicalResult emitRecursivePureHelperSCC(unsigned sccId) {
    if (emittedPureHelperSCCs.contains(sccId)) {
      return success();
    }
    [[maybe_unused]] bool inserted = activePureHelperSCCs.insert(sccId).second;
    assert(inserted && "recursive SCC should not be re-entered during emission");
    auto cleanup = llvm::make_scope_exit([this, sccId] { activePureHelperSCCs.erase(sccId); });

    SmallVector<PureHelperDefinition> definitions;
    definitions.reserve(pureHelperSCCs[sccId].size());
    for (func::FuncOp memberFunc : pureHelperSCCs[sccId]) {
      auto calleesIt = pureHelperCallees.find(memberFunc.getOperation());
      if (calleesIt != pureHelperCallees.end()) {
        for (func::FuncOp callee : calleesIt->second) {
          unsigned calleeSccId = pureHelperSCCId[callee.getOperation()];
          if (calleeSccId != sccId && failed(ensurePureHelperEmitted(callee))) {
            return failure();
          }
        }
      }
      auto definition = buildPureHelperDefinition(memberFunc);
      if (failed(definition)) {
        return failure();
      }
      definitions.push_back(std::move(*definition));
    }

    if (definitions.size() == 1) {
      const PureHelperDefinition &definition = definitions.front();
      os << "(define-fun-rec " << definition.symbol << " (";
      for (auto [index, argName] : llvm::enumerate(definition.argNames)) {
        if (index != 0) {
          os << ' ';
        }
        os << '(' << argName << ' ' << definition.argSorts[index] << ')';
      }
      os << ") " << definition.resultSort << ' ' << definition.bodyExpr << ")\n";
    } else {
      os << "(define-funs-rec (\n";
      for (size_t index = 0; index < pureHelperSCCs[sccId].size(); index++) {
        const PureHelperDefinition &definition = definitions[index];
        os << "  (" << definition.symbol << " (";
        for (auto [argIndex, argName] : llvm::enumerate(definition.argNames)) {
          if (argIndex != 0) {
            os << ' ';
          }
          os << '(' << argName << ' ' << definition.argSorts[argIndex] << ')';
        }
        os << ") " << definition.resultSort << ")\n";
      }
      os << ") (\n";
      for (const PureHelperDefinition &definition : definitions) {
        os << "  " << definition.bodyExpr << '\n';
      }
      os << "))\n";
    }

    emittedPureHelperSCCs.insert(sccId);
    for (func::FuncOp memberFunc : pureHelperSCCs[sccId]) {
      emittedPureHelpers.insert(memberFunc.getOperation());
    }
    return success();
  }

  LogicalResult ensurePureHelperEmitted(func::FuncOp func) {
    if (emittedPureHelpers.contains(func.getOperation())) {
      return success();
    }
    if (!func || func.empty()) {
      return func.emitError("smt-to-smtlib requires non-empty helper funcs");
    }
    if (func.getFunctionType().getNumResults() != 1) {
      return func.emitError("pure SMT helper definitions require exactly one return value");
    }
    if (failed(initializePureHelperSCCs())) {
      return failure();
    }
    auto sccIt = pureHelperSCCId.find(func.getOperation());
    if (sccIt == pureHelperSCCId.end()) {
      return func.emitError("smt-to-smtlib could not classify pure helper recursion");
    }

    unsigned sccId = sccIt->second;
    if (activePureHelperSCCs.contains(sccId)) {
      return success();
    }
    if (recursivePureHelperSCCs.contains(sccId)) {
      for (func::FuncOp callee : pureHelperCallees[func.getOperation()]) {
        unsigned calleeSccId = pureHelperSCCId[callee.getOperation()];
        if (calleeSccId != sccId && failed(ensurePureHelperEmitted(callee))) {
          return failure();
        }
      }
      return emitRecursivePureHelperSCC(sccId);
    }

    auto calleesIt = pureHelperCallees.find(func.getOperation());
    if (calleesIt != pureHelperCallees.end()) {
      for (func::FuncOp callee : calleesIt->second) {
        if (failed(ensurePureHelperEmitted(callee))) {
          return failure();
        }
      }
    }

    auto definition = buildPureHelperDefinition(func);
    if (failed(definition)) {
      return failure();
    }
    if (failed(emitPureHelperDefinition(*definition))) {
      return failure();
    }
    emittedPureHelpers.insert(func.getOperation());
    return success();
  }

  /// Expand a script-style helper directly into the caller's linearized script.
  ///
  /// This binds callee block arguments to already-rendered caller expressions,
  /// emits the helper body in source order, and then returns the rendered
  /// `func.return` operands back to the caller for subsequent substitution.
  FailureOr<SmallVector<std::string>>
  inlineHelper(func::FuncOp func, ArrayRef<std::string> argExprs) {
    if (!func || func.empty()) {
      return func.emitError("smt-to-smtlib requires non-empty helper funcs");
    }
    if (!activeInlineHelpers.insert(func.getOperation()).second) {
      return func.emitError(
          "recursive script helpers are not supported by smt-to-smtlib because SMT-LIB has no "
          "recursive command-sequence abstraction"
      );
    }
    Operation *funcOp = func.getOperation();
    auto cleanup = llvm::make_scope_exit([this, funcOp] { activeInlineHelpers.erase(funcOp); });

    EvalContext helperCtx;
    helperCtx.preserveSharing = helperIsPurelyExpressionBased(func);
    for (auto [arg, expr] : llvm::zip(func.getArguments(), argExprs)) {
      helperCtx.values[arg] = expr;
    }

    if (failed(emitBlock(func.getBody().front(), helperCtx))) {
      return failure();
    }

    auto returnOp = dyn_cast<func::ReturnOp>(func.getBody().front().getTerminator());
    if (!returnOp) {
      return func.emitError("helper func must terminate with func.return");
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

  LogicalResult
  verifyEmptyCheckRegion(llzk::smt::CheckOp checkOp, StringRef regionName, Region &region) {
    if (!llvm::hasSingleElement(region) || !region.front().without_terminator().empty()) {
      return checkOp.emitOpError()
             << "cannot lower smt.check with non-empty result regions because "
                "SMT-LIB scripts cannot branch on check-sat results; '"
             << regionName << "' must be empty";
    }
    return success();
  }

  LogicalResult emitCheck(llzk::smt::CheckOp checkOp, EvalContext &ctx) {
    (void)ctx;
    if (checkOp.getNumResults() != 0) {
      return checkOp.emitOpError(
          "cannot lower result-producing smt.check because SMT-LIB has no "
          "standard return channel from check-sat into later script terms"
      );
    }
    if (failed(verifyEmptyCheckRegion(checkOp, "sat", checkOp.getSatRegion())) ||
        failed(verifyEmptyCheckRegion(checkOp, "unknown", checkOp.getUnknownRegion())) ||
        failed(verifyEmptyCheckRegion(checkOp, "unsat", checkOp.getUnsatRegion()))) {
      return failure();
    }

    os << "(check-sat)\n";
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
      return func.emitError("smt-to-smtlib requires non-empty helper funcs");
    }

    EvalContext ctx;
    ctx.preserveSharing = helperIsPurelyExpressionBased(func);
    if (func.getNumArguments() != argExprs.size()) {
      return func.emitError("helper argument arity mismatch during SMTLIB export");
    }
    for (auto [arg, expr] : llvm::zip(func.getArguments(), argExprs)) {
      ctx.values[arg] = expr;
    }

    Block &block = func.getBody().front();
    for (Operation &op : block.without_terminator()) {
      if (isa<llzk::smt::SetInfoOp, llzk::smt::DeclareFunOp, llzk::smt::AssertOp, llzk::smt::PushOp,
              llzk::smt::PopOp, llzk::smt::CheckOp>(op)) {
        return op.emitError("script-style SMT ops cannot appear in pure helper definitions");
      }
      if (failed(emitOperation(&op, ctx))) {
        return failure();
      }
    }

    auto returnOp = dyn_cast<func::ReturnOp>(block.getTerminator());
    if (!returnOp) {
      return func.emitError("helper func must terminate with func.return");
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
      if (isa<llzk::smt::SetInfoOp, llzk::smt::DeclareFunOp, llzk::smt::AssertOp, llzk::smt::PushOp,
              llzk::smt::PopOp, llzk::smt::CheckOp>(op)) {
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
        .Case<llzk::smt::EqOp>([&](auto exprOp) {
      return buildSExpr("=", exprOp.getInputs(), ctx);
    })
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
  ModuleOp selectedRootModule;
  llvm::raw_ostream &os;
  unsigned nextTempId = 0;
  unsigned pushDepth = 0;
  DenseSet<Operation *> activeInlineHelpers;
  DenseSet<Operation *> helperModesInProgress;
  DenseMap<Operation *, HelperMode> helperModes;
  llvm::StringMap<unsigned> emittedSymbolCounts;
  llvm::StringSet<> emittedAssertions;
  DenseMap<Operation *, std::string> helperSymbols;
  DenseSet<Operation *> emittedPureHelpers;
  bool pureHelperSCCsInitialized = false;
  DenseMap<Operation *, SmallVector<func::FuncOp>> pureHelperCallees;
  DenseMap<Operation *, unsigned> pureHelperSCCId;
  SmallVector<SmallVector<func::FuncOp>> pureHelperSCCs;
  DenseSet<unsigned> recursivePureHelperSCCs;
  DenseSet<unsigned> activePureHelperSCCs;
  DenseSet<unsigned> emittedPureHelperSCCs;
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
