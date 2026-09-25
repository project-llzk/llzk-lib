//===-- SMTDialectToSMTLIB.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "smt/Target/SMTLIBEmitter.h"

#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/SMTInfo/IR/SMTInfoOps.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SMT/IR/SMTAttributes.h>
#include <mlir/Dialect/SMT/IR/SMTDialect.h>
#include <mlir/Dialect/SMT/IR/SMTOps.h>
#include <mlir/Dialect/SMT/IR/SMTTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/FileUtilities.h>

#include <llvm/ADT/APSInt.h>
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

using namespace mlir;
using namespace llzk;

namespace {

/// Return whether a bare SMT-LIB symbol spelling is reserved by the language.
static bool isReservedIdentifier(StringRef name) {
  return llvm::is_contained(
      ArrayRef<StringRef> {
          "!",
          "_",
          "as",
          "BINARY",
          "DECIMAL",
          "exists",
          "forall",
          "HEXADECIMAL",
          "let",
          "match",
          "NUMERAL",
          "par",
          "STRING",
          "true",
          "false",
      },
      name
  );
}

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
  if (llvm::isDigit(out.front()) || isReservedIdentifier(out)) {
    out.insert(out.begin(), '_');
  }
  return out;
}

/// Append an SMT dialect sort using SMT-LIB textual syntax.
static void printSortForType(llvm::raw_ostream &os, Type type) {
  TypeSwitch<Type>(type)
      .Case<mlir::smt::IntType>([&os](auto) { os << "Int"; })
      .Case<mlir::smt::BoolType>([&os](auto) { os << "Bool"; })
      .Case<mlir::smt::SMTFuncType>([&os](auto funcType) {
    os << "((";
    llvm::interleave(funcType.getDomainTypes(), [&os](Type domainType) {
      printSortForType(os, domainType);
    }, [&os] { os << ' '; });
    os << ") ";
    printSortForType(os, funcType.getRangeType());
    os << ')';
  })
      .Case<mlir::smt::BitVectorType>([&os](auto bvType) {
    os << "(_ BitVec " << bvType.getWidth() << ')';
  })
      .Case<mlir::smt::ArrayType>([&os](auto arrayType) {
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

/// Print an SMT-LIB boolean literal for a native boolean value.
static void printBoolLiteral(llvm::raw_ostream &os, bool value) {
  os << (value ? "true" : "false");
}

/// Render an SMT-LIB boolean literal for a native boolean value.
static std::string formatBoolLiteral(bool value) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  printBoolLiteral(os, value);
  return storage;
}

/// Print a signed integer literal using SMT-LIB term syntax.
static void printIntegerLiteral(llvm::raw_ostream &os, const llvm::APInt &value) {
  llvm::APSInt signedValue(value, /*isUnsigned=*/false);
  if (!signedValue.isNegative()) {
    SmallString<32> valueText;
    signedValue.toString(valueText, /*Radix=*/10);
    os << valueText;
    return;
  }

  llvm::APSInt magnitude(signedValue.abs(), /*isUnsigned=*/false);
  SmallString<32> valueText;
  magnitude.toString(valueText, /*Radix=*/10);
  os << "(- " << valueText << ')';
}

/// Render a signed integer literal using SMT-LIB term syntax.
static std::string formatIntegerLiteral(const llvm::APInt &value) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  printIntegerLiteral(os, value);
  return storage;
}

/// Print a structured SMT-LIB `set-info` value attribute.
static void printSetInfoValue(llvm::raw_ostream &os, Attribute value) {
  TypeSwitch<Attribute>(value)
      .Case<smt_info::KeywordAttr>([&os](auto keywordAttr) { os << keywordAttr.getValue(); })
      .Case<smt_info::SymbolAttr>([&os](auto symbolAttr) { os << symbolAttr.getValue(); })
      .Case<StringAttr>([&os](auto strAttr) { strAttr.print(os); })
      .Case<BoolAttr>([&os](auto boolAttr) { printBoolLiteral(os, boolAttr.getValue()); })
      .Case<IntegerAttr>([&os](auto intAttr) {
    printIntegerLiteral(os, intAttr.getValue());
  }).Case<ArrayAttr>([&os](auto arrayAttr) {
    os << '(';
    llvm::interleave(arrayAttr, [&os](Attribute element) {
      printSetInfoValue(os, element);
    }, [&os] { os << ' '; });
    os << ')';
  });
}

/// Lowering mode for a helper `func.func` referenced from the script root.
enum class HelperMode : std::uint8_t {
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
  /// One rendered SSA binding tracked while linearizing SMT dialect IR.
  ///
  /// `text` is the SMT-LIB fragment currently bound to an MLIR value. The
  /// `survivesReset` bit records whether that fragment remains valid after an
  /// SMT-LIB `(reset)`.
  ///
  /// This distinction is necessary because the exporter caches SSA bindings
  /// across the whole current script, while SMT-LIB reset clears solver state
  /// such as declarations and function definitions. A cached binding like `x`
  /// or `(helper x)` becomes invalid after reset unless the underlying symbol
  /// or helper is re-emitted, whereas self-contained terms like `(- 1)` or
  /// `(+ (- 1) 2)` remain valid and should not be discarded unnecessarily.
  struct ValueBinding {
    /// Rendered SMT-LIB term or symbol name for the bound value.
    std::string text;

    /// Whether `text` can still be referenced after an SMT-LIB `(reset)`.
    bool survivesReset = false;
  };

  /// Per-block evaluation state used while emitting or evaluating SMT ops.
  ///
  /// The exporter maps MLIR SSA values to their rendered SMT-LIB bindings,
  /// accumulates `let` bindings when preserving sharing for pure helper bodies,
  /// and carries the current sharing policy through nested evaluation.
  struct EvalContext {
    /// Current SSA-to-SMT-LIB binding environment.
    DenseMap<Value, ValueBinding> values;

    /// Pending `let` bindings emitted around a final expression in order.
    SmallVector<std::pair<std::string, std::string>> letBindings;

    /// Whether pure expression evaluation should preserve common subterms.
    bool preserveSharing = false;
  };

  /// Fully rendered data for one pure helper definition.
  ///
  /// Pure helpers are emitted as top-level `define-fun` or recursive
  /// `define-fun-rec`/`define-funs-rec` declarations, so the exporter stores
  /// the fully rendered symbol, signature, and body before printing them.
  struct PureHelperDefinition {
    /// Printed helper symbol name after SMT-LIB sanitization/uniquing.
    std::string symbol;

    /// Printed parameter names used in the emitted helper definition.
    SmallVector<std::string> argNames;

    /// SMT-LIB sorts for each helper parameter.
    SmallVector<std::string> argSorts;

    /// SMT-LIB sort of the helper result.
    std::string resultSort;

    /// Rendered SMT-LIB body term for the helper.
    std::string bodyExpr;
  };

  /// Select the effective module root used for solver discovery and helper lookup.
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

  /// Find the unique top-level solver root in the effective root module.
  FailureOr<mlir::smt::SolverOp> collectRoot() {
    selectedRootModule = getEffectiveRootModule();
    mlir::smt::SolverOp solver;
    for (Operation &op : selectedRootModule.getBody()->getOperations()) {
      if (auto solverOp = dyn_cast<mlir::smt::SolverOp>(op); solverOp && !solver) {
        solver = solverOp;
      } else if (solverOp) {
        return selectedRootModule.emitError(
            "smt-to-smtlib requires exactly one top-level smt.solver in the root module"
        );
      }
    }
    if (!solver) {
      return selectedRootModule.emitError(
          "smt-to-smtlib could not find a top-level smt.solver in the root "
          "module"
      );
    }
    return solver;
  }

  /// Return the module that owns helpers reachable from the selected root.
  ModuleOp getSelectedRootModule() {
    return selectedRootModule ? selectedRootModule : getEffectiveRootModule();
  }

  /// Dispatch emission once the root solver has been selected.
  LogicalResult emitRoot(mlir::smt::SolverOp solver, bool emitReset) {
    return emitSolverRoot(solver, emitReset);
  }

  /// Check whether the selected solver already sets the SMT-LIB logic.
  static bool solverHasExplicitSetLogic(mlir::smt::SolverOp solver) {
    return llvm::any_of(solver.getBodyRegion().front().without_terminator(), [](Operation &op) {
      return isa<mlir::smt::SetLogicOp>(op);
    });
  }

  /// Reset exporter-side state that tracks emitted script structure.
  void resetScriptState() {
    nextTempId = 0;
    pushDepth = 0;
    emittedSymbolCounts.clear();
    emittedAssertions.clear();
    helperSymbols.clear();
    emittedPureHelpers.clear();
    emittedPureHelperSCCs.clear();
  }

  /// Reserve one unique SMT-LIB symbol spelling, including generated suffixes.
  std::string reserveUniqueSymbol(std::string symbol) {
    while (emittedSymbolCounts.contains(symbol)) {
      unsigned &count = emittedSymbolCounts[symbol];
      if (count == 0) {
        count = 1;
      }
      std::string suffixedSymbol;
      llvm::raw_string_ostream exprStream(suffixedSymbol);
      exprStream << symbol << '_' << ++count;
      symbol = std::move(suffixedSymbol);
    }
    emittedSymbolCounts[symbol] = 1;
    return symbol;
  }

  /// Drop any cached SSA binding that depends on pre-reset solver state.
  static void pruneResetSensitiveBindings(EvalContext &ctx) {
    ctx.letBindings.clear();
    ctx.values.remove_if([](const auto &entry) { return !entry.second.survivesReset; });
  }

  /// Emit the script preamble and initialize per-script export state.
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

  /// Emit the selected solver as one complete SMT-LIB script body.
  LogicalResult emitSolverRoot(mlir::smt::SolverOp solver, bool emitReset) {
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

  /// Emit each non-terminator operation in a block in source order.
  LogicalResult emitBlock(Block &block, EvalContext &ctx) {
    for (Operation &op : block.without_terminator()) {
      if (failed(emitOperation(&op, ctx))) {
        return failure();
      }
    }
    return success();
  }

  /// Lower one top-level SMT dialect operation into SMT-LIB text or bindings.
  LogicalResult emitOperation(Operation *op, EvalContext &ctx) {
    auto bind = [&](auto exprOp) { return bindExpr(exprOp, ctx); };

    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<mlir::smt::SetLogicOp>([this](auto setLogicOp) {
      os << "(set-logic " << setLogicOp.getLogic() << ")\n";
      return success();
    })
        .Case<smt_info::SMTInfoSetOp>([this](auto setInfoOp) {
      os << "(set-info " << setInfoOp.getKey().getValue() << ' ';
      printSetInfoValue(os, setInfoOp.getValueAttr());
      os << ")\n";
      return success();
    })
        .Case<mlir::smt::DeclareFunOp>([&](auto declareOp) { return emitDeclare(declareOp, ctx); })
        .Case<mlir::smt::AssertOp>([&](auto assertOp) { return emitAssert(assertOp, ctx); })
        .Case<mlir::smt::ResetOp>([this, &ctx](auto) {
      os << "(reset)\n";
      resetScriptState();
      pruneResetSensitiveBindings(ctx);
      return success();
    })
        .Case<mlir::smt::PushOp>([this](auto pushOp) {
      pushDepth += pushOp.getCount();
      os << "(push " << pushOp.getCount() << ")\n";
      return success();
    })
        .Case<mlir::smt::PopOp>([this](auto popOp) {
      pushDepth -= popOp.getCount();
      os << "(pop " << popOp.getCount() << ")\n";
      return success();
    })
        .Case<mlir::smt::CheckOp>([&](auto checkOp) { return emitCheck(checkOp, ctx); })
        .Case<mlir::smt::SolverOp>([&](auto solverOp) {
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
        .Case<mlir::smt::BoolConstantOp>(bind)
        .Case<mlir::smt::IntConstantOp>(bind)
        .Case<mlir::smt::BVConstantOp>(bind)
        .Case<mlir::smt::EqOp>(bind)
        .Case<mlir::smt::NotOp>(bind)
        .Case<mlir::smt::AndOp>(bind)
        .Case<mlir::smt::OrOp>(bind)
        .Case<mlir::smt::XOrOp>(bind)
        .Case<mlir::smt::ImpliesOp>(bind)
        .Case<mlir::smt::IteOp>(bind)
        .Case<mlir::smt::IntAddOp>(bind)
        .Case<mlir::smt::IntMulOp>(bind)
        .Case<mlir::smt::IntSubOp>(bind)
        .Case<mlir::smt::IntDivOp>(bind)
        .Case<mlir::smt::IntModOp>(bind)
        .Case<mlir::smt::IntCmpOp>(bind)
        .Case<mlir::smt::Int2BVOp>(bind)
        .Case<mlir::smt::BV2IntOp>(bind)
        .Case<mlir::smt::DistinctOp>(bind)
        .Case<mlir::smt::IntAbsOp>(bind)
        .Case<mlir::smt::BVNegOp>(bind)
        .Case<mlir::smt::BVAndOp>(bind)
        .Case<mlir::smt::BVAddOp>(bind)
        .Case<mlir::smt::BVMulOp>(bind)
        .Case<mlir::smt::BVUDivOp>(bind)
        .Case<mlir::smt::BVSDivOp>(bind)
        .Case<mlir::smt::BVURemOp>(bind)
        .Case<mlir::smt::BVSRemOp>(bind)
        .Case<mlir::smt::BVSModOp>(bind)
        .Case<mlir::smt::BVOrOp>(bind)
        .Case<mlir::smt::BVXOrOp>(bind)
        .Case<mlir::smt::BVNotOp>(bind)
        .Case<mlir::smt::BVShlOp>(bind)
        .Case<mlir::smt::BVLShrOp>(bind)
        .Case<mlir::smt::BVAShrOp>(bind)
        .Case<mlir::smt::BVCmpOp>(bind)
        .Case<mlir::smt::ConcatOp>(bind)
        .Case<mlir::smt::ExtractOp>(bind)
        .Case<mlir::smt::RepeatOp>(bind)
        .Case<mlir::smt::ApplyFuncOp>(bind)
        .Case<mlir::smt::ArraySelectOp>(bind)
        .Case<mlir::smt::ArrayStoreOp>(bind)
        .Case<mlir::smt::ArrayBroadcastOp>(bind)
        .Case<mlir::smt::ForallOp>(bind)
        .Case<mlir::smt::ExistsOp>(bind)
        .Case<boolean::AssertOp>([&](auto assertOp) {
      return assertOp.emitError("boolean.assert is not serializable to SMT-LIB");
    }).Default([&](Operation *unknownOp) {
      return unknownOp->emitError("unsupported operation in smt-to-smtlib");
    });
  }

  /// Bind a single-result expression op to its rendered SMT-LIB expression.
  template <typename OpTy> LogicalResult bindExpr(OpTy op, EvalContext &ctx) {
    auto expr = buildExpr(op.getOperation(), ctx);
    if (failed(expr)) {
      return failure();
    }
    if (op->getNumResults() != 1) {
      return op.emitError("smt-to-smtlib only supports single-result expression ops");
    }
    if (ctx.preserveSharing && !expr->text.starts_with("(")) {
      ctx.values[op->getResult(0)] = std::move(*expr);
      return success();
    }
    if (ctx.preserveSharing) {
      std::string name = makeLetName();
      ctx.letBindings.emplace_back(name, std::move(expr->text));
      ctx.values[op->getResult(0)] = ValueBinding {std::move(name), expr->survivesReset};
      return success();
    }
    ctx.values[op->getResult(0)] = std::move(*expr);
    return success();
  }

  /// Emit a declared SMT symbol and record its printed SMT-LIB identifier.
  LogicalResult emitDeclare(mlir::smt::DeclareFunOp declareOp, EvalContext &ctx) {
    std::string symbol;
    if (auto prefix = declareOp.getNamePrefix()) {
      symbol = sanitizeSymbol(*prefix);
    } else {
      symbol = "tmp" + std::to_string(nextTempId++);
    }
    symbol = reserveUniqueSymbol(std::move(symbol));
    ctx.values[declareOp.getResult()] = ValueBinding {symbol, /*survivesReset=*/false};
    if (auto funcType = dyn_cast<mlir::smt::SMTFuncType>(declareOp.getType())) {
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

  /// Emit an assertion, deduplicating top-level assertions when possible.
  LogicalResult emitAssert(mlir::smt::AssertOp assertOp, EvalContext &ctx) {
    auto expr = lookup(assertOp.getInput(), ctx);
    if (failed(expr)) {
      return assertOp.emitError("missing SMTLIB expression for assertion input");
    }
    std::string rendered = wrapWithLets(expr->text, ctx.letBindings);
    ctx.letBindings.clear();
    if (pushDepth == 0 && !emittedAssertions.insert(rendered).second) {
      return success();
    }
    os << "(assert " << rendered << ")\n";
    return success();
  }

  /// Bind an `arith.constant` that already maps directly onto SMT-LIB syntax.
  LogicalResult emitArithConstant(arith::ConstantOp constOp, EvalContext &ctx) {
    if (constOp->getNumResults() != 1) {
      return constOp.emitOpError("smt-to-smtlib only supports single-result arith.constant");
    }
    if (auto boolAttr = dyn_cast<BoolAttr>(constOp.getValue())) {
      ctx.values[constOp.getResult()] =
          ValueBinding {formatBoolLiteral(boolAttr.getValue()), /*survivesReset=*/true};
      return success();
    }
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      ctx.values[constOp.getResult()] =
          ValueBinding {formatIntegerLiteral(intAttr.getValue()), /*survivesReset=*/true};
      return success();
    }
    return constOp.emitOpError("unsupported arith.constant for smt-to-smtlib");
  }

  /// Thread one-to-one unrealized casts through the current binding environment.
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

  /// Lower a helper call as either a pure function application or an inline script.
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
      argExprs.push_back(std::move(arg->text));
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
      ctx.values[callOp.getResult(0)] =
          ValueBinding {buildHelperApplication(callee, argExprs), /*survivesReset=*/false};
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
      ctx.values[result] = std::move(expr);
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
    if (llvm::any_of(
            func.getArgumentTypes(), [](Type type) { return isa<mlir::smt::SMTFuncType>(type); }
        ) ||
        isa<mlir::smt::SMTFuncType>(func.getResultTypes().front())) {
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
    auto cleanup = llvm::scope_exit([this, funcOp] { helperModesInProgress.erase(funcOp); });

    for (Operation &op : func.getBody().front().without_terminator()) {
      if (isa<mlir::smt::SetLogicOp, smt_info::SMTInfoSetOp, mlir::smt::DeclareFunOp,
              mlir::smt::AssertOp, mlir::smt::ResetOp, mlir::smt::PushOp, mlir::smt::PopOp,
              mlir::smt::CheckOp, mlir::smt::SolverOp>(op)) {
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
    symbol = reserveUniqueSymbol(std::move(symbol));
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
    return expr;
  }

  /// Build SCC metadata for the pure-helper call graph.
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

  /// Collect the pure helper callees used when emitting recursive definitions.
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

  /// Render a pure helper into the data needed for one SMT-LIB function definition.
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
    definition.bodyExpr = std::move(results->front().text);
    return definition;
  }

  /// Emit one helper parameter list in SMT-LIB function-definition syntax.
  void emitHelperParameters(const PureHelperDefinition &definition) {
    llvm::interleave(
        llvm::zip_equal(definition.argNames, definition.argSorts), [this](auto argAndSort) {
      const auto &[argName, argSort] = argAndSort;
      os << '(' << argName << ' ' << argSort << ')';
    }, [this] { os << ' '; }
    );
  }

  /// Emit a non-recursive pure helper as a `define-fun`.
  LogicalResult emitPureHelperDefinition(const PureHelperDefinition &definition) {
    os << "(define-fun " << definition.symbol << " (";
    emitHelperParameters(definition);
    os << ") " << definition.resultSort << ' ' << definition.bodyExpr << ")\n";
    return success();
  }

  /// Emit one recursive SCC of pure helpers using SMT-LIB recursive definitions.
  LogicalResult emitRecursivePureHelperSCC(unsigned sccId) {
    if (emittedPureHelperSCCs.contains(sccId)) {
      return success();
    }
    [[maybe_unused]] bool inserted = activePureHelperSCCs.insert(sccId).second;
    assert(inserted && "recursive SCC should not be re-entered during emission");
    auto cleanup = llvm::scope_exit([this, sccId] { activePureHelperSCCs.erase(sccId); });

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
      emitHelperParameters(definition);
      os << ") " << definition.resultSort << ' ' << definition.bodyExpr << ")\n";
    } else {
      os << "(define-funs-rec (\n";
      for (size_t index = 0; index < pureHelperSCCs[sccId].size(); index++) {
        const PureHelperDefinition &definition = definitions[index];
        os << "  (" << definition.symbol << " (";
        emitHelperParameters(definition);
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

  /// Ensure a pure helper and its pure dependencies have been emitted.
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
  FailureOr<SmallVector<ValueBinding>>
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
    auto cleanup = llvm::scope_exit([this, funcOp] { activeInlineHelpers.erase(funcOp); });

    EvalContext helperCtx;
    helperCtx.preserveSharing = helperIsPurelyExpressionBased(func);
    for (auto [arg, expr] : llvm::zip(func.getArguments(), argExprs)) {
      helperCtx.values[arg] = ValueBinding {expr, /*survivesReset=*/false};
    }

    if (failed(emitBlock(func.getBody().front(), helperCtx))) {
      return failure();
    }

    auto returnOp = dyn_cast<func::ReturnOp>(func.getBody().front().getTerminator());
    if (!returnOp) {
      return func.emitError("helper func must terminate with func.return");
    }

    SmallVector<ValueBinding> results;
    results.reserve(returnOp.getNumOperands());
    for (Value operand : returnOp.getOperands()) {
      auto expr = lookup(operand, helperCtx);
      if (failed(expr)) {
        return failure();
      }
      auto binding = *expr;
      binding.text = wrapWithLets(binding.text, helperCtx.letBindings);
      results.push_back(std::move(binding));
    }
    return results;
  }

  /// Require that an `smt.check` result region is structurally empty.
  LogicalResult
  verifyEmptyCheckRegion(mlir::smt::CheckOp checkOp, StringRef regionName, Region &region) {
    if (!llvm::hasSingleElement(region) || !region.front().without_terminator().empty()) {
      return checkOp.emitOpError()
             << "cannot lower smt.check with non-empty result regions because "
                "SMT-LIB scripts cannot branch on check-sat results; '"
             << regionName << "' must be empty";
    }
    return success();
  }

  /// Emit the restricted SMT-LIB-compatible form of `smt.check`.
  LogicalResult emitCheck(mlir::smt::CheckOp checkOp, EvalContext &) {
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

  /// Look up the currently rendered SMT-LIB expression bound to an SSA value.
  FailureOr<ValueBinding> lookup(Value value, EvalContext &ctx) {
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

  /// Evaluate a pure helper body into rendered result expressions.
  FailureOr<SmallVector<ValueBinding>>
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
      ctx.values[arg] = ValueBinding {expr, /*survivesReset=*/false};
    }

    Block &block = func.getBody().front();
    for (Operation &op : block.without_terminator()) {
      if (isa<smt_info::SMTInfoSetOp, mlir::smt::DeclareFunOp, mlir::smt::AssertOp,
              mlir::smt::PushOp, mlir::smt::PopOp, mlir::smt::CheckOp>(op)) {
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

    SmallVector<ValueBinding> results;
    results.reserve(returnOp.getNumOperands());
    for (Value operand : returnOp.getOperands()) {
      auto expr = lookup(operand, ctx);
      if (failed(expr)) {
        return failure();
      }
      auto binding = *expr;
      binding.text = wrapWithLets(binding.text, ctx.letBindings);
      results.push_back(std::move(binding));
    }
    return results;
  }

  /// Determine whether a helper body can preserve sharing with `let` bindings.
  bool helperIsPurelyExpressionBased(func::FuncOp func) const {
    for (Operation &op : func.getBody().front().without_terminator()) {
      if (isa<smt_info::SMTInfoSetOp, mlir::smt::DeclareFunOp, mlir::smt::AssertOp,
              mlir::smt::PushOp, mlir::smt::PopOp, mlir::smt::CheckOp>(op)) {
        return false;
      }
      if (auto callOp = dyn_cast<func::CallOp>(op); callOp && callOp.getNumResults() == 0) {
        return false;
      }
    }
    return true;
  }

  /// Render one expression-producing operation into an SMT-LIB term.
  FailureOr<ValueBinding> buildExpr(Operation *op, EvalContext &ctx) {
    return TypeSwitch<Operation *, FailureOr<ValueBinding>>(op)
        .Case<mlir::smt::BoolConstantOp>([](auto constOp) {
      return ValueBinding {formatBoolLiteral(constOp.getValue()), /*survivesReset=*/true};
    })
        .Case<mlir::smt::IntConstantOp>([](auto constOp) {
      return ValueBinding {formatIntegerLiteral(constOp.getValue()), /*survivesReset=*/true};
    })
        .Case<mlir::smt::BVConstantOp>([](auto constOp) {
      return ValueBinding {constOp.getValue().getValueAsString(), /*survivesReset=*/true};
    })
        .Case<mlir::smt::EqOp>([&](auto exprOp) {
      return buildSExpr("=", exprOp.getInputs(), ctx);
    })
        .Case<mlir::smt::DistinctOp>([&](auto exprOp) {
      return buildSExpr("distinct", exprOp.getInputs(), ctx);
    })
        .Case<mlir::smt::NotOp>([&](auto exprOp) {
      return buildSExpr("not", ValueRange {exprOp.getInput()}, ctx);
    })
        .Case<mlir::smt::AndOp>([&](auto exprOp) {
      return buildSExpr("and", exprOp.getInputs(), ctx);
    })
        .Case<mlir::smt::OrOp>([&](auto exprOp) {
      return buildSExpr("or", exprOp.getInputs(), ctx);
    })
        .Case<mlir::smt::XOrOp>([&](auto exprOp) {
      return buildSExpr("xor", exprOp.getInputs(), ctx);
    })
        .Case<mlir::smt::ImpliesOp>([&](auto exprOp) {
      return buildSExpr("=>", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<mlir::smt::IteOp>([&](auto exprOp) {
      return buildSExpr(
          "ite", ValueRange {exprOp.getCond(), exprOp.getThenValue(), exprOp.getElseValue()}, ctx
      );
    })
        .Case<mlir::smt::IntAbsOp>([&](auto exprOp) {
      return buildSExpr("abs", ValueRange {exprOp.getInput()}, ctx);
    })
        .Case<mlir::smt::IntAddOp>([&](auto exprOp) {
      return buildSExpr("+", exprOp.getInputs(), ctx);
    })
        .Case<mlir::smt::IntMulOp>([&](auto exprOp) {
      return buildSExpr("*", exprOp.getInputs(), ctx);
    })
        .Case<mlir::smt::IntSubOp>([&](auto exprOp) {
      return buildSExpr("-", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<mlir::smt::IntDivOp>([&](auto exprOp) {
      return buildSExpr("div", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<mlir::smt::IntModOp>([&](auto exprOp) {
      return buildSExpr("mod", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<mlir::smt::IntCmpOp>([&](auto cmpOp) { return buildCmpExpr(cmpOp, ctx); })
        .Case<mlir::smt::Int2BVOp>([&](auto exprOp) { return buildInt2BVExpr(exprOp, ctx); })
        .Case<mlir::smt::BV2IntOp>([&](auto exprOp) { return buildBV2IntExpr(exprOp, ctx); })
        .Case<mlir::smt::BVNegOp>([&](auto exprOp) {
      return buildSExpr("bvneg", ValueRange {exprOp.getInput()}, ctx);
    })
        .Case<mlir::smt::BVAndOp>([&](auto exprOp) {
      return buildSExpr("bvand", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<mlir::smt::BVAddOp>([&](auto exprOp) {
      return buildSExpr("bvadd", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<mlir::smt::BVMulOp>([&](auto exprOp) {
      return buildSExpr("bvmul", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<mlir::smt::BVUDivOp>([&](auto exprOp) {
      return buildSExpr("bvudiv", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<mlir::smt::BVSDivOp>([&](auto exprOp) {
      return buildSExpr("bvsdiv", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<mlir::smt::BVURemOp>([&](auto exprOp) {
      return buildSExpr("bvurem", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<mlir::smt::BVSRemOp>([&](auto exprOp) {
      return buildSExpr("bvsrem", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<mlir::smt::BVSModOp>([&](auto exprOp) {
      return buildSExpr("bvsmod", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<mlir::smt::BVOrOp>([&](auto exprOp) {
      return buildSExpr("bvor", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<mlir::smt::BVXOrOp>([&](auto exprOp) {
      return buildSExpr("bvxor", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<mlir::smt::BVNotOp>([&](auto exprOp) {
      return buildSExpr("bvnot", ValueRange {exprOp.getInput()}, ctx);
    })
        .Case<mlir::smt::BVShlOp>([&](auto exprOp) {
      return buildSExpr("bvshl", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<mlir::smt::BVLShrOp>([&](auto exprOp) {
      return buildSExpr("bvlshr", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<mlir::smt::BVAShrOp>([&](auto exprOp) {
      return buildSExpr("bvashr", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<mlir::smt::BVCmpOp>([&](auto exprOp) { return buildBVCmpExpr(exprOp, ctx); })
        .Case<mlir::smt::ConcatOp>([&](auto exprOp) {
      return buildSExpr("concat", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
    })
        .Case<mlir::smt::ExtractOp>([&](auto exprOp) { return buildExtractExpr(exprOp, ctx); })
        .Case<mlir::smt::RepeatOp>([&](auto exprOp) { return buildRepeatExpr(exprOp, ctx); })
        .Case<mlir::smt::ArraySelectOp>([&](auto exprOp) {
      return buildSExpr("select", ValueRange {exprOp.getArray(), exprOp.getIndex()}, ctx);
    })
        .Case<mlir::smt::ArrayStoreOp>([&](auto exprOp) {
      return buildSExpr(
          "store", ValueRange {exprOp.getArray(), exprOp.getIndex(), exprOp.getValue()}, ctx
      );
    })
        .Case<mlir::smt::ArrayBroadcastOp>([&](auto exprOp) -> FailureOr<ValueBinding> {
      auto valueExpr = lookup(exprOp.getValue(), ctx);
      if (failed(valueExpr)) {
        return failure();
      }
      std::string expr;
      llvm::raw_string_ostream exprStream(expr);
      exprStream << "((as const " << sortForType(exprOp.getType()) << ") " << valueExpr->text
                 << ')';
      return ValueBinding {
          std::move(expr),
          valueExpr->survivesReset,
      };
    })
        .Case<mlir::smt::ApplyFuncOp>([&](auto exprOp) -> FailureOr<ValueBinding> {
      auto funcExpr = lookup(exprOp.getFunc(), ctx);
      if (failed(funcExpr)) {
        return failure();
      }
      if (exprOp.getArgs().empty()) {
        return *funcExpr;
      }
      std::string expr;
      llvm::raw_string_ostream exprStream(expr);
      exprStream << '(' << funcExpr->text;
      bool survivesReset = funcExpr->survivesReset;
      for (Value arg : exprOp.getArgs()) {
        auto argExpr = lookup(arg, ctx);
        if (failed(argExpr)) {
          return failure();
        }
        survivesReset &= argExpr->survivesReset;
        exprStream << ' ' << argExpr->text;
      }
      exprStream << ')';
      return ValueBinding {std::move(expr), survivesReset};
    })
        .Case<mlir::smt::ForallOp>([&](auto exprOp) {
      return buildQuantifierExpr("forall", exprOp, ctx);
    })
        .Case<mlir::smt::ExistsOp>([&](auto exprOp) {
      return buildQuantifierExpr("exists", exprOp, ctx);
    }).Case<arith::ConstantOp>([&](auto constOp) {
      return buildArithConstantExpr(constOp);
    }).Default([&](Operation *unknownOp) {
      return unknownOp->emitError("unsupported expression op in smt-to-smtlib");
    });
  }

  /// Render an `arith.constant` as an SMT-LIB literal term.
  FailureOr<ValueBinding> buildArithConstantExpr(arith::ConstantOp constOp) {
    if (auto boolAttr = dyn_cast<BoolAttr>(constOp.getValue())) {
      return ValueBinding {formatBoolLiteral(boolAttr.getValue()), /*survivesReset=*/true};
    }
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      return ValueBinding {formatIntegerLiteral(intAttr.getValue()), /*survivesReset=*/true};
    }
    return constOp.emitOpError("unsupported arith.constant expression");
  }

  /// Render an integer comparison predicate using SMT-LIB comparison syntax.
  FailureOr<ValueBinding> buildCmpExpr(mlir::smt::IntCmpOp cmpOp, EvalContext &ctx) {
    StringRef pred;
    switch (cmpOp.getPred()) {
    case mlir::smt::IntPredicate::lt:
      pred = "<";
      break;
    case mlir::smt::IntPredicate::le:
      pred = "<=";
      break;
    case mlir::smt::IntPredicate::gt:
      pred = ">";
      break;
    case mlir::smt::IntPredicate::ge:
      pred = ">=";
      break;
    }
    return buildSExpr(pred, ValueRange {cmpOp.getLhs(), cmpOp.getRhs()}, ctx);
  }

  /// Render integer-to-bitvector conversion with an explicit target width.
  FailureOr<ValueBinding> buildInt2BVExpr(mlir::smt::Int2BVOp op, EvalContext &ctx) {
    auto input = lookup(op.getInput(), ctx);
    if (failed(input)) {
      return failure();
    }
    auto resultType = cast<mlir::smt::BitVectorType>(op.getResult().getType());
    std::string expr;
    llvm::raw_string_ostream exprStream(expr);
    exprStream << "((_ int_to_bv " << resultType.getWidth() << ") " << input->text << ')';
    return ValueBinding {
        std::move(expr),
        input->survivesReset,
    };
  }

  /// Render bitvector-to-integer conversion with the requested signedness.
  FailureOr<ValueBinding> buildBV2IntExpr(mlir::smt::BV2IntOp op, EvalContext &ctx) {
    auto input = lookup(op.getInput(), ctx);
    if (failed(input)) {
      return failure();
    }
    std::string expr;
    llvm::raw_string_ostream exprStream(expr);
    exprStream << '(' << (op.getIsSigned() ? "sbv_to_int" : "ubv_to_int") << ' ' << input->text
               << ')';
    return ValueBinding {
        std::move(expr),
        input->survivesReset,
    };
  }

  /// Render a bitvector comparison predicate using the matching SMT-LIB op.
  FailureOr<ValueBinding> buildBVCmpExpr(mlir::smt::BVCmpOp cmpOp, EvalContext &ctx) {
    static constexpr std::pair<mlir::smt::BVCmpPredicate, StringLiteral> predicateSpellings[] = {
        {mlir::smt::BVCmpPredicate::slt, "bvslt"}, {mlir::smt::BVCmpPredicate::sle, "bvsle"},
        {mlir::smt::BVCmpPredicate::sgt, "bvsgt"}, {mlir::smt::BVCmpPredicate::sge, "bvsge"},
        {mlir::smt::BVCmpPredicate::ult, "bvult"}, {mlir::smt::BVCmpPredicate::ule, "bvule"},
        {mlir::smt::BVCmpPredicate::ugt, "bvugt"}, {mlir::smt::BVCmpPredicate::uge, "bvuge"},
    };
    const auto *it = llvm::find_if(predicateSpellings, [pred = cmpOp.getPred()](const auto &entry) {
      return entry.first == pred;
    });
    assert(it != std::end(predicateSpellings) && "unhandled BVCmpPredicate");
    return buildSExpr(it->second, ValueRange {cmpOp.getLhs(), cmpOp.getRhs()}, ctx);
  }

  /// Render bitvector extraction with SMT-LIB's indexed `extract` operator.
  FailureOr<ValueBinding> buildExtractExpr(mlir::smt::ExtractOp op, EvalContext &ctx) {
    auto input = lookup(op.getInput(), ctx);
    if (failed(input)) {
      return failure();
    }
    unsigned lowBit = op.getLowBit();
    unsigned highBit = lowBit + cast<mlir::smt::BitVectorType>(op.getType()).getWidth() - 1;
    std::string expr;
    llvm::raw_string_ostream exprStream(expr);
    exprStream << "((_ extract " << highBit << ' ' << lowBit << ") " << input->text << ')';
    return ValueBinding {
        std::move(expr),
        input->survivesReset,
    };
  }

  /// Render bitvector repetition with SMT-LIB's indexed `repeat` operator.
  FailureOr<ValueBinding> buildRepeatExpr(mlir::smt::RepeatOp op, EvalContext &ctx) {
    auto input = lookup(op.getInput(), ctx);
    if (failed(input)) {
      return failure();
    }
    std::string expr;
    llvm::raw_string_ostream exprStream(expr);
    exprStream << "((_ repeat " << op.getCount() << ") " << input->text << ')';
    return ValueBinding {
        std::move(expr),
        input->survivesReset,
    };
  }

  /// Build a generic SMT-LIB s-expression from an operator name and operands.
  FailureOr<ValueBinding> buildSExpr(StringRef opName, ValueRange operands, EvalContext &ctx) {
    if (opName == "and") {
      SmallVector<std::string> renderedOperands;
      renderedOperands.reserve(operands.size());
      bool survivesReset = true;
      for (Value operand : operands) {
        auto value = lookup(operand, ctx);
        if (failed(value)) {
          return failure();
        }
        survivesReset &= value->survivesReset;
        if (value->text == "false") {
          return ValueBinding {std::string("false"), survivesReset};
        }
        if (value->text == "true") {
          continue;
        }
        renderedOperands.push_back(std::move(value->text));
      }
      if (renderedOperands.empty()) {
        return ValueBinding {std::string("true"), survivesReset};
      }
      if (renderedOperands.size() == 1) {
        return ValueBinding {renderedOperands.front(), survivesReset};
      }
      std::string expr;
      llvm::raw_string_ostream exprStream(expr);
      exprStream << "(and";
      for (const std::string &operand : renderedOperands) {
        exprStream << ' ' << operand;
      }
      exprStream << ')';
      return ValueBinding {std::move(expr), survivesReset};
    }

    std::string expr;
    llvm::raw_string_ostream exprStream(expr);
    exprStream << '(' << opName;
    bool survivesReset = true;
    for (Value operand : operands) {
      auto value = lookup(operand, ctx);
      if (failed(value)) {
        return failure();
      }
      survivesReset &= value->survivesReset;
      exprStream << ' ' << value->text;
    }
    exprStream << ')';
    return ValueBinding {std::move(expr), survivesReset};
  }

  /// Render a quantifier body that matches the exporter's structural restrictions.
  template <typename QuantifierOpTy>
  FailureOr<ValueBinding>
  buildQuantifierExpr(StringRef quantifierName, QuantifierOpTy op, EvalContext &ctx) {
    if (!op.getPatterns().empty()) {
      return op.emitError("smt-to-smtlib does not yet support quantified pattern emission");
    }
    if (!llvm::hasSingleElement(op.getBody())) {
      return op.emitError("smt-to-smtlib requires quantifier bodies with a single block");
    }

    EvalContext bodyCtx = ctx;
    bodyCtx.letBindings.clear();
    Block &body = op.getBody().front();
    std::string expr;
    llvm::raw_string_ostream exprStream(expr);
    exprStream << '(' << quantifierName << " (";
    auto namesAttr = op.getBoundVarNames();
    llvm::StringMap<unsigned> binderCounts;
    llvm::interleave(
        llvm::enumerate(body.getArguments()),
        [&binderCounts, &bodyCtx, &namesAttr, &exprStream, this](const auto &it) {
      auto [index, arg] = it;
      std::string name = namesAttr && index < namesAttr->size()
                             ? sanitizeSymbol(cast<StringAttr>((*namesAttr)[index]).getValue())
                             : "q" + std::to_string(nextTempId++);
      if (unsigned &count = binderCounts[name]; count++ != 0) {
        name += "_" + std::to_string(count);
      }
      bodyCtx.values[arg] = ValueBinding {name, /*survivesReset=*/true};
      exprStream << '(' << name << ' ' << sortForType(arg.getType()) << ')';
    }, [&exprStream] { exprStream << ' '; }
    );
    exprStream << ") ";

    for (Operation &nestedOp : body.without_terminator()) {
      if (isa<mlir::smt::SetLogicOp, smt_info::SMTInfoSetOp, mlir::smt::DeclareFunOp,
              mlir::smt::AssertOp, mlir::smt::ResetOp, mlir::smt::PushOp, mlir::smt::PopOp,
              mlir::smt::CheckOp, mlir::smt::SolverOp, func::CallOp, boolean::AssertOp>(nestedOp)) {
        return nestedOp.emitError(
            "smt-to-smtlib quantifier bodies may only contain expression ops because SMT-LIB "
            "terms cannot contain script commands or helper emissions"
        );
      }
      if (failed(emitOperation(&nestedOp, bodyCtx))) {
        return failure();
      }
    }

    auto yieldOp = dyn_cast<mlir::smt::YieldOp>(body.getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
      return op.emitError("smt-to-smtlib requires quantifier bodies to yield exactly one value");
    }
    auto yielded = lookup(yieldOp.getOperand(0), bodyCtx);
    if (failed(yielded)) {
      return failure();
    }
    exprStream << wrapWithLets(yielded->text, bodyCtx.letBindings) << ')';
    return ValueBinding {std::move(expr), yielded->survivesReset};
  }

  /// Create a fresh name for a local `let` binding.
  std::string makeLetName() { return "__let" + std::to_string(nextTempId++); }

  /// Wrap an expression with accumulated `let` bindings in dominance order.
  static std::string
  wrapWithLets(std::string expr, ArrayRef<std::pair<std::string, std::string>> letBindings) {
    std::string result;
    llvm::raw_string_ostream os(result);
    for (const auto &binding : letBindings) {
      os << "(let ((" << binding.first << ' ' << binding.second << ")) ";
    }
    os << expr;
    for (size_t i = 0; i < letBindings.size(); ++i) {
      os << ')';
    }
    return result;
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

} // namespace

/// Export the selected SMT solver rooted in `module` to SMT-LIB text.
LogicalResult llzk::smt::emitSMTLIBModule(ModuleOp module, llvm::raw_ostream &os) {
  return SMTLIBEmitter(module, os).emit();
}
