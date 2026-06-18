//===-- PrettyPrintSMTLIB.cpp -----------------------------------*- C++ -*-===//
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
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/raw_ostream.h>

#include <optional>
#include <string>
#include <utility>

namespace llzk::smt {
#define GEN_PASS_DEF_SMTDIALECTTOSMTLIBPASS
#include "smt/Transforms/SMTPasses.h.inc"
} // namespace llzk::smt

using namespace mlir;

namespace {

static bool isEntryPoint(func::FuncOp func) {
  return func.getSymName().starts_with("smt_verif_") && func.getSymName().ends_with("_entry");
}

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

static std::string sortForType(Type type) {
  return TypeSwitch<Type, std::string>(type).Case<llzk::smt::IntType>([](auto) { return "Int"; }
  ).Case<llzk::smt::BoolType>([](auto) {
    return "Bool";
  }).Default([&](Type) -> std::string {
    llvm::report_fatal_error("unsupported SMTLIB sort in smt-to-smtlib");
  });
}

struct CheckSuccessInfo {
  StringRef expectedOutcome;
  Region *successRegion;
};

class SMTLIBEmitter {
public:
  SMTLIBEmitter(
      ModuleOp module, llvm::raw_ostream &os, const llzk::smt::SMTLIBExportOptions &options
  )
      : module(module), os(os), options(options) {}

  LogicalResult emit() {
    SmallVector<func::FuncOp> roots;
    if (failed(collectRoots(roots))) {
      return failure();
    }

    for (auto [index, root] : llvm::enumerate(roots)) {
      if (index != 0) {
        os << "\n(reset)\n";
      }
      os << "(set-logic " << options.logic << ")\n";
      os << "; root: " << root.getSymName() << "\n";
      if (failed(emitRoot(root))) {
        return failure();
      }
    }
    return success();
  }

private:
  struct EvalContext {
    DenseMap<Value, std::string> values;
    std::optional<std::string> pendingStageLabel;
  };

  LogicalResult collectRoots(SmallVectorImpl<func::FuncOp> &roots) {
    if (options.entry) {
      auto root = module.lookupSymbol<func::FuncOp>(*options.entry);
      if (!root) {
        module.emitError() << "smt-to-smtlib could not find entry symbol @" << *options.entry;
        return failure();
      }
      roots.push_back(root);
      return success();
    }

    module.walk([&](func::FuncOp func) {
      if (isEntryPoint(func)) {
        roots.push_back(func);
      }
    });

    if (roots.empty()) {
      module.emitError(
          "smt-to-smtlib found no smt_verif_*_entry roots; pass --entry to export "
          "a different func.func"
      );
      return failure();
    }
    return success();
  }

  LogicalResult emitRoot(func::FuncOp root) {
    if (!root || root.empty()) {
      root.emitError("smt-to-smtlib requires non-empty root func.func");
      return failure();
    }

    EvalContext ctx;
    std::string rootPrefix = sanitizeSymbol(root.getSymName());
    for (auto [index, arg] : llvm::enumerate(root.getArguments())) {
      std::string name = rootPrefix + "__arg" + std::to_string(index);
      ctx.values[arg] = name;
      os << "(declare-const " << name << " " << sortForType(arg.getType()) << ")\n";
    }

    return emitBlock(root.getBody().front(), ctx);
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
        .Case<llzk::smt::DeclareFunOp>([&](auto declareOp) { return emitDeclare(declareOp, ctx); })
        .Case<llzk::smt::AssertOp>([&](auto assertOp) { return emitAssert(assertOp, ctx); })
        .Case<llzk::smt::PushOp>([&](auto pushOp) {
      os << "(push " << pushOp.getCount() << ")\n";
      return success();
    })
        .Case<llzk::smt::PopOp>([&](auto popOp) {
      os << "(pop " << popOp.getCount() << ")\n";
      return success();
    })
        .Case<llzk::smt::CheckOp>([&](auto checkOp) { return emitCheck(checkOp, ctx); })
        .Case<func::CallOp>([&](auto callOp) { return emitCall(callOp, ctx); })
        .Case<UnrealizedConversionCastOp>([&](auto castOp) {
      return emitUnrealizedCast(castOp, ctx);
    })
        .Case<arith::ConstantOp>([&](auto constOp) { return emitArithConstant(constOp, ctx); })
        .Case<llzk::smt::BoolConstantOp>([&](auto constOp) { return bindExpr(constOp, ctx); })
        .Case<llzk::smt::IntConstantOp>([&](auto constOp) { return bindExpr(constOp, ctx); })
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
    ctx.values[declareOp.getResult()] = symbol;
    os << "(declare-fun " << symbol << " () " << sortForType(declareOp.getType()) << ")\n";
    return success();
  }

  LogicalResult emitAssert(llzk::smt::AssertOp assertOp, EvalContext &ctx) {
    auto expr = lookup(assertOp.getInput(), ctx);
    if (failed(expr)) {
      return assertOp.emitError("missing SMTLIB expression for assertion input");
    }
    os << "(assert " << *expr << ")\n";
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

    if (callOp.getNumResults() == 0) {
      return emitScriptHelper(callee, argExprs);
    }

    auto results = evalHelper(callee, argExprs);
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

    if (callOp.getNumResults() == 1) {
      std::optional<std::string> label = getStageLabel(callOp.getCallee());
      if (label) {
        ctx.pendingStageLabel = *label;
      }
    }
    return success();
  }

  LogicalResult emitScriptHelper(func::FuncOp func, ArrayRef<std::string> argExprs) {
    if (!func || func.empty()) {
      func.emitError("smt-to-smtlib requires non-empty helper funcs");
      return failure();
    }
    if (func.getFunctionType().getNumResults() != 0) {
      func.emitError("script helper inlining requires a void callee");
      return failure();
    }
    if (func.getNumArguments() != argExprs.size()) {
      func.emitError("helper argument arity mismatch during SMTLIB export");
      return failure();
    }

    EvalContext helperCtx;
    for (auto [arg, expr] : llvm::zip(func.getArguments(), argExprs)) {
      helperCtx.values[arg] = expr;
    }
    return emitBlock(func.getBody().front(), helperCtx);
  }

  std::optional<std::string> getStageLabel(StringRef callee) const {
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

    os << "; check-sat";
    if (ctx.pendingStageLabel) {
      os << " stage=" << *ctx.pendingStageLabel;
    }
    os << " expect=" << successInfo->expectedOutcome << "\n";
    os << "(check-sat)\n";
    ctx.pendingStageLabel.reset();
    return emitBlock(successInfo->successRegion->front(), ctx);
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
    if (func.getNumArguments() != argExprs.size()) {
      func.emitError("helper argument arity mismatch during SMTLIB export");
      return failure();
    }
    for (auto [arg, expr] : llvm::zip(func.getArguments(), argExprs)) {
      ctx.values[arg] = expr;
    }

    Block &block = func.getBody().front();
    for (Operation &op : block.without_terminator()) {
      if (isa<llzk::smt::AssertOp, llzk::smt::PushOp, llzk::smt::PopOp, llzk::smt::CheckOp>(op)) {
        op.emitError("stateful SMT ops are not allowed in inlined helper functions");
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
      results.push_back(std::move(*expr));
    }
    return results;
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
        .Case<llzk::smt::EqOp>([&](auto exprOp) { return buildSExpr("=", exprOp.getInputs(), ctx); }
        )
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
    }).Case<llzk::smt::IntCmpOp>([&](auto cmpOp) {
      return buildCmpExpr(cmpOp, ctx);
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

  FailureOr<std::string> buildSExpr(StringRef opName, ValueRange operands, EvalContext &ctx) {
    std::string expr = "(" + opName.str();
    for (Value operand : operands) {
      auto value = lookup(operand, ctx);
      if (failed(value)) {
        return failure();
      }
      expr += " " + *value;
    }
    expr += ")";
    return expr;
  }

  ModuleOp module;
  llvm::raw_ostream &os;
  const llzk::smt::SMTLIBExportOptions &options;
  unsigned nextTempId = 0;
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

    llzk::smt::SMTLIBExportOptions exportOptions;
    exportOptions.logic = logic;
    if (!entry.empty()) {
      exportOptions.entry = entry;
    }

    if (failed(llzk::smt::emitSMTLIBModule(getOperation(), *stream, exportOptions))) {
      signalPassFailure();
      return;
    }

    if (outputFile) {
      outputFile->keep();
    }
  }
};

} // namespace

LogicalResult llzk::smt::emitSMTLIBModule(
    ModuleOp module, llvm::raw_ostream &os, const llzk::smt::SMTLIBExportOptions &options
) {
  return SMTLIBEmitter(module, os, options).emit();
}
