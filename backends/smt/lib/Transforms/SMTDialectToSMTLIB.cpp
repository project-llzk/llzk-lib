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

static bool isVerificationHelperRoot(func::FuncOp func) {
  StringRef name = func.getSymName();
  return name.starts_with("smt_verif_") &&
         (name.ends_with("_internal_entry") || name.ends_with("_compute_entry") ||
          name.ends_with("_constrain_entry"));
}

static bool isEntryPoint(func::FuncOp func) {
  StringRef name = func.getSymName();
  return name.starts_with("smt_verif_") && name.ends_with("_entry") &&
         !isVerificationHelperRoot(func);
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
  return TypeSwitch<Type, std::string>(type)
      .Case<llzk::smt::IntType>([](auto) { return "Int"; })
      .Case<llzk::smt::BoolType>([](auto) { return "Bool"; })
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

    bool needsReset = false;
    for (auto [index, root] : llvm::enumerate(roots)) {
      (void)index;
      if (isEntryPoint(root)) {
        if (failed(emitIsolatedEntryRoot(root, needsReset))) {
          return failure();
        }
        needsReset = true;
        continue;
      }
      if (failed(emitRoot(root, needsReset))) {
        return failure();
      }
      needsReset = true;
    }
    return success();
  }

private:
  enum class EntryStageMode { target, post };

  struct EvalContext {
    DenseMap<Value, std::string> values;
    std::optional<std::string> pendingStageLabel;
    SmallVector<std::pair<std::string, std::string>> letBindings;
    bool preserveSharing = false;
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
    module.emitError("smt-to-smtlib requires --entry to select a root func.func");
    return failure();
  }

  LogicalResult emitRoot(func::FuncOp root, bool emitReset) {
    if (failed(emitRootPreamble(root, emitReset))) {
      return failure();
    }

    EvalContext ctx;
    if (failed(initializeRootArgs(root, ctx))) {
      return failure();
    }
    return emitBlock(root.getBody().front(), ctx);
  }

  LogicalResult emitIsolatedEntryRoot(func::FuncOp root, bool emitReset) {
    if (failed(emitEntryStage(root, EntryStageMode::target, emitReset))) {
      return failure();
    }
    return emitEntryStage(root, EntryStageMode::post, /*emitReset=*/true);
  }

  LogicalResult emitEntryStage(func::FuncOp root, EntryStageMode mode, bool emitReset) {
    if (failed(emitRootPreamble(root, emitReset))) {
      return failure();
    }

    if (!root || root.empty()) {
      root.emitError("smt-to-smtlib requires non-empty root func.func");
      return failure();
    }

    EvalContext ctx;
    if (failed(initializeRootArgs(root, ctx))) {
      return failure();
    }

    auto getStageFamily = [](StringRef stage) -> StringRef {
      if (stage == "pre") {
        return "pre";
      }
      if (stage == "target" || stage.starts_with("target[")) {
        return "target";
      }
      if (stage == "post" || stage.starts_with("post[")) {
        return "post";
      }
      return "";
    };

    bool emittedRequestedStage = false;
    bool skippingOuterTargetProof = false;
    Block &block = root.getBody().front();
    for (Operation &op : block.without_terminator()) {
      if (auto checkOp = dyn_cast<llzk::smt::CheckOp>(op)) {
        std::string stage = ctx.pendingStageLabel.value_or("");
        StringRef family = getStageFamily(stage);
        if (skippingOuterTargetProof) {
          if (failed(emitSuccessEffects(checkOp, ctx))) {
            return failure();
          }
          ctx.pendingStageLabel.reset();
          skippingOuterTargetProof = false;
          continue;
        }

        if (mode == EntryStageMode::target) {
          if (family == "target") {
            if (failed(emitCheck(checkOp, ctx))) {
              return failure();
            }
            emittedRequestedStage = true;
            continue;
          }
          if (emittedRequestedStage) {
            return success();
          }
        } else {
          if (!emittedRequestedStage && family == "target") {
            if (failed(emitSuccessEffects(checkOp, ctx))) {
              return failure();
            }
            ctx.pendingStageLabel.reset();
            continue;
          }
          if (family == "pre" || family == "target" || family == "post") {
            if (failed(emitCheck(checkOp, ctx))) {
              return failure();
            }
            emittedRequestedStage = true;
            continue;
          }
        }
      }

      if (skippingOuterTargetProof) {
        continue;
      }

      if (failed(emitOperation(&op, ctx))) {
        return failure();
      }
      if (mode == EntryStageMode::target && emittedRequestedStage && ctx.pendingStageLabel &&
          getStageFamily(*ctx.pendingStageLabel) != "target") {
        return success();
      }
      if (mode == EntryStageMode::post && isa<func::CallOp>(op) && ctx.pendingStageLabel &&
          getStageFamily(*ctx.pendingStageLabel) == "target" && !emittedRequestedStage) {
        skippingOuterTargetProof = true;
      }
    }

    if (emittedRequestedStage) {
      return success();
    }
    root.emitError("smt-to-smtlib could not find requested entry stage");
    return failure();
  }

  LogicalResult emitRootPreamble(func::FuncOp root, bool emitReset) {
    if (emitReset) {
      os << "\n(reset)\n";
    }
    emittedSymbolCounts.clear();
    emittedAssertions.clear();
    nextTargetStageIndex = 0;
    nextPostStageIndex = 0;
    os << "(set-logic " << options.logic << ")\n";
    os << "; root: " << root.getSymName() << "\n";
    return success();
  }

  LogicalResult initializeRootArgs(func::FuncOp root, EvalContext &ctx) {
    std::string rootPrefix = sanitizeSymbol(root.getSymName());
    for (auto [index, arg] : llvm::enumerate(root.getArguments())) {
      std::string name = rootPrefix + "__arg" + std::to_string(index);
      ctx.values[arg] = name;
      os << "(declare-const " << name << " " << sortForType(arg.getType()) << ")\n";
    }
    return success();
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
        .Case<llzk::smt::BVAndOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BVOrOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BVXOrOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BVNotOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BVShlOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
        .Case<llzk::smt::BVLShrOp>([&](auto exprOp) { return bindExpr(exprOp, ctx); })
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

    if (callOp.getNumResults() == 0) {
      return emitScriptHelper(callee, argExprs);
    }

    bool allowStatefulGeneratedHelper = callee.getSymName().starts_with("smt_");
    auto results = evalHelper(callee, argExprs, allowStatefulGeneratedHelper);
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

    os << "; check-sat";
    if (ctx.pendingStageLabel) {
      os << " stage=" << formatStageLabel(*ctx.pendingStageLabel);
    }
    os << " expect=" << successInfo->expectedOutcome << "\n";
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
  evalHelper(func::FuncOp func, ArrayRef<std::string> argExprs, bool allowStatefulOps) {
    if (!func || func.empty()) {
      func.emitError("smt-to-smtlib requires non-empty helper funcs");
      return failure();
    }

    if (!activePureHelpers.insert(func.getOperation()).second) {
      func.emitError("recursive pure helper calls are not supported by smt-to-smtlib");
      return failure();
    }
    auto cleanup = llvm::make_scope_exit([&] { activePureHelpers.erase(func.getOperation()); });

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
      if (!allowStatefulOps && isa<llzk::smt::AssertOp>(op)) {
        op.emitError("stateful SMT ops are not allowed in inlined helper functions");
        return failure();
      }
      if (isa<llzk::smt::PushOp, llzk::smt::PopOp, llzk::smt::CheckOp>(op)) {
        op.emitError("stack or solver SMT ops are not allowed in inlined helper functions");
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
    })
        .Case<llzk::smt::IntCmpOp>([&](auto cmpOp) { return buildCmpExpr(cmpOp, ctx); })
        .Case<llzk::smt::Int2BVOp>([&](auto exprOp) { return buildInt2BVExpr(exprOp, ctx); })
        .Case<llzk::smt::BV2IntOp>([&](auto exprOp) { return buildBV2IntExpr(exprOp, ctx); })
        .Case<llzk::smt::BVAndOp>([&](auto exprOp) {
      return buildSExpr("bvand", ValueRange {exprOp.getLhs(), exprOp.getRhs()}, ctx);
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
    if (op.getIsSigned()) {
      op.emitOpError("signed smt.bv2int is not supported by smt-to-smtlib");
      return failure();
    }
    auto input = lookup(op.getInput(), ctx);
    if (failed(input)) {
      return failure();
    }
    return "(ubv_to_int " + *input + ")";
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
        expr += " " + operand;
      }
      expr += ")";
      return expr;
    }

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
        expr += " ";
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
  const llzk::smt::SMTLIBExportOptions &options;
  unsigned nextTempId = 0;
  unsigned nextTargetStageIndex = 0;
  unsigned nextPostStageIndex = 0;
  unsigned pushDepth = 0;
  DenseSet<Operation *> activePureHelpers;
  llvm::StringMap<unsigned> emittedSymbolCounts;
  llvm::StringSet<> emittedAssertions;
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
