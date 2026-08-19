//===-- Modes.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "Modes.h"

#include "TypeConverter.h"

#include "llzk/Dialect/Array/IR/Dialect.h"
#include "llzk/Dialect/Bool/IR/Dialect.h"
#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/Global/IR/Dialect.h"
#include "llzk/Dialect/Include/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/POD/IR/Dialect.h"
#include "llzk/Dialect/RAM/IR/Dialect.h"
#include "llzk/Dialect/SMT/IR/SMTDialect.h"
#include "llzk/Dialect/String/IR/Dialect.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Dominance.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/StringSet.h>

using namespace mlir;
using namespace pcl::lowering;
using namespace llzk::component;
using namespace llzk::function;
using namespace llzk;

LogicalResult StubbedLoweringMode::runStep1() {
  if (failed(analyze())) {
    return failure();
  }
  checkAnalysisResult();

  // After the analysis, do the conversion as normal.
  ConversionTarget target(getContext());
  RewritePatternSet patterns(&getContext());
  PCLTypeConverter tc;
  populateStep1ConversionPatterns(tc, patterns);
  populateStep1ConversionTarget(target);

  return applyFullConversion(getOperation(), target, std::move(patterns));
}

bool BaseMode::isStep1LegalOp(Operation *op) {
  if (auto structDefOp = op->getParentOfType<StructDefOp>()) {
    auto funcDefOp = op->getParentOfType<FuncDefOp>();
    // Legal if either:
    //  - Not within a function definition.
    //  - The containing function definition is not the struct's constrain function.
    return !funcDefOp || structDefOp.getConstrainFuncOp() != funcDefOp;
  }

  auto funcDefOp = op->getParentOfType<FuncDefOp>();
  // Legal if:
  //  - Not within a free function definition.
  //  - The free function is not in the set of used free functions.
  return !funcDefOp || !usedFreeFunctions.contains(funcDefOp);
}

void BaseMode::populateStep1ConversionTarget(ConversionTarget &target) {
  target.addLegalDialect<pcl::PCLDialect, func::FuncDialect>();
  target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
  target.addDynamicallyLegalDialect<
      // clang-format off
      LLZKDialect,
      array::ArrayDialect, 
      boolean::BoolDialect, 
      component::StructDialect, 
      constrain::ConstrainDialect, 
      function::FunctionDialect, 
      global::GlobalDialect,
      include::IncludeDialect, 
      pod::PODDialect, 
      polymorphic::PolymorphicDialect, 
      ram::RAMDialect,
      scf::SCFDialect,
      smt::SMTDialect, 
      string::StringDialect, 
      verif::VerifDialect, 
      arith::ArithDialect,
      cast::CastDialect, 
      felt::FeltDialect
      // clang-format on
      >([this](Operation *op) { return isStep1LegalOp(op); });
  target.addLegalOp<FuncDefOp>();

  target.addDynamicallyLegalOp<NonDetOp>([this](NonDetOp op) {
    return isStep1LegalOp(op) && names.find(op) == names.end();
  });
}

void BaseMode::populateStep2ConversionTarget(ConversionTarget &target) {
  target.addLegalDialect<pcl::PCLDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::CallOp, func::ReturnOp>();
  target.addIllegalOp<StructDefOp, FuncDefOp>();
}

void BaseMode::populateStep3ConversionTarget(
    ConversionTarget &target, DupVarsReplacements &replacements
) {
  target.addLegalDialect<pcl::PCLDialect, func::FuncDialect>();
  target.addDynamicallyLegalOp<pcl::VarOp>([&replacements](pcl::VarOp op) {
    return replacements.find(op) == replacements.end();
  });
  target.addDynamicallyLegalOp<ModuleOp>([this](ModuleOp op) { return op == getOperation(); });
}

void BaseMode::collectNonDetOpNames() {
  uint64_t count = 0;
  StringSet<> usedNames;

  getOperation()->walk([&usedNames](MemberDefOp op) { usedNames.insert(op.getSymName()); });

  getOperation()->walk([&count, this, &usedNames](NonDetOp op) {
    if (!llvm::isa<FeltType>(op.getType())) {
      return;
    }
    StringRef nameRef;
    SmallString<32> nameStorage;
    do {
      nameStorage.clear();
      nameRef = ("_nondet_internal_var__" + Twine(count)).toStringRef(nameStorage);
      count++;
    } while (usedNames.contains(nameRef));
    names.insert({op, StringAttr::get(&getContext(), nameRef)});
  });
}

LogicalResult BaseMode::runStep2() {
  ConversionTarget target(getContext());
  RewritePatternSet patterns(&getContext());
  PCLTypeConverter tc;
  populateStep2ConversionPatterns(tc, patterns);
  populateStep2ConversionTarget(target);

  return applyFullConversion(getOperation(), target, std::move(patterns));
}

LogicalResult BaseMode::runStep3() {
  DupVarsReplacements replacements = collectDupVarsReplacements();

  ConversionTarget target(getContext());
  RewritePatternSet patterns(&getContext());
  populateStep3ConversionPatterns(patterns, replacements);
  populateStep3ConversionTarget(target, replacements);

  return applyFullConversion(getOperation(), target, std::move(patterns));
}

DupVarsReplacements BaseMode::collectDupVarsReplacements() {
  DupVarsReplacements replacements;
  getOperation()->walk([&replacements](func::FuncOp fn) {
    mlir::DominanceInfo dom(fn);
    llvm::StringMap<SmallVector<pcl::VarOp, 1>> varsByName;
    fn->walk([&varsByName](pcl::VarOp var) { varsByName[var.getName()].push_back(var); });

    for (auto &[_, vars] : varsByName) {
      if (vars.empty()) {
        continue;
      }
      std::stable_sort(vars.begin(), vars.end(), [&dom](pcl::VarOp lhs, pcl::VarOp rhs) {
        return dom.dominates(lhs.getOperation(), rhs);
      });
      auto fst = vars[0];
      for (auto other : ArrayRef(vars).drop_front()) {
        replacements[other] = fst;
      }
    }
  });
  return replacements;
}

BaseMode::BaseMode(ModuleOp op) : module(op), usedFreeFunctions(op) { collectNonDetOpNames(); }

LogicalResult FullLoweringMode::runStep1() {
  ConversionTarget target(getContext());
  RewritePatternSet patterns(&getContext());
  PCLTypeConverter tc;
  populateStep1ConversionPatterns(tc, patterns);
  populateStep1ConversionTarget(target);

  return applyFullConversion(getOperation(), target, std::move(patterns));
}

LogicalResult StubbedLoweringMode::analyze() {
  ConversionTarget target(getContext());
  RewritePatternSet patterns(&getContext());
  PCLTypeConverter tc;
  populateStep1ConversionPatterns(tc, patterns);
  populateStep1ConversionTarget(target);

  return applyAnalysisConversion(
      getOperation(), target, std::move(patterns), {.legalizableOps = &legalizableOps}
  );
}

void StubbedLoweringMode::checkAnalysisResult() {
  for (auto funcOp : getUsedFreeFunctions()) {
    funcOp->walk([&funcOp, this](Operation *op) {
      // If the op is not in the set of ops that would get converted,
      // mark the op as a stub and stop the search.
      if (!legalizableOps.contains(op)) {
        stubs.insert(funcOp);
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }

  // Remove all the stubs from the used functions set.
  for (auto stub : stubs) {
    getUsedFreeFunctions().erase(stub);
  }
}
