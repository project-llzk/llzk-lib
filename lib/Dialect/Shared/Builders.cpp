//===-- Builders.cpp - Operation builder implementations --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/Shared/Builders.h"
#include "llzk/Util/SymbolHelper.h"

#include <llvm/Support/ErrorHandling.h>

namespace llzk {

using namespace mlir;
using namespace component;
using namespace function;
using namespace polymorphic;

OwningOpRef<ModuleOp> createLLZKModule(MLIRContext * /*context*/, Location loc) {
  auto mod = ModuleOp::create(loc);
  addLangAttrForLLZKDialect(mod);
  return mod;
}

void addLangAttrForLLZKDialect(ModuleOp mod) {
  MLIRContext *ctx = mod.getContext();
  if (auto *dialect = ctx->getOrLoadDialect<LLZKDialect>()) {
    mod->setAttr(LANG_ATTR_NAME, StringAttr::get(ctx, dialect->getNamespace()));
  } else {
    llvm::report_fatal_error("Could not load LLZK dialect!");
  }
}

/* ModuleBuilder */

void ModuleBuilder::ensureNoSuchFreeFunc(std::string_view funcName) {
  if (freeFuncMap.find(funcName) != freeFuncMap.end()) {
    llvm::report_fatal_error("global function " + Twine(funcName) + " already exists!");
  }
}

void ModuleBuilder::ensureFreeFnExists(std::string_view funcName) {
  if (freeFuncMap.find(funcName) == freeFuncMap.end()) {
    llvm::report_fatal_error("global function " + Twine(funcName) + " does not exist!");
  }
}

void ModuleBuilder::ensureNoSuchStruct(std::string_view structName) {
  if (structMap.find(structName) != structMap.end()) {
    llvm::report_fatal_error("struct " + Twine(structName) + " already exists!");
  }
}

void ModuleBuilder::ensureStructExists(std::string_view structName) {
  if (structMap.find(structName) == structMap.end()) {
    llvm::report_fatal_error("struct " + Twine(structName) + " does not exist!");
  }
}

void ModuleBuilder::ensureNoSuchComputeFn(std::string_view structName) {
  if (computeFnMap.find(structName) != computeFnMap.end()) {
    llvm::report_fatal_error("struct " + Twine(structName) + " already has a compute function!");
  }
}

void ModuleBuilder::ensureComputeFnExists(std::string_view structName) {
  if (computeFnMap.find(structName) == computeFnMap.end()) {
    llvm::report_fatal_error("struct " + Twine(structName) + " has no compute function!");
  }
}

void ModuleBuilder::ensureNoSuchConstrainFn(std::string_view structName) {
  if (constrainFnMap.find(structName) != constrainFnMap.end()) {
    llvm::report_fatal_error("struct " + Twine(structName) + " already has a constrain function!");
  }
}

void ModuleBuilder::ensureConstrainFnExists(std::string_view structName) {
  if (constrainFnMap.find(structName) == constrainFnMap.end()) {
    llvm::report_fatal_error("struct " + Twine(structName) + " has no constrain function!");
  }
}

void ModuleBuilder::ensureNoSuchProductFn(std::string_view structName) {
  if (productFnMap.find(structName) != productFnMap.end()) {
    llvm::report_fatal_error("struct " + Twine(structName) + " already has a product function!");
  }
}

void ModuleBuilder::ensureProductFnExists(std::string_view structName) {
  if (productFnMap.find(structName) == productFnMap.end()) {
    llvm::report_fatal_error("struct " + Twine(structName) + " has no product function!");
  }
}

ModuleBuilder &
ModuleBuilder::insertEmptyStruct(std::string_view structName, Location loc, int numStructParams) {
  ensureNoSuchStruct(structName);

  OpBuilder opBuilder(rootModule.getBody(), rootModule.getBody()->begin());
  if (numStructParams >= 0) { // Note: `==0` creates TemplateOp with no TemplateParamOp
    auto templateOp =
        opBuilder.create<TemplateOp>(loc, StringAttr::get(context, Twine("Template") + structName));
    opBuilder.setInsertionPointToStart(&templateOp.getBodyRegion().emplaceBlock());
    for (int i = 0; i < numStructParams; ++i) {
      opBuilder.create<TemplateParamOp>(
          loc, StringAttr::get(context, 'T' + std::to_string(i)), TypeAttr()
      );
    }
  }
  auto structDef = opBuilder.create<StructDefOp>(loc, StringAttr::get(context, structName));
  // populate the initial region
  auto &region = structDef.getRegion();
  (void)region.emplaceBlock();
  structMap[structName] = structDef;

  return *this;
}

FuncDefOp ModuleBuilder::buildComputeFn(StructDefOp op, Location loc) {
  MLIRContext *context = op.getContext();
  OpBuilder opBuilder(op.getBodyRegion());
  auto fnOp = opBuilder.create<FuncDefOp>(
      loc, StringAttr::get(context, FUNC_NAME_COMPUTE),
      FunctionType::get(context, {}, {op.getType()})
  );
  fnOp.setAllowWitnessAttr();
  fnOp.addEntryBlock();
  return fnOp;
}

ModuleBuilder &ModuleBuilder::insertComputeFn(StructDefOp op, Location loc) {
  ensureNoSuchComputeFn(op.getName());
  computeFnMap[op.getName()] = buildComputeFn(op, loc);
  return *this;
}

ModuleBuilder &ModuleBuilder::insertComputeFn(std::string_view structName, Location loc) {
  ensureStructExists(structName);
  return insertComputeFn(structMap.at(structName), loc);
}

FuncDefOp ModuleBuilder::buildConstrainFn(StructDefOp op, Location loc) {
  MLIRContext *context = op.getContext();
  OpBuilder opBuilder(op.getBodyRegion());
  auto fnOp = opBuilder.create<FuncDefOp>(
      loc, StringAttr::get(context, FUNC_NAME_CONSTRAIN),
      FunctionType::get(context, {op.getType()}, {})
  );
  fnOp.setAllowConstraintAttr();
  fnOp.addEntryBlock();
  return fnOp;
}

ModuleBuilder &ModuleBuilder::insertConstrainFn(StructDefOp op, Location loc) {
  ensureNoSuchConstrainFn(op.getName());
  constrainFnMap[op.getName()] = buildConstrainFn(op, loc);
  return *this;
}

ModuleBuilder &ModuleBuilder::insertConstrainFn(std::string_view structName, Location loc) {
  ensureStructExists(structName);
  return insertConstrainFn(structMap.at(structName), loc);
}

FuncDefOp ModuleBuilder::buildProductFn(StructDefOp op, Location loc) {
  MLIRContext *context = op.getContext();
  OpBuilder opBuilder(op.getBodyRegion());
  auto fnOp = opBuilder.create<FuncDefOp>(
      loc, StringAttr::get(context, FUNC_NAME_PRODUCT),
      FunctionType::get(context, {}, {op.getType()})
  );
  fnOp.setAllowWitnessAttr();
  fnOp.setAllowConstraintAttr();
  fnOp.addEntryBlock();
  return fnOp;
}

ModuleBuilder &ModuleBuilder::insertProductFn(StructDefOp op, Location loc) {
  ensureNoSuchProductFn(op.getName());
  productFnMap[op.getName()] = buildProductFn(op, loc);
  return *this;
}

ModuleBuilder &ModuleBuilder::insertProductFn(std::string_view structName, Location loc) {
  ensureStructExists(structName);
  return insertProductFn(structMap.at(structName), loc);
}

ModuleBuilder &
ModuleBuilder::insertComputeCall(StructDefOp caller, StructDefOp callee, Location callLoc) {
  ensureComputeFnExists(caller.getName());
  ensureComputeFnExists(callee.getName());

  auto callerFn = computeFnMap.at(caller.getName());
  auto calleeFn = computeFnMap.at(callee.getName());

  OpBuilder builder(callerFn.getBody());
  builder.create<CallOp>(callLoc, calleeFn);
  updateComputeReachability(caller, callee);
  return *this;
}

ModuleBuilder &ModuleBuilder::insertComputeCall(
    std::string_view caller, std::string_view callee, Location callLoc
) {
  ensureStructExists(caller);
  ensureStructExists(callee);
  return insertComputeCall(structMap.at(caller), structMap.at(callee), callLoc);
}

ModuleBuilder &ModuleBuilder::insertConstrainCall(
    StructDefOp caller, StructDefOp callee, Location callLoc, Location memberDefLoc
) {
  ensureConstrainFnExists(caller.getName());
  ensureConstrainFnExists(callee.getName());

  FuncDefOp callerFn = constrainFnMap.at(caller.getName());
  FuncDefOp calleeFn = constrainFnMap.at(callee.getName());
  StructType calleeTy = callee.getType();

  size_t numOps = caller.getBody()->getOperations().size();
  auto memberName = StringAttr::get(context, callee.getName().str() + std::to_string(numOps));

  // Insert the member declaration op
  {
    OpBuilder builder(caller.getBodyRegion());
    builder.create<MemberDefOp>(memberDefLoc, memberName, calleeTy);
  }

  // Insert the constrain function ops
  {
    OpBuilder builder(callerFn.getBody());

    auto member = builder.create<MemberReadOp>(
        callLoc, calleeTy, callerFn.getSelfValueFromConstrain(), memberName
    );
    builder.create<CallOp>(
        callLoc, TypeRange {}, calleeFn.getFullyQualifiedName(), ValueRange {member}
    );
  }
  updateConstrainReachability(caller, callee);
  return *this;
}

ModuleBuilder &ModuleBuilder::insertConstrainCall(
    std::string_view caller, std::string_view callee, Location callLoc, Location memberDefLoc
) {
  ensureStructExists(caller);
  ensureStructExists(callee);
  return insertConstrainCall(structMap.at(caller), structMap.at(callee), callLoc, memberDefLoc);
}

ModuleBuilder &
ModuleBuilder::insertFreeFunc(std::string_view funcName, FunctionType type, Location loc) {
  ensureNoSuchFreeFunc(funcName);

  OpBuilder opBuilder(rootModule.getBody(), rootModule.getBody()->begin());
  auto funcDef = opBuilder.create<FuncDefOp>(loc, funcName, type);
  (void)funcDef.addEntryBlock();
  freeFuncMap[funcName] = funcDef;

  return *this;
}

ModuleBuilder &
ModuleBuilder::insertFreeCall(FuncDefOp caller, std::string_view callee, Location callLoc) {
  ensureFreeFnExists(callee);
  FuncDefOp calleeFn = freeFuncMap.at(callee);

  OpBuilder builder(caller.getBody());
  builder.create<CallOp>(callLoc, calleeFn);
  return *this;
}

bool ModuleBuilder::computeReachable(std::string_view caller, std::string_view callee) {
  ensureStructExists(caller);
  ensureStructExists(callee);
  return computeReachable(structMap.at(caller), structMap.at(callee));
}

bool ModuleBuilder::constrainReachable(std::string_view caller, std::string_view callee) {
  ensureStructExists(caller);
  ensureStructExists(callee);
  return constrainReachable(structMap.at(caller), structMap.at(callee));
}

} // namespace llzk
