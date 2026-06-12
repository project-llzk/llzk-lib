//===-- Builders.cpp - Operation builder implementations --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Shared/Builders.h"

#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include <mlir/IR/Builders.h>
#include <mlir/Support/LLVM.h>

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

/* ModuleLikeBuilder */

template <typename Derived>
void ModuleLikeBuilder<Derived>::ensureNoSuchFreeFunc(std::string_view funcName) {
  if (freeFuncMap.find(funcName) != freeFuncMap.end()) {
    llvm::report_fatal_error("global function " + Twine(funcName) + " already exists!");
  }
}

template <typename Derived>
void ModuleLikeBuilder<Derived>::ensureFreeFnExists(std::string_view funcName) {
  if (freeFuncMap.find(funcName) == freeFuncMap.end()) {
    llvm::report_fatal_error("global function " + Twine(funcName) + " does not exist!");
  }
}

template <typename Derived>
void ModuleLikeBuilder<Derived>::ensureNoSuchStruct(std::string_view structName) {
  if (structMap.find(structName) != structMap.end()) {
    llvm::report_fatal_error("struct " + Twine(structName) + " already exists!");
  }
}

template <typename Derived>
void ModuleLikeBuilder<Derived>::ensureStructExists(std::string_view structName) {
  if (structMap.find(structName) == structMap.end()) {
    llvm::report_fatal_error("struct " + Twine(structName) + " does not exist!");
  }
}

template <typename Derived>
void ModuleLikeBuilder<Derived>::ensureNoSuchComputeFn(std::string_view structName) {
  if (computeFnMap.find(structName) != computeFnMap.end()) {
    llvm::report_fatal_error("struct " + Twine(structName) + " already has a compute function!");
  }
}

template <typename Derived>
void ModuleLikeBuilder<Derived>::ensureComputeFnExists(std::string_view structName) {
  if (computeFnMap.find(structName) == computeFnMap.end()) {
    llvm::report_fatal_error("struct " + Twine(structName) + " has no compute function!");
  }
}

template <typename Derived>
void ModuleLikeBuilder<Derived>::ensureNoSuchConstrainFn(std::string_view structName) {
  if (constrainFnMap.find(structName) != constrainFnMap.end()) {
    llvm::report_fatal_error("struct " + Twine(structName) + " already has a constrain function!");
  }
}

template <typename Derived>
void ModuleLikeBuilder<Derived>::ensureConstrainFnExists(std::string_view structName) {
  if (constrainFnMap.find(structName) == constrainFnMap.end()) {
    llvm::report_fatal_error("struct " + Twine(structName) + " has no constrain function!");
  }
}

template <typename Derived>
void ModuleLikeBuilder<Derived>::ensureNoSuchProductFn(std::string_view structName) {
  if (productFnMap.find(structName) != productFnMap.end()) {
    llvm::report_fatal_error("struct " + Twine(structName) + " already has a product function!");
  }
}

template <typename Derived>
void ModuleLikeBuilder<Derived>::ensureProductFnExists(std::string_view structName) {
  if (productFnMap.find(structName) == productFnMap.end()) {
    llvm::report_fatal_error("struct " + Twine(structName) + " has no product function!");
  }
}

template <typename Derived>
Derived &ModuleLikeBuilder<Derived>::insertEmptyStruct(std::string_view structName, Location loc) {
  ensureNoSuchStruct(structName);

  OpBuilder opBuilder(this->getBodyRegion());
  auto structDef = opBuilder.create<StructDefOp>(loc, StringAttr::get(context, structName));
  // populate the initial region
  (void)structDef.getRegion().emplaceBlock();
  structMap[structName] = structDef;

  return static_cast<Derived &>(*this);
}

template <typename Derived>
FuncDefOp ModuleLikeBuilder<Derived>::buildComputeFn(StructDefOp op, Location loc) {
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

template <typename Derived>
Derived &ModuleLikeBuilder<Derived>::insertComputeFn(StructDefOp op, Location loc) {
  ensureNoSuchComputeFn(op.getName());
  computeFnMap[op.getName()] = buildComputeFn(op, loc);
  return static_cast<Derived &>(*this);
}

template <typename Derived>
Derived &ModuleLikeBuilder<Derived>::insertComputeFn(std::string_view structName, Location loc) {
  ensureStructExists(structName);
  return insertComputeFn(structMap.at(structName), loc);
}

template <typename Derived>
FuncDefOp ModuleLikeBuilder<Derived>::buildConstrainFn(StructDefOp op, Location loc) {
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

template <typename Derived>
Derived &ModuleLikeBuilder<Derived>::insertConstrainFn(StructDefOp op, Location loc) {
  ensureNoSuchConstrainFn(op.getName());
  constrainFnMap[op.getName()] = buildConstrainFn(op, loc);
  return static_cast<Derived &>(*this);
}

template <typename Derived>
Derived &ModuleLikeBuilder<Derived>::insertConstrainFn(std::string_view structName, Location loc) {
  ensureStructExists(structName);
  return insertConstrainFn(structMap.at(structName), loc);
}

template <typename Derived>
FuncDefOp ModuleLikeBuilder<Derived>::buildProductFn(StructDefOp op, Location loc) {
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

template <typename Derived>
Derived &ModuleLikeBuilder<Derived>::insertProductFn(StructDefOp op, Location loc) {
  ensureNoSuchProductFn(op.getName());
  productFnMap[op.getName()] = buildProductFn(op, loc);
  return static_cast<Derived &>(*this);
}

template <typename Derived>
Derived &ModuleLikeBuilder<Derived>::insertProductFn(std::string_view structName, Location loc) {
  ensureStructExists(structName);
  return insertProductFn(structMap.at(structName), loc);
}

template <typename Derived>
Derived &ModuleLikeBuilder<Derived>::insertComputeCall(
    StructDefOp caller, StructDefOp callee, Location callLoc
) {
  ensureComputeFnExists(caller.getName());
  ensureComputeFnExists(callee.getName());

  auto callerFn = computeFnMap.at(caller.getName());
  auto calleeFn = computeFnMap.at(callee.getName());

  OpBuilder builder(callerFn.getBody());
  builder.create<CallOp>(callLoc, calleeFn);
  return static_cast<Derived &>(*this);
}

template <typename Derived>
Derived &ModuleLikeBuilder<Derived>::insertComputeCall(
    std::string_view caller, std::string_view callee, Location callLoc
) {
  ensureStructExists(caller);
  ensureStructExists(callee);
  return insertComputeCall(structMap.at(caller), structMap.at(callee), callLoc);
}

template <typename Derived>
Derived &ModuleLikeBuilder<Derived>::insertConstrainCall(
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
  return static_cast<Derived &>(*this);
}

template <typename Derived>
Derived &ModuleLikeBuilder<Derived>::insertConstrainCall(
    std::string_view caller, std::string_view callee, Location callLoc, Location memberDefLoc
) {
  ensureStructExists(caller);
  ensureStructExists(callee);
  return insertConstrainCall(structMap.at(caller), structMap.at(callee), callLoc, memberDefLoc);
}

template <typename Derived>
Derived &ModuleLikeBuilder<Derived>::insertFreeFunc(
    std::string_view funcName, FunctionType type, Location loc, llvm::function_ref<void(OpBuilder&)> fnBody
) {
  ensureNoSuchFreeFunc(funcName);

  OpBuilder opBuilder(this->getBodyRegion());
  auto funcDef = opBuilder.create<FuncDefOp>(loc, funcName, type);
  auto *block = funcDef.addEntryBlock();
  if (fnBody) {
    OpBuilder::InsertionGuard guard(opBuilder);
    opBuilder.setInsertionPointToEnd(block);
    fnBody(opBuilder);
  }
  freeFuncMap[funcName] = funcDef;

  return static_cast<Derived &>(*this);
}

template <typename Derived>
Derived &ModuleLikeBuilder<Derived>::insertFreeCall(
    FuncDefOp caller, std::string_view callee, Location callLoc
) {
  ensureFreeFnExists(callee);
  FuncDefOp calleeFn = freeFuncMap.at(callee);

  OpBuilder builder(caller.getBody());
  builder.create<CallOp>(callLoc, calleeFn);
  return static_cast<Derived &>(*this);
}

/* ModuleBuilder */

void ModuleBuilder::ensureNoSuchTemplate(std::string_view templateName) {
  if (templateMap.find(templateName) != templateMap.end()) {
    llvm::report_fatal_error("template " + Twine(templateName) + " already exists!");
  }
}

void ModuleBuilder::ensureTemplateExists(std::string_view templateName) {
  if (templateMap.find(templateName) == templateMap.end()) {
    llvm::report_fatal_error("template " + Twine(templateName) + " does not exist!");
  }
}

ModuleBuilder &
ModuleBuilder::insertTemplate(std::string_view templateName, Location loc, unsigned numParams) {
  ensureNoSuchTemplate(templateName);

  OpBuilder opBuilder(myModule.getBodyRegion());
  auto templateDef = opBuilder.create<TemplateOp>(loc, StringAttr::get(context, templateName));
  opBuilder.setInsertionPointToStart(&templateDef.getBodyRegion().emplaceBlock());
  for (unsigned i = 0; i < numParams; ++i) {
    opBuilder.create<TemplateParamOp>(
        loc, StringAttr::get(context, 'T' + std::to_string(i)), TypeAttr()
    );
  }

  auto key = templateDef.getName();
  templateMap.emplace(key, std::make_unique<TemplateBuilder>(templateDef));

  return *this;
}

void ModuleBuilder::ensureNoSuchNestedModule(std::string_view moduleName) {
  if (nestedModuleMap.find(moduleName) != nestedModuleMap.end()) {
    llvm::report_fatal_error("nested module " + Twine(moduleName) + " already exists!");
  }
}

void ModuleBuilder::ensureNestedModuleExists(std::string_view moduleName) {
  if (nestedModuleMap.find(moduleName) == nestedModuleMap.end()) {
    llvm::report_fatal_error("nested module " + Twine(moduleName) + " does not exist!");
  }
}

ModuleBuilder &ModuleBuilder::insertNestedModule(std::string_view moduleName, Location loc) {
  ensureNoSuchNestedModule(moduleName);

  OpBuilder opBuilder(myModule.getBodyRegion());
  auto nestedMod = opBuilder.create<ModuleOp>(loc);
  nestedMod.setSymName(moduleName);

  auto key = *nestedMod.getSymName();
  nestedModuleMap.emplace(key, std::make_unique<ModuleBuilder>(nestedMod));

  return *this;
}

/* Explicit template instantiations */

template class ModuleLikeBuilder<ModuleBuilder>;
template class ModuleLikeBuilder<TemplateBuilder>;

} // namespace llzk
