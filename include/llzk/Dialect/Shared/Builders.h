//===-- Builders.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>

#include <memory>
#include <unordered_map>

namespace llzk {

inline mlir::Location getUnknownLoc(mlir::MLIRContext *context) {
  return mlir::UnknownLoc::get(context);
}

mlir::OwningOpRef<mlir::ModuleOp> createLLZKModule(mlir::MLIRContext *context, mlir::Location loc);

inline mlir::OwningOpRef<mlir::ModuleOp> createLLZKModule(mlir::MLIRContext *context) {
  return createLLZKModule(context, getUnknownLoc(context));
}

void addLangAttrForLLZKDialect(mlir::ModuleOp mod);

class BaseBuilder {
protected:
  mlir::MLIRContext *context;

public:
  BaseBuilder(mlir::MLIRContext *ctx) : context(ctx) {}

  inline mlir::Location getUnknownLoc() { return llzk::getUnknownLoc(context); }
};

template <typename Derived> class ModuleLikeBuilder : public BaseBuilder {
protected:
  // keyed on function name
  std::unordered_map<std::string_view, function::FuncDefOp> freeFuncMap;
  // keyed on struct name
  std::unordered_map<std::string_view, component::StructDefOp> structMap;
  // keyed on struct name
  std::unordered_map<std::string_view, function::FuncDefOp> computeFnMap;
  // keyed on struct name
  std::unordered_map<std::string_view, function::FuncDefOp> constrainFnMap;
  // keyed on struct name
  std::unordered_map<std::string_view, function::FuncDefOp> productFnMap;

  /// @brief Ensure that a global function with the given funcName has not been added,
  /// reporting a fatal error otherwise.
  /// @param funcName
  void ensureNoSuchFreeFunc(std::string_view funcName);

  /// @brief Ensure that a global function with the given funcName has been added,
  /// reporting a fatal error otherwise.
  /// @param funcName
  void ensureFreeFnExists(std::string_view funcName);

  /// @brief Ensure that a struct with the given structName has not been added,
  /// reporting a fatal error otherwise.
  /// @param structName
  void ensureNoSuchStruct(std::string_view structName);

  /// @brief Ensure that a struct with the given structName exists,
  /// reporting a fatal error otherwise.
  /// @param structName
  void ensureStructExists(std::string_view structName);

  /// @brief Ensure that the given struct does not have a compute function,
  /// reporting a fatal error otherwise.
  /// @param structName
  void ensureNoSuchComputeFn(std::string_view structName);

  /// @brief Ensure that the given struct has a compute function,
  /// reporting a fatal error otherwise.
  /// @param structName
  void ensureComputeFnExists(std::string_view structName);

  /// @brief Ensure that the given struct does not have a constrain function,
  /// reporting a fatal error otherwise.
  /// @param structName
  void ensureNoSuchConstrainFn(std::string_view structName);

  /// @brief Ensure that the given struct has a constrain function,
  /// reporting a fatal error otherwise.
  /// @param structName
  void ensureConstrainFnExists(std::string_view structName);

  /// @brief Ensure that the given struct does not have a product function,
  /// reporting a fatal error otherwise.
  /// @param structName
  void ensureNoSuchProductFn(std::string_view structName);

  /// @brief Ensure that the given struct has a product function,
  /// reporting a fatal error otherwise.
  /// @param structName
  void ensureProductFnExists(std::string_view structName);

public:
  ModuleLikeBuilder(mlir::MLIRContext *ctx) : BaseBuilder(ctx) {}

  /* Getter methods */

  inline mlir::Region &getBodyRegion() { return static_cast<Derived *>(this)->getBodyRegion(); }

  mlir::FailureOr<component::StructDefOp> getStruct(std::string_view structName) const {
    if (structMap.find(structName) != structMap.end()) {
      return structMap.at(structName);
    }
    return mlir::failure();
  }

  mlir::FailureOr<function::FuncDefOp> getComputeFn(std::string_view structName) const {
    if (computeFnMap.find(structName) != computeFnMap.end()) {
      return computeFnMap.at(structName);
    }
    return mlir::failure();
  }
  inline mlir::FailureOr<function::FuncDefOp> getComputeFn(component::StructDefOp op) const {
    return getComputeFn(op.getName());
  }

  mlir::FailureOr<function::FuncDefOp> getConstrainFn(std::string_view structName) const {
    if (constrainFnMap.find(structName) != constrainFnMap.end()) {
      return constrainFnMap.at(structName);
    }
    return mlir::failure();
  }
  inline mlir::FailureOr<function::FuncDefOp> getConstrainFn(component::StructDefOp op) const {
    return getConstrainFn(op.getName());
  }

  mlir::FailureOr<function::FuncDefOp> getProductFn(std::string_view structName) const {
    if (productFnMap.find(structName) != productFnMap.end()) {
      return productFnMap.at(structName);
    }
    return mlir::failure();
  }
  inline mlir::FailureOr<function::FuncDefOp> getProductFn(component::StructDefOp op) const {
    return getProductFn(op.getName());
  }

  mlir::FailureOr<function::FuncDefOp> getFreeFunc(std::string_view funcName) const {
    if (freeFuncMap.find(funcName) != freeFuncMap.end()) {
      return freeFuncMap.at(funcName);
    }
    return mlir::failure();
  }

  /* Builder methods */

  Derived &insertEmptyStruct(std::string_view structName, mlir::Location loc);
  inline Derived &insertEmptyStruct(std::string_view structName) {
    return insertEmptyStruct(structName, getUnknownLoc());
  }

  Derived &insertComputeOnlyStruct(
      std::string_view structName, mlir::Location structLoc, mlir::Location computeLoc
  ) {
    insertEmptyStruct(structName, structLoc);
    insertComputeFn(structName, computeLoc);
    return static_cast<Derived &>(*this);
  }

  Derived &insertComputeOnlyStruct(std::string_view structName) {
    auto unk = getUnknownLoc();
    return insertComputeOnlyStruct(structName, unk, unk);
  }

  Derived &insertConstrainOnlyStruct(
      std::string_view structName, mlir::Location structLoc, mlir::Location constrainLoc
  ) {
    insertEmptyStruct(structName, structLoc);
    insertConstrainFn(structName, constrainLoc);
    return static_cast<Derived &>(*this);
  }

  Derived &insertConstrainOnlyStruct(std::string_view structName) {
    auto unk = getUnknownLoc();
    return insertConstrainOnlyStruct(structName, unk, unk);
  }

  Derived &insertFullStruct(
      std::string_view structName, mlir::Location structLoc, mlir::Location computeLoc,
      mlir::Location constrainLoc
  ) {
    insertEmptyStruct(structName, structLoc);
    insertComputeFn(structName, computeLoc);
    insertConstrainFn(structName, constrainLoc);
    return static_cast<Derived &>(*this);
  }

  /// Inserts a struct with both compute and constrain functions.
  Derived &insertFullStruct(std::string_view structName) {
    auto unk = getUnknownLoc();
    return insertFullStruct(structName, unk, unk, unk);
  }

  Derived &insertProductStruct(
      std::string_view structName, mlir::Location structLoc, mlir::Location productLoc
  ) {
    insertEmptyStruct(structName, structLoc);
    insertProductFn(structName, productLoc);
    return static_cast<Derived &>(*this);
  }

  Derived &insertProductStruct(std::string_view structName) {
    auto unk = getUnknownLoc();
    return insertProductStruct(structName, unk, unk);
  }

  /**
   * compute returns the type of the struct that defines it.
   * Since this is for testing, we accept no arguments.
   */
  static function::FuncDefOp buildComputeFn(component::StructDefOp op, mlir::Location loc);
  Derived &insertComputeFn(component::StructDefOp op, mlir::Location loc);
  Derived &insertComputeFn(std::string_view structName, mlir::Location loc);
  inline Derived &insertComputeFn(std::string_view structName) {
    return insertComputeFn(structName, getUnknownLoc());
  }

  /**
   * constrain accepts the struct type as the first argument.
   */
  static function::FuncDefOp buildConstrainFn(component::StructDefOp op, mlir::Location loc);
  Derived &insertConstrainFn(component::StructDefOp op, mlir::Location loc);
  Derived &insertConstrainFn(std::string_view structName, mlir::Location loc);
  inline Derived &insertConstrainFn(std::string_view structName) {
    return insertConstrainFn(structName, getUnknownLoc());
  }

  /**
   * product returns the type of the struct that defines it.
   * Since this is for testing, we accept no arguments.
   */
  static function::FuncDefOp buildProductFn(component::StructDefOp op, mlir::Location loc);
  Derived &insertProductFn(component::StructDefOp op, mlir::Location loc);
  Derived &insertProductFn(std::string_view structName, mlir::Location loc);
  inline Derived &insertProductFn(std::string_view structName) {
    return insertProductFn(structName, getUnknownLoc());
  }

  /**
   * Only requirement for compute is the call itself.
   * It should also initialize the internal member, but we can ignore those
   * ops for the sake of testing.
   */
  Derived &insertComputeCall(
      component::StructDefOp caller, component::StructDefOp callee, mlir::Location callLoc
  );
  Derived &
  insertComputeCall(std::string_view caller, std::string_view callee, mlir::Location callLoc);
  Derived &insertComputeCall(std::string_view caller, std::string_view callee) {
    return insertComputeCall(caller, callee, getUnknownLoc());
  }

  /**
   * To call a constraint function, you must:
   * 1. Add the callee as an internal member of the caller,
   * 2. Read the callee in the caller's constraint function,
   * 3. Call the callee's constraint function.
   */
  Derived &insertConstrainCall(
      component::StructDefOp caller, component::StructDefOp callee, mlir::Location callLoc,
      mlir::Location memberDefLoc
  );
  Derived &insertConstrainCall(
      std::string_view caller, std::string_view callee, mlir::Location callLoc,
      mlir::Location memberDefLoc
  );
  Derived &insertConstrainCall(std::string_view caller, std::string_view callee) {
    return insertConstrainCall(caller, callee, getUnknownLoc(), getUnknownLoc());
  }

  Derived &insertFreeFunc(std::string_view funcName, ::mlir::FunctionType type, mlir::Location loc);
  inline Derived &insertFreeFunc(std::string_view funcName, ::mlir::FunctionType type) {
    return insertFreeFunc(funcName, type, getUnknownLoc());
  }

  Derived &
  insertFreeCall(function::FuncDefOp caller, std::string_view callee, mlir::Location callLoc);
  Derived &insertFreeCall(function::FuncDefOp caller, std::string_view callee) {
    return insertFreeCall(caller, callee, getUnknownLoc());
  }
};

/// @brief Builds out a LLZK-compliant template and provides utilities for populating
/// that template. This class is designed to be used by front-ends looking to
/// generate LLZK IR programmatically and is also a useful unit testing facility.
class TemplateBuilder : public ModuleLikeBuilder<TemplateBuilder> {
  polymorphic::TemplateOp myTemplate;

public:
  TemplateBuilder(polymorphic::TemplateOp t) : ModuleLikeBuilder(t.getContext()), myTemplate(t) {}

  /* Getter methods */

  mlir::Region &getBodyRegion() { return myTemplate.getBodyRegion(); }

  /// Get the associated template of this builder.
  polymorphic::TemplateOp &getTemplate() { return myTemplate; }

  mlir::FailureOr<polymorphic::TemplateParamOp> getParam(std::string_view name) {
    auto op = myTemplate.getConstNamed<polymorphic::TemplateParamOp>(name);
    if (op) {
      return op;
    }
    return mlir::failure();
  }

  mlir::FailureOr<polymorphic::TemplateExprOp> getExpr(std::string_view name) {
    auto op = myTemplate.getConstNamed<polymorphic::TemplateExprOp>(name);
    if (op) {
      return op;
    }
    return mlir::failure();
  }

  TemplateBuilder &
  insertParam(std::string_view name, mlir::Location loc, mlir::TypeAttr type = {}) {
    if (succeeded(getParam(name))) {
      llvm::report_fatal_error("Duplicate TemplateParamOp insertion attempted");
    }

    mlir::OpBuilder builder(context);

    auto &region = getBodyRegion();
    if (region.empty()) {
      region.emplaceBlock();
    }

    builder.setInsertionPointToEnd(&region.front());

    auto nameAttr = builder.getStringAttr(name);

    builder.create<polymorphic::TemplateParamOp>(loc, nameAttr, type);

    return *this;
  }

  inline TemplateBuilder &insertParam(std::string_view name) {
    return insertParam(name, getUnknownLoc());
  }

  TemplateBuilder &insertExpr(std::string_view name, mlir::Location loc) {
    if (succeeded(getExpr(name))) {
      llvm::report_fatal_error("Duplicate TemplateExprOp insertion attempted");
    }

    mlir::OpBuilder builder(context);

    auto &region = getBodyRegion();
    if (region.empty()) {
      region.emplaceBlock();
    }

    builder.setInsertionPointToEnd(&region.front());

    auto nameAttr = builder.getStringAttr(name);

    builder.create<polymorphic::TemplateExprOp>(loc, nameAttr);

    return *this;
  }

  inline TemplateBuilder &insertExpr(std::string_view name) {
    return insertExpr(name, getUnknownLoc());
  }
};

/// @brief Builds out a LLZK-compliant module and provides utilities for populating
/// that module. This class is designed to be used by front-ends looking to
/// generate LLZK IR programmatically and is also a useful unit testing facility.
class ModuleBuilder : public ModuleLikeBuilder<ModuleBuilder> {
  mlir::ModuleOp myModule;

  // keyed on template name
  std::unordered_map<std::string_view, std::unique_ptr<TemplateBuilder>> templateMap;
  // keyed on nested module name
  std::unordered_map<std::string_view, std::unique_ptr<ModuleBuilder>> nestedModuleMap;

  /// @brief Ensure that a template with the given templateName has not been added,
  /// reporting a fatal error otherwise.
  /// @param templateName
  void ensureNoSuchTemplate(std::string_view templateName);

  /// @brief Ensure that a template with the given templateName has been added,
  /// reporting a fatal error otherwise.
  /// @param templateName
  void ensureTemplateExists(std::string_view templateName);

  /// @brief Ensure that a nested module with the given moduleName has not been added,
  /// reporting a fatal error otherwise.
  /// @param moduleName
  void ensureNoSuchNestedModule(std::string_view moduleName);

  /// @brief Ensure that a nested module with the given moduleName has been added,
  /// reporting a fatal error otherwise.
  /// @param moduleName
  void ensureNestedModuleExists(std::string_view moduleName);

public:
  ModuleBuilder(mlir::ModuleOp m) : ModuleLikeBuilder(m.getContext()), myModule(m) {}

  /* Getter methods */

  mlir::Region &getBodyRegion() { return myModule.getBodyRegion(); }

  /// Get the associated module of this builder.
  mlir::ModuleOp &getModule() { return myModule; }

  mlir::FailureOr<TemplateBuilder *> getTemplate(std::string_view templateName) const {
    auto it = templateMap.find(templateName);
    if (it != templateMap.end()) {
      return it->second.get();
    }
    return mlir::failure();
  }

  mlir::FailureOr<ModuleBuilder *> getNestedModule(std::string_view moduleName) const {
    auto it = nestedModuleMap.find(moduleName);
    if (it != nestedModuleMap.end()) {
      return it->second.get();
    }
    return mlir::failure();
  }

  /* Builder methods */

  ModuleBuilder &
  insertTemplate(std::string_view templateName, mlir::Location loc, unsigned numParams = 0);
  inline ModuleBuilder &insertTemplate(std::string_view templateName, unsigned numParams = 0) {
    return insertTemplate(templateName, getUnknownLoc(), numParams);
  }

  ModuleBuilder &insertNestedModule(std::string_view moduleName, mlir::Location loc);
  inline ModuleBuilder &insertNestedModule(std::string_view moduleName) {
    return insertNestedModule(moduleName, getUnknownLoc());
  }
};

} // namespace llzk
