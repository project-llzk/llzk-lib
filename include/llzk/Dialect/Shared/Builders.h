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

#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>

#include <deque>
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

/// @brief Builds out a LLZK-compliant module and provides utilities for populating
/// that module. This class is designed to be used by front-ends looking to
/// generate LLZK IR programmatically and is also a useful unit testing facility.
/// TODO: this is a WIP, flesh this class out as needed.
class ModuleBuilder {
public:
  ModuleBuilder(mlir::ModuleOp m) : context(m.getContext()), rootModule(m) {}

  /* Builder methods */

  inline mlir::Location getUnknownLoc() { return llzk::getUnknownLoc(context); }

  ModuleBuilder &
  insertEmptyStruct(std::string_view structName, mlir::Location loc, int numStructParams = -1);
  inline ModuleBuilder &insertEmptyStruct(std::string_view structName, int numStructParams = -1) {
    return insertEmptyStruct(structName, getUnknownLoc(), numStructParams);
  }

  ModuleBuilder &insertComputeOnlyStruct(
      std::string_view structName, mlir::Location structLoc, mlir::Location computeLoc
  ) {
    insertEmptyStruct(structName, structLoc);
    insertComputeFn(structName, computeLoc);
    return *this;
  }

  ModuleBuilder &insertComputeOnlyStruct(std::string_view structName) {
    auto unk = getUnknownLoc();
    return insertComputeOnlyStruct(structName, unk, unk);
  }

  ModuleBuilder &insertConstrainOnlyStruct(
      std::string_view structName, mlir::Location structLoc, mlir::Location constrainLoc
  ) {
    insertEmptyStruct(structName, structLoc);
    insertConstrainFn(structName, constrainLoc);
    return *this;
  }

  ModuleBuilder &insertConstrainOnlyStruct(std::string_view structName) {
    auto unk = getUnknownLoc();
    return insertConstrainOnlyStruct(structName, unk, unk);
  }

  ModuleBuilder &insertFullStruct(
      std::string_view structName, mlir::Location structLoc, mlir::Location computeLoc,
      mlir::Location constrainLoc, int numStructParams = -1
  ) {
    insertEmptyStruct(structName, structLoc, numStructParams);
    insertComputeFn(structName, computeLoc);
    insertConstrainFn(structName, constrainLoc);
    return *this;
  }

  /// Inserts a struct with both compute and constrain functions.
  ModuleBuilder &insertFullStruct(std::string_view structName, int numStructParams = -1) {
    auto unk = getUnknownLoc();
    return insertFullStruct(structName, unk, unk, unk, numStructParams);
  }

  ModuleBuilder &insertProductStruct(
      std::string_view structName, mlir::Location structLoc, mlir::Location productLoc
  ) {
    insertEmptyStruct(structName, structLoc);
    insertProductFn(structName, productLoc);
    return *this;
  }

  ModuleBuilder &insertProductStruct(std::string_view structName) {
    auto unk = getUnknownLoc();
    return insertProductStruct(structName, unk, unk);
  }

  /**
   * compute returns the type of the struct that defines it.
   * Since this is for testing, we accept no arguments.
   */
  static function::FuncDefOp buildComputeFn(component::StructDefOp op, mlir::Location loc);
  ModuleBuilder &insertComputeFn(component::StructDefOp op, mlir::Location loc);
  inline ModuleBuilder &insertComputeFn(std::string_view structName, mlir::Location loc) {
    return insertComputeFn(*getStruct(structName), loc);
  }
  inline ModuleBuilder &insertComputeFn(std::string_view structName) {
    return insertComputeFn(structName, getUnknownLoc());
  }

  /**
   * constrain accepts the struct type as the first argument.
   */
  static function::FuncDefOp buildConstrainFn(component::StructDefOp op, mlir::Location loc);
  ModuleBuilder &insertConstrainFn(component::StructDefOp op, mlir::Location loc);
  inline ModuleBuilder &insertConstrainFn(std::string_view structName, mlir::Location loc) {
    return insertConstrainFn(*getStruct(structName), getUnknownLoc());
  }
  inline ModuleBuilder &insertConstrainFn(std::string_view structName) {
    return insertConstrainFn(structName, getUnknownLoc());
  }

  /**
   * product returns the type of the struct that defines it.
   * Since this is for testing, we accept no arguments.
   */
  static function::FuncDefOp buildProductFn(component::StructDefOp op, mlir::Location loc);
  ModuleBuilder &insertProductFn(component::StructDefOp op, mlir::Location loc);
  inline ModuleBuilder &insertProductFn(std::string_view structName, mlir::Location loc) {
    return insertProductFn(*getStruct(structName), loc);
  }
  inline ModuleBuilder &insertProductFn(std::string_view structName) {
    return insertProductFn(structName, getUnknownLoc());
  }

  /**
   * Only requirement for compute is the call itself.
   * It should also initialize the internal member, but we can ignore those
   * ops for the sake of testing.
   */
  ModuleBuilder &insertComputeCall(
      component::StructDefOp caller, component::StructDefOp callee, mlir::Location callLoc
  );
  ModuleBuilder &
  insertComputeCall(std::string_view caller, std::string_view callee, mlir::Location callLoc) {
    return insertComputeCall(*getStruct(caller), *getStruct(callee), callLoc);
  }
  ModuleBuilder &insertComputeCall(std::string_view caller, std::string_view callee) {
    return insertComputeCall(caller, callee, getUnknownLoc());
  }

  /**
   * To call a constraint function, you must:
   * 1. Add the callee as an internal member of the caller,
   * 2. Read the callee in the caller's constraint function,
   * 3. Call the callee's constraint function.
   */
  ModuleBuilder &insertConstrainCall(
      component::StructDefOp caller, component::StructDefOp callee, mlir::Location callLoc,
      mlir::Location memberDefLoc
  );
  ModuleBuilder &insertConstrainCall(
      std::string_view caller, std::string_view callee, mlir::Location callLoc,
      mlir::Location memberDefLoc
  ) {
    return insertConstrainCall(*getStruct(caller), *getStruct(callee), callLoc, memberDefLoc);
  }
  ModuleBuilder &insertConstrainCall(std::string_view caller, std::string_view callee) {
    return insertConstrainCall(caller, callee, getUnknownLoc(), getUnknownLoc());
  }

  ModuleBuilder &
  insertFreeFunc(std::string_view funcName, ::mlir::FunctionType type, mlir::Location loc);
  inline ModuleBuilder &insertFreeFunc(std::string_view funcName, ::mlir::FunctionType type) {
    return insertFreeFunc(funcName, type, getUnknownLoc());
  }

  ModuleBuilder &
  insertFreeCall(function::FuncDefOp caller, std::string_view callee, mlir::Location callLoc);
  ModuleBuilder &insertFreeCall(function::FuncDefOp caller, std::string_view callee) {
    return insertFreeCall(caller, callee, getUnknownLoc());
  }

  /* Getter methods */

  /// Get the top-level LLZK module.
  mlir::ModuleOp &getRootModule() { return rootModule; }

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

  inline mlir::FailureOr<function::FuncDefOp>
  getFunc(function::FunctionKind kind, std::string_view name) const {
    switch (kind) {
    case function::FunctionKind::StructCompute:
      return getComputeFn(name);
    case function::FunctionKind::StructConstrain:
      return getConstrainFn(name);
    case function::FunctionKind::StructProduct:
      return getProductFn(name);
    case function::FunctionKind::Free:
      return getFreeFunc(name);
    }
    return mlir::failure();
  }

  /* Helper functions */

  /**
   * Returns if the callee compute function is reachable by the caller by construction.
   */
  bool computeReachable(component::StructDefOp caller, component::StructDefOp callee) {
    return isReachable(computeNodes, caller, callee);
  }
  bool computeReachable(std::string_view caller, std::string_view callee) {
    return computeReachable(*getStruct(caller), *getStruct(callee));
  }

  /**
   * Returns if the callee compute function is reachable by the caller by construction.
   */
  bool constrainReachable(component::StructDefOp caller, component::StructDefOp callee) {
    return isReachable(constrainNodes, caller, callee);
  }
  bool constrainReachable(std::string_view caller, std::string_view callee) {
    return constrainReachable(*getStruct(caller), *getStruct(callee));
  }

private:
  mlir::MLIRContext *context;
  mlir::ModuleOp rootModule;

  struct CallNode {
    mlir::DenseMap<component::StructDefOp, CallNode *> callees;
  };

  using Def2NodeMap = mlir::DenseMap<component::StructDefOp, CallNode>;
  using StructDefSet = mlir::DenseSet<component::StructDefOp>;

  Def2NodeMap computeNodes, constrainNodes;

  std::unordered_map<std::string_view, function::FuncDefOp> freeFuncMap;
  std::unordered_map<std::string_view, component::StructDefOp> structMap;
  std::unordered_map<std::string_view, function::FuncDefOp> computeFnMap;
  std::unordered_map<std::string_view, function::FuncDefOp> constrainFnMap;
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

  void updateComputeReachability(component::StructDefOp caller, component::StructDefOp callee) {
    updateReachability(computeNodes, caller, callee);
  }

  void updateConstrainReachability(component::StructDefOp caller, component::StructDefOp callee) {
    updateReachability(constrainNodes, caller, callee);
  }

  void
  updateReachability(Def2NodeMap &m, component::StructDefOp caller, component::StructDefOp callee) {
    auto &callerNode = m[caller];
    auto &calleeNode = m[callee];
    callerNode.callees[callee] = &calleeNode;
  }

  bool isReachable(Def2NodeMap &m, component::StructDefOp caller, component::StructDefOp callee) {
    StructDefSet visited;
    std::deque<component::StructDefOp> frontier;
    frontier.push_back(caller);

    while (!frontier.empty()) {
      auto s = frontier.front();
      frontier.pop_front();
      if (!visited.insert(s).second) {
        continue;
      }

      if (s == callee) {
        return true;
      }
      for (auto &[calleeStruct, _] : m[s].callees) {
        frontier.push_back(calleeStruct);
      }
    }
    return false;
  }
};

} // namespace llzk
