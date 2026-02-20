//===-- CAPITestBase.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/CAPI/Builder.h"
#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/Bool/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Shared/Builders.h"

#include "llzk-c/Builder.h"
#include "llzk-c/InitDialects.h"

#include <mlir/CAPI/Wrap.h>
#include <mlir/Dialect/Arith/IR/Arith.h>

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/IR.h>
#include <mlir-c/RegisterEverything.h>

#include <gtest/gtest.h>

class CAPITest : public ::testing::Test {
protected:
  CAPITest() : context(mlirContextCreate()) {
    auto registry = mlirDialectRegistryCreate();
    mlirRegisterAllDialects(registry);
    llzkRegisterAllDialects(registry);
    mlirContextAppendDialectRegistry(context, registry);
    mlirContextLoadAllAvailableDialects(context);
    mlirDialectRegistryDestroy(registry);
  }

  ~CAPITest() override { mlirContextDestroy(context); }

public:
  MlirContext context;

  /// Helper to get IndexType
  inline MlirType createIndexType() const { return mlirIndexTypeGet(context); }

  /// Helper to create a simple IntegerAttr with IndexType
  inline MlirAttribute createIndexAttribute(int64_t value) const {
    return mlirIntegerAttrGet(createIndexType(), value);
  }

  /// Helper to create a simple IntegerAttr with IndexType and value 0
  /// Note: This no-parameter overload must exist because `llzk-tblgen` generates
  /// code that calls `createIndex[Type/Attribute/Operation]` without parameters.
  inline MlirAttribute createIndexAttribute() const { return createIndexAttribute(0); }

  // Helper to create a simple test operation: `arith.constant 0 : index`
  MlirOperation createIndexOperation() const {
    MlirType indexType = createIndexType();
    MlirOperationState op_state = mlirOperationStateGet(
        mlirStringRefCreateFromCString("arith.constant"), mlirLocationUnknownGet(context)
    );
    mlirOperationStateAddResults(&op_state, 1, &indexType);

    MlirNamedAttribute attr = mlirNamedAttributeGet(
        mlirIdentifierGet(context, mlirStringRefCreateFromCString("value")), createIndexAttribute()
    );
    mlirOperationStateAddAttributes(&op_state, 1, &attr);

    return mlirOperationCreate(&op_state);
  }

  /// Get FeltType, using C++ API to avoid indirectly testing other LLZK C API functions
  /// within the tests.
  static mlir::Type cppGetFeltType(MlirOpBuilder builder) {
    return unwrap(builder)->getType<llzk::felt::FeltType>();
  }

  /// Build a boolean constant value, using C++ API to avoid indirectly testing other
  ///  LLZK C API functions within the tests.
  static mlir::Value cppGenBoolConstant(MlirOpBuilder builder, MlirLocation location) {
    mlir::OpBuilder *cppBuilder = unwrap(builder);
    return cppBuilder->create<mlir::arith::ConstantOp>(
        unwrap(location), cppBuilder->getAttr<mlir::BoolAttr>(true)
    );
  }

  /// Build a felt constant value, using C++ API to avoid indirectly testing other
  ///  LLZK C API functions within the tests.
  static mlir::Value cppGenFeltConstant(MlirOpBuilder builder, MlirLocation location) {
    mlir::OpBuilder *cppBuilder = unwrap(builder);
    return cppBuilder->create<llzk::felt::FeltConstantOp>(
        unwrap(location), cppBuilder->getAttr<llzk::felt::FeltConstAttr>(llvm::APInt())
    );
  }

  /// Generate a new `ModuleOp` using C++ API to avoid indirectly testing other LLZK C API functions
  /// within the tests and set the insertion point of the builder to the body of the new `ModuleOp`.
  mlir::OwningOpRef<mlir::ModuleOp>
  cppNewModuleAndSetInsertionPoint(MlirOpBuilder builder, MlirLocation location) const {
    mlir::MLIRContext *cppCtx = unwrap(context);
    mlir::Location cppLoc = unwrap(location);
    mlir::OwningOpRef<mlir::ModuleOp> newModule = llzk::createLLZKModule(cppCtx, cppLoc);
    unwrap(builder)->setInsertionPointToStart(newModule->getBody());
    return newModule;
  }

  /// Build a struct, using C++ API to avoid indirectly testing other LLZK C API functions
  /// within the tests, and set the insertion point within the function body indicated by
  /// the FunctionKind parameter.
  mlir::OwningOpRef<mlir::ModuleOp> cppGenStructAndSetInsertionPoint(
      MlirOpBuilder builder, MlirLocation location, llzk::function::FunctionKind kind
  ) const {
    assert(kind != llzk::function::FunctionKind::Free && "supports only struct functions");
    auto newModule = this->cppNewModuleAndSetInsertionPoint(builder, location);
    llzk::ModuleBuilder cppBldr(newModule.get());
    auto name = "TestStruct";
    if (kind == llzk::function::FunctionKind::StructProduct) {
      cppBldr.insertProductStruct(name);
    } else {
      cppBldr.insertFullStruct(name);
    }
    llzk::function::FuncDefOp fDef = cppBldr.getFunc(kind, name).value();
    unwrap(builder)->setInsertionPointToStart(&fDef.getBody().emplaceBlock());
    return newModule;
  }
};

/// @brief
/// @tparam GTestBaseClass must be a subclass of `CAPITest`
template <typename GTestBaseClass> struct TestAnyBuildFuncHelper {
  virtual ~TestAnyBuildFuncHelper() = default;

  virtual bool callIsA(MlirOperation builtOp) = 0;
  virtual MlirOperation callBuild(const GTestBaseClass &, MlirOpBuilder, MlirLocation) = 0;
  virtual void doOtherChecks(MlirOperation builtOp) {}

  void run(const GTestBaseClass &testClass) {
    MlirOpBuilder builder = mlirOpBuilderCreate(testClass.context);
    MlirLocation location = mlirLocationUnknownGet(testClass.context);

    MlirOperation op = callBuild(testClass, builder, location);

    EXPECT_NE(op.ptr, (void *)NULL);
    EXPECT_TRUE(mlirOperationVerify(op));
    EXPECT_TRUE(callIsA(op));

    doOtherChecks(op);

    mlirOperationDestroy(op);
    mlirOpBuilderDestroy(builder);
  }
};
