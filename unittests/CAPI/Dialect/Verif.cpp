//===-- Verif.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Verif.h"

#include "../CAPITestBase.h"

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Shared/Builders.h"
#include "llzk/Dialect/Verif/IR/Ops.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/IR.h>

#include <mlir/CAPI/Wrap.h>

#include <llvm/ADT/SmallVector.h>

// Include the auto-generated tests
#include "llzk/Dialect/Verif/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Verif/IR/Ops.capi.test.cpp.inc"

namespace {

static MlirAttribute createFlatSymbolRefAttr(MlirContext ctx, const char *name) {
  return mlirFlatSymbolRefAttrGet(ctx, mlirStringRefCreateFromCString(name));
}

static MlirAttribute createStringAttr(MlirContext ctx, const char *value) {
  return mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(value));
}

static MlirAttribute createEmptyFunctionTypeAttr(MlirContext ctx) {
  MlirType type = mlirFunctionTypeGet(ctx, 0, nullptr, 0, nullptr);
  return mlirTypeAttrGet(type);
}

static mlir::OwningOpRef<mlir::ModuleOp> createModuleWithTargetFunc(
    const CAPITest &test, MlirOpBuilder builder, MlirLocation location, llvm::StringRef name
) {
  auto newModule = test.cppNewModuleAndSetInsertionPoint(builder, location);
  llzk::ModuleBuilder modBuilder(newModule.get());
  modBuilder.insertFreeFunc(
      name, mlir::FunctionType::get(unwrap(test.context), mlir::TypeRange {}, mlir::TypeRange {}),
      unwrap(location)
  );
  unwrap(builder)->setInsertionPointToStart(newModule->getBody());
  return newModule;
}

static llzk::verif::ContractOp createCppContract(
    MlirOpBuilder builder, MlirLocation location, llvm::StringRef name, llvm::StringRef target
) {
  auto contract = unwrap(builder)->create<llzk::verif::ContractOp>(
      unwrap(location), name, mlir::SymbolRefAttr::get(unwrap(builder)->getContext(), target),
      mlir::FunctionType::get(unwrap(builder)->getContext(), {}, {}), mlir::ArrayAttr()
  );
  contract.getBody().emplaceBlock();
  return contract;
}

} // namespace

TEST_F(CAPITest, llzk_verif_include_op_build_smoke) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation location = mlirLocationUnknownGet(context);
  auto module = cppNewModuleAndSetInsertionPoint(builder, location);

  llzk::ModuleBuilder modBuilder(module.get());
  auto funcType = mlir::FunctionType::get(unwrap(context), {}, {});
  modBuilder.insertFreeFunc("target", funcType, unwrap(location));

  auto base = unwrap(builder)->create<llzk::verif::ContractOp>(
      unwrap(location), "Base", mlir::SymbolRefAttr::get(unwrap(context), "target"), funcType,
      mlir::ArrayAttr()
  );
  base.getBody().emplaceBlock();

  auto wrapper = unwrap(builder)->create<llzk::verif::ContractOp>(
      unwrap(location), "Wrapper", mlir::SymbolRefAttr::get(unwrap(context), "target"), funcType,
      mlir::ArrayAttr()
  );
  wrapper.getBody().emplaceBlock();
  unwrap(builder)->setInsertionPointToStart(&wrapper.getBody().front());

  MlirOperation includeOp = llzkVerif_IncludeOpBuild(
      builder, location, createFlatSymbolRefAttr(context, "Base"), {.values = nullptr, .size = 0},
      mlirAttributeGetNull()
  );

  EXPECT_NE(includeOp.ptr, (void *)NULL);
  EXPECT_TRUE(llzkOperationIsA_Verif_IncludeOp(includeOp));
  EXPECT_TRUE(mlirOperationVerify(includeOp));
  EXPECT_TRUE(mlirAttributeEqual(
      llzkVerif_IncludeOpGetCallee(includeOp), createFlatSymbolRefAttr(context, "Base")
  ));

  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzk_verif_contract_op_build_and_verify_in_module) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation location = mlirLocationUnknownGet(context);
  auto module = cppNewModuleAndSetInsertionPoint(builder, location);

  llzk::ModuleBuilder modBuilder(module.get());
  auto funcType = mlir::FunctionType::get(unwrap(context), {}, {});
  modBuilder.insertFreeFunc("target", funcType, unwrap(location));
  unwrap(builder)->setInsertionPointToStart(module->getBody());

  MlirOperation op = llzkVerif_ContractOpBuild(
      builder, location,
      mlirIdentifierGet(context, mlirStringRefCreateFromCString("ContractUnderTest")),
      createFlatSymbolRefAttr(context, "target"), createEmptyFunctionTypeAttr(context),
      mlirAttributeGetNull()
  );

  EXPECT_NE(op.ptr, (void *)NULL);
  EXPECT_TRUE(llzkOperationIsA_Verif_ContractOp(op));
  EXPECT_TRUE(mlirOperationVerify(op));
  EXPECT_TRUE(mlirAttributeEqual(
      llzkVerif_ContractOpGetTarget(op), createFlatSymbolRefAttr(context, "target")
  ));
  EXPECT_TRUE(!llzkVerif_ContractOpHasStructTarget(op));
  EXPECT_TRUE(llzkVerif_ContractOpHasFuncTarget(op));

  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzk_verif_include_op_build_and_resolve_callable) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation location = mlirLocationUnknownGet(context);
  auto module = cppNewModuleAndSetInsertionPoint(builder, location);

  llzk::ModuleBuilder modBuilder(module.get());
  auto funcType = mlir::FunctionType::get(unwrap(context), {}, {});
  modBuilder.insertFreeFunc("target", funcType, unwrap(location));

  auto base = unwrap(builder)->create<llzk::verif::ContractOp>(
      unwrap(location), "Base", mlir::SymbolRefAttr::get(unwrap(context), "target"), funcType,
      mlir::ArrayAttr()
  );
  base.getBody().emplaceBlock();

  auto wrapper = unwrap(builder)->create<llzk::verif::ContractOp>(
      unwrap(location), "Wrapper", mlir::SymbolRefAttr::get(unwrap(context), "target"), funcType,
      mlir::ArrayAttr()
  );
  wrapper.getBody().emplaceBlock();
  unwrap(builder)->setInsertionPointToStart(&wrapper.getBody().front());

  MlirOperation includeOp = llzkVerif_IncludeOpBuild(
      builder, location, createFlatSymbolRefAttr(context, "Base"), {.values = nullptr, .size = 0},
      mlirAttributeGetNull()
  );

  ASSERT_NE(includeOp.ptr, (void *)NULL);
  EXPECT_TRUE(mlirOperationVerify(includeOp));
  EXPECT_TRUE(!llzkVerif_IncludeOpContractTargetsStruct(includeOp));
  EXPECT_TRUE(mlirTypeEqual(
      llzkVerif_IncludeOpGetTypeSignature(includeOp),
      mlirFunctionTypeGet(context, 0, nullptr, 0, nullptr)
  ));

  MlirOperation callable = llzkVerif_IncludeOpResolveCallable(includeOp);
  ASSERT_NE(callable.ptr, (void *)NULL);
  EXPECT_TRUE(llzkOperationIsA_Verif_ContractOp(callable));
  EXPECT_TRUE(mlirAttributeEqual(
      llzkVerif_ContractOpGetSymName(callable), createStringAttr(context, "Base")
  ));

  mlirOpBuilderDestroy(builder);
}

// Implementation for `ContractOp_build_pass` test.
std::unique_ptr<ContractOpBuildFuncHelper> ContractOpBuildFuncHelper::get() {
  struct Impl : public ContractOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;

    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      this->parentModule = createModuleWithTargetFunc(testClass, builder, location, "target");
      return llzkVerif_ContractOpBuild(
          builder, location,
          mlirIdentifierGet(testClass.context, mlirStringRefCreateFromCString("ContractUnderTest")),
          createFlatSymbolRefAttr(testClass.context, "target"),
          createEmptyFunctionTypeAttr(testClass.context), mlirAttributeGetNull()
      );
    }
  };
  return std::make_unique<Impl>();
}

namespace {
struct VerifConditionOpBuildBase {
  mlir::OwningOpRef<mlir::ModuleOp> parentModule;

  MlirValue
  prepareInsertionSite(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) {
    this->parentModule = createModuleWithTargetFunc(testClass, builder, location, "target");
    auto contract = createCppContract(builder, location, "ContractUnderTest", "target");
    unwrap(builder)->setInsertionPointToStart(&contract.getBody().front());
    return wrap(CAPITest::cppGenBoolConstant(builder, location));
  }
};
} // namespace

std::unique_ptr<EnsureComputeOpBuildFuncHelper> EnsureComputeOpBuildFuncHelper::get() {
  struct Impl : public EnsureComputeOpBuildFuncHelper, VerifConditionOpBuildBase {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      MlirValue cond = prepareInsertionSite(testClass, builder, location);
      return llzkVerif_EnsureComputeOpBuild(builder, location, cond);
    }
  };
  return std::make_unique<Impl>();
}

std::unique_ptr<EnsureConstrainOpBuildFuncHelper> EnsureConstrainOpBuildFuncHelper::get() {
  struct Impl : public EnsureConstrainOpBuildFuncHelper, VerifConditionOpBuildBase {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      MlirValue cond = prepareInsertionSite(testClass, builder, location);
      return llzkVerif_EnsureConstrainOpBuild(builder, location, cond);
    }
  };
  return std::make_unique<Impl>();
}

std::unique_ptr<RequireComputeOpBuildFuncHelper> RequireComputeOpBuildFuncHelper::get() {
  struct Impl : public RequireComputeOpBuildFuncHelper, VerifConditionOpBuildBase {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      MlirValue cond = prepareInsertionSite(testClass, builder, location);
      return llzkVerif_RequireComputeOpBuild(builder, location, cond);
    }
  };
  return std::make_unique<Impl>();
}

std::unique_ptr<RequireConstrainOpBuildFuncHelper> RequireConstrainOpBuildFuncHelper::get() {
  struct Impl : public RequireConstrainOpBuildFuncHelper, VerifConditionOpBuildBase {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      MlirValue cond = prepareInsertionSite(testClass, builder, location);
      return llzkVerif_RequireConstrainOpBuild(builder, location, cond);
    }
  };
  return std::make_unique<Impl>();
}
