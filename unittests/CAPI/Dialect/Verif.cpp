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

#include "llzk-c/Builder.h"

#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Shared/Builders.h"
#include "llzk/Dialect/Verif/IR/Ops.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/IR.h>

#include <mlir/CAPI/Wrap.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

#include <cstdint>

// Include the auto-generated tests
#include "llzk/Dialect/Verif/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Verif/IR/Ops.capi.test.cpp.inc"

namespace {

static MlirAttribute createFlatSymbolRefAttr(MlirContext ctx, const char *name) {
  return mlirFlatSymbolRefAttrGet(ctx, mlirStringRefCreateFromCString(name));
}

static MlirAttribute createNestedSymbolRefAttr(
    MlirContext ctx, const char *root, std::initializer_list<const char *> nested
) {
  llvm::SmallVector<MlirAttribute> nestedRefs;
  nestedRefs.reserve(nested.size());
  for (const char *piece : nested) {
    nestedRefs.push_back(createFlatSymbolRefAttr(ctx, piece));
  }
  return mlirSymbolRefAttrGet(
      ctx, mlirStringRefCreateFromCString(root), llzk::checkedCast<intptr_t>(nestedRefs.size()),
      nestedRefs.data()
  );
}

static MlirAttribute createStringAttr(MlirContext ctx, const char *value) {
  return mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(value));
}

static MlirAttribute createEmptyFunctionTypeAttr(MlirContext ctx) {
  MlirType type = mlirFunctionTypeGet(ctx, 0, nullptr, 0, nullptr);
  return mlirTypeAttrGet(type);
}

static mlir::OwningOpRef<mlir::ModuleOp> createModuleWithTargetFunc(
    const CAPITest &test, MlirOpBuilder builder, MlirLocation location, llvm::StringRef name,
    llvm::function_ref<void(mlir::OpBuilder &)> fnBody = nullptr
) {
  auto newModule = test.cppNewModuleAndSetInsertionPoint(builder, location);
  llzk::ModuleBuilder modBuilder(newModule.get());
  modBuilder.insertFreeFunc(
      name, mlir::FunctionType::get(unwrap(test.context), mlir::TypeRange {}, mlir::TypeRange {}),
      unwrap(location), fnBody
  );
  unwrap(builder)->setInsertionPointToStart(newModule->getBody());
  return newModule;
}

static llzk::verif::ContractOp createCppContract(
    MlirOpBuilder builder, MlirLocation location, llvm::StringRef name, llvm::StringRef target
) {
  return unwrap(builder)->create<llzk::verif::ContractOp>(
      unwrap(location), name, mlir::SymbolRefAttr::get(unwrap(builder)->getContext(), target),
      mlir::FunctionType::get(unwrap(builder)->getContext(), {}, {}), mlir::ArrayAttr()
  );
}

static void expectContractHasImplicitTerminator(MlirOperation op) {
  auto contract = llvm::cast<llzk::verif::ContractOp>(unwrap(op));
  auto &body = contract.getBody();
  ASSERT_FALSE(body.empty());
  ASSERT_EQ(body.getBlocks().size(), 1u);
  ASSERT_NE(body.front().getTerminator(), nullptr);
  EXPECT_TRUE(llvm::isa<llzk::verif::ContractEndOp>(body.front().getTerminator()));
}

} // namespace

TEST_F(CAPITest, llzkVerifIncludeOpBuildSmoke) {
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
  expectContractHasImplicitTerminator(wrap(base));

  auto wrapper = unwrap(builder)->create<llzk::verif::ContractOp>(
      unwrap(location), "Wrapper", mlir::SymbolRefAttr::get(unwrap(context), "target"), funcType,
      mlir::ArrayAttr()
  );
  unwrap(builder)->setInsertionPointToStart(&wrapper.getBody().front());

  MlirOperation includeOp = llzkVerif_IncludeOpBuild(
      builder, location, createFlatSymbolRefAttr(context, "Base"), {.values = nullptr, .size = 0},
      mlirAttributeGetNull()
  );

  EXPECT_TRUE(llzkOperationIsA_Verif_IncludeOp(includeOp));
  EXPECT_TRUE(mlirOperationVerify(includeOp));
  EXPECT_TRUE(mlirAttributeEqual(
      llzkVerif_IncludeOpGetCallee(includeOp), createFlatSymbolRefAttr(context, "Base")
  ));

  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzkVerifContractOpBuildAndVerifyInModule) {
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
  expectContractHasImplicitTerminator(op);

  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzkVerifContractOpBuildFromTarget) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation location = mlirLocationUnknownGet(context);
  auto module = cppNewModuleAndSetInsertionPoint(builder, location);

  llzk::ModuleBuilder modBuilder(module.get());
  auto funcType = mlir::FunctionType::get(unwrap(context), {}, {});
  modBuilder.insertFreeFunc("target", funcType, unwrap(location));
  unwrap(builder)->setInsertionPointToStart(module->getBody());

  MlirOperation op = llzkVerif_ContractOpBuildFromTargetIdentifier(
      builder, location,
      mlirIdentifierGet(context, mlirStringRefCreateFromCString("ContractUnderTest")),
      mlirIdentifierGet(context, mlirStringRefCreateFromCString("target"))
  );

  EXPECT_NE(op.ptr, (void *)NULL);
  EXPECT_TRUE(llzkOperationIsA_Verif_ContractOp(op));
  EXPECT_TRUE(mlirOperationVerify(op));
  EXPECT_TRUE(mlirAttributeEqual(
      llzkVerif_ContractOpGetTarget(op), createFlatSymbolRefAttr(context, "target")
  ));
  EXPECT_TRUE(!llzkVerif_ContractOpHasStructTarget(op));
  EXPECT_TRUE(llzkVerif_ContractOpHasFuncTarget(op));
  expectContractHasImplicitTerminator(op);

  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzkVerifContractOpBuildFromTargetAttr) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation location = mlirLocationUnknownGet(context);
  auto module = cppNewModuleAndSetInsertionPoint(builder, location);

  llzk::ModuleBuilder modBuilder(module.get());
  auto funcType = mlir::FunctionType::get(unwrap(context), {}, {});
  modBuilder.insertFreeFunc("target", funcType, unwrap(location));
  unwrap(builder)->setInsertionPointToStart(module->getBody());

  MlirOperation op = llzkVerif_ContractOpBuildFromTargetAttr(
      builder, location,
      mlirIdentifierGet(context, mlirStringRefCreateFromCString("ContractUnderTest")),
      createFlatSymbolRefAttr(context, "target")
  );

  EXPECT_NE(op.ptr, (void *)NULL);
  EXPECT_TRUE(llzkOperationIsA_Verif_ContractOp(op));
  EXPECT_TRUE(mlirOperationVerify(op));
  EXPECT_TRUE(mlirAttributeEqual(
      llzkVerif_ContractOpGetTarget(op), createFlatSymbolRefAttr(context, "target")
  ));
  expectContractHasImplicitTerminator(op);

  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzkVerifContractOpBuildFromTargetAttrNested) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation location = mlirLocationUnknownGet(context);
  auto module = parseSourceString<mlir::ModuleOp>(
      R"mlir(
module attributes {llzk.lang} {
  poly.template @T {
    poly.param @N : index
    function.def @target() {
      function.return
    }
  }
}
)mlir",
      mlir::ParserConfig(unwrap(context))
  );
  ASSERT_TRUE(module);
  unwrap(builder)->setInsertionPointToStart(module->getBody());

  MlirAttribute target = createNestedSymbolRefAttr(context, "T", {"target"});
  MlirOperation op = llzkVerif_ContractOpBuildFromTargetAttr(
      builder, location,
      mlirIdentifierGet(context, mlirStringRefCreateFromCString("ContractUnderTest")), target
  );

  EXPECT_NE(op.ptr, (void *)NULL);
  EXPECT_TRUE(llzkOperationIsA_Verif_ContractOp(op));
  EXPECT_TRUE(mlirOperationVerify(op));
  EXPECT_TRUE(mlirAttributeEqual(llzkVerif_ContractOpGetTarget(op), target));
  expectContractHasImplicitTerminator(op);

  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzkVerifIncludeOpBuildAndResolveCallable) {
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
  expectContractHasImplicitTerminator(wrap(base));

  auto wrapper = unwrap(builder)->create<llzk::verif::ContractOp>(
      unwrap(location), "Wrapper", mlir::SymbolRefAttr::get(unwrap(context), "target"), funcType,
      mlir::ArrayAttr()
  );
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

struct ContractOpBuildFuncHelper : public TestAnyBuildFuncHelper<CAPITest> {
  bool callIsA(MlirOperation op) override { return llzkOperationIsA_Verif_ContractOp(op); }
  void doOtherChecks(MlirOperation op) override { expectContractHasImplicitTerminator(op); }
  static std::unique_ptr<ContractOpBuildFuncHelper> get();

protected:
  ContractOpBuildFuncHelper() = default;
};

TEST_F(CAPITest, ContractOp_build_pass) { ContractOpBuildFuncHelper::get()->run(*this); }

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

std::unique_ptr<VerifProveOpBuildFuncHelper> VerifProveOpBuildFuncHelper::get() {
  struct Impl : public VerifProveOpBuildFuncHelper, VerifConditionOpBuildBase {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      MlirValue cond = prepareInsertionSite(testClass, builder, location);
      return llzkVerif_VerifProveOpBuild(builder, location, cond);
    }
  };
  return std::make_unique<Impl>();
}

std::unique_ptr<VerifAssertOpBuildFuncHelper> VerifAssertOpBuildFuncHelper::get() {
  struct Impl : public VerifAssertOpBuildFuncHelper, VerifConditionOpBuildBase {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      MlirValue cond = prepareInsertionSite(testClass, builder, location);
      return llzkVerif_VerifAssertOpBuild(builder, location, cond);
    }
  };
  return std::make_unique<Impl>();
}

std::unique_ptr<VerifSMTProveOpBuildFuncHelper> VerifSMTProveOpBuildFuncHelper::get() {
  struct Impl : public VerifSMTProveOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &, MlirOpBuilder builder, MlirLocation location) override {
      MlirValue cond = wrap(CAPITest::cppGenSMTBoolConstant(builder, location));
      return llzkVerif_VerifSMTProveOpBuild(builder, location, cond);
    }
  };
  return std::make_unique<Impl>();
}

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

std::unique_ptr<ContractEndOpBuildFuncHelper> ContractEndOpBuildFuncHelper::get() {
  struct Impl : public ContractEndOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;

    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      this->parentModule = createModuleWithTargetFunc(testClass, builder, location, "target");
      auto contract = createCppContract(builder, location, "ContractUnderTest", "target");
      contract.getBody().front().getTerminator()->erase();
      unwrap(builder)->setInsertionPointToEnd(&contract.getBody().front());
      return llzkVerif_ContractEndOpBuild(builder, location);
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

TEST_F(CAPITest, llzkVerifInvariantOpBuild) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation location = mlirLocationUnknownGet(context);
  auto module = parseSourceString<mlir::ModuleOp>(
      R"mlir(
module attributes {llzk.lang} {
 function.def @target() attributes {function.allow_witness} {
    scf.while : () -> () {
      %true = arith.constant  true 
      scf.condition(%true)
    } do {
      scf.yield
    } attributes {loop_label = "loopA"}
    function.return
  }
}
)mlir",
      mlir::ParserConfig(unwrap(context))
  );
  ASSERT_TRUE(module);
  unwrap(builder)->setInsertionPointToEnd(module->getBody());

  auto contract = createCppContract(builder, location, "ContractUnderTest", "target");
  unwrap(builder)->setInsertionPointToStart(&contract.getBody().front());

  auto invariant = llzkVerif_InvariantOpBuild(
      builder, location, mlirStringRefCreateFromCString("loopA"), 0, nullptr, nullptr
  );
  EXPECT_TRUE(mlirOperationVerify(invariant));

  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzkVerifInvariantOpBuildWithArgs) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation location = mlirLocationUnknownGet(context);
  auto module = parseSourceString<mlir::ModuleOp>(
      R"mlir(
module attributes {llzk.lang} {
 function.def @target() attributes {function.allow_witness} {
      %c0 = arith.constant 0 : index 
      %c10 = arith.constant 10 : index 
      %c1 = arith.constant 1  : index 
      scf.for %iv = %c0 to %c10 step %c1 {
        scf.yield
      } {loop_label = "loopA"}
    function.return
  }
}
)mlir",
      mlir::ParserConfig(unwrap(context))
  );
  ASSERT_TRUE(module);
  unwrap(builder)->setInsertionPointToEnd(module->getBody());
  size_t argCount = 4;
  llvm::SmallVector<MlirType> argTypes(argCount, mlirIndexTypeGet(context));
  llvm::SmallVector<MlirLocation> argLocs(argCount, location);

  auto contract = createCppContract(builder, location, "ContractUnderTest", "target");
  unwrap(builder)->setInsertionPointToStart(&contract.getBody().front());

  auto invariant = llzkVerif_InvariantOpBuild(
      builder, location, mlirStringRefCreateFromCString("loopA"), static_cast<intptr_t>(argCount),
      argTypes.data(), argLocs.data()
  );
  EXPECT_TRUE(mlirOperationVerify(invariant));

  mlirOpBuilderDestroy(builder);
}

namespace {
struct VerifInvariantInnerOpBuildBase {
  mlir::OwningOpRef<mlir::ModuleOp> parentModule;

  mlir::ValueRange
  prepareInsertionSite(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) {
    this->parentModule = parseSourceString<mlir::ModuleOp>(
        R"mlir(
module attributes {llzk.lang} {
 function.def @target() attributes {function.allow_witness} {
      %c0 = arith.constant 0 : index 
      %c10 = arith.constant 10 : index 
      %c1 = arith.constant 1  : index 
      scf.for %iv = %c0 to %c10 step %c1 {
        scf.yield
      } {loop_label = "loopA"}
    function.return
  }
}
)mlir",
        mlir::ParserConfig(unwrap(testClass.context))
    );
    unwrap(builder)->setInsertionPointToEnd(parentModule->getBody());

    size_t argCount = 4;
    llvm::SmallVector<MlirType> argTypes(argCount, mlirIndexTypeGet(testClass.context));
    llvm::SmallVector<MlirLocation> argLocs(argCount, location);

    auto contract = createCppContract(builder, location, "ContractUnderTest", "target");
    unwrap(builder)->setInsertionPointToStart(&contract.getBody().front());

    auto invariant = llzkVerif_InvariantOpBuild(
        builder, location, mlirStringRefCreateFromCString("loopA"), static_cast<intptr_t>(argCount),
        argTypes.data(), argLocs.data()
    );
    auto *invariantBody = cast<llzk::verif::InvariantOp>(unwrap(invariant)).getBody();
    unwrap(builder)->setInsertionPointToStart(invariantBody);
    return invariantBody->getArguments();
  }
};
} // namespace

std::unique_ptr<IncreasesOpBuildFuncHelper> IncreasesOpBuildFuncHelper::get() {
  struct Impl : public IncreasesOpBuildFuncHelper, VerifInvariantInnerOpBuildBase {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      auto args = prepareInsertionSite(testClass, builder, location);
      auto step = unwrap(builder)->create<llzk::cast::IntToFeltOp>(unwrap(location), args[3]);
      return llzkVerif_IncreasesOpBuild(builder, location, wrap(step->getResult(0)));
    }
  };
  return std::make_unique<Impl>();
}

std::unique_ptr<DecreasesOpBuildFuncHelper> DecreasesOpBuildFuncHelper::get() {
  struct Impl : public DecreasesOpBuildFuncHelper, VerifInvariantInnerOpBuildBase {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      auto args = prepareInsertionSite(testClass, builder, location);
      auto step = unwrap(builder)->create<llzk::cast::IntToFeltOp>(unwrap(location), args[3]);
      return llzkVerif_DecreasesOpBuild(builder, location, wrap(step->getResult(0)));
    }
  };
  return std::make_unique<Impl>();
}

std::unique_ptr<StepOpBuildFuncHelper> StepOpBuildFuncHelper::get() {
  struct Impl : public StepOpBuildFuncHelper, VerifInvariantInnerOpBuildBase {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      prepareInsertionSite(testClass, builder, location);
      auto op = llzkVerif_StepOpBuild(builder, location);
      auto *region = unwrap(llzkVerif_StepOpGetRegion(op));
      unwrap(builder)->setInsertionPointToEnd(&region->emplaceBlock());
      auto trueOp = unwrap(builder)->create<mlir::arith::ConstantIntOp>(
          unwrap(location), 1, unwrap(builder)->getI1Type()
      );
      unwrap(builder)->create<llzk::verif::StepYieldOp>(unwrap(location), trueOp);
      return op;
    }
  };
  return std::make_unique<Impl>();
}

std::unique_ptr<StepYieldOpBuildFuncHelper> StepYieldOpBuildFuncHelper::get() {
  struct Impl : public StepYieldOpBuildFuncHelper, VerifInvariantInnerOpBuildBase {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      prepareInsertionSite(testClass, builder, location);
      auto stepOp = unwrap(builder)->create<llzk::verif::StepOp>(unwrap(location));
      unwrap(builder)->setInsertionPointToEnd(&stepOp.getRegion().emplaceBlock());
      auto trueOp = unwrap(builder)->create<mlir::arith::ConstantIntOp>(
          unwrap(location), 1, unwrap(builder)->getI1Type()
      );
      return llzkVerif_StepYieldOpBuild(builder, location, wrap(trueOp.getResult()));
    }
  };
  return std::make_unique<Impl>();
}

std::unique_ptr<OldOpBuildFuncHelper> OldOpBuildFuncHelper::get() {
  struct Impl : public OldOpBuildFuncHelper, VerifInvariantInnerOpBuildBase {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      auto args = prepareInsertionSite(testClass, builder, location);
      auto stepOp = unwrap(builder)->create<llzk::verif::StepOp>(unwrap(location));
      unwrap(builder)->setInsertionPointToEnd(&stepOp.getRegion().emplaceBlock());
      auto op = llzkVerif_OldOpBuild(builder, location, wrap(args[1]));

      auto trueOp = unwrap(builder)->create<mlir::arith::ConstantIntOp>(
          unwrap(location), 1, unwrap(builder)->getI1Type()
      );
      unwrap(builder)->create<llzk::verif::StepYieldOp>(unwrap(location), trueOp);
      return op;
    }
  };
  return std::make_unique<Impl>();
}
