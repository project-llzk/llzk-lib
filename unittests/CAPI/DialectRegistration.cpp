//===-- DialectRegistration.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/DialectRegistration.h"

#include "llzk/Config/Config.h"

#include <mlir-c/IR.h>

#include <mlir/Pass/PassRegistry.h>

#include <gtest/gtest.h>
#include <initializer_list>

namespace {

/// Verify that a registration wrapper makes each expected dialect loadable.
void checkDialects(
    void (*registerDialects)(MlirDialectRegistry), std::initializer_list<const char *> names
) {
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  registerDialects(registry);
  MlirContext context = mlirContextCreateWithRegistry(registry, false);
  mlirDialectRegistryDestroy(registry);
  for (const char *name : names) {
    EXPECT_FALSE(mlirDialectIsNull(
        mlirContextGetOrLoadDialect(context, mlirStringRefCreateFromCString(name))
    )) << name;
  }
  mlirContextDestroy(context);
}

/// Verify that a registration wrapper exposes the expected passes by name.
void checkPasses(
    void (*registerPasses)(MlirDialectRegistry), std::initializer_list<const char *> names
) {
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  registerPasses(registry);
  for (const char *name : names) {
    EXPECT_NE(mlir::PassInfo::lookup(name), nullptr) << name;
  }
  mlirDialectRegistryDestroy(registry);
}

} // namespace

TEST(DialectRegistration, RegisterCoreDialects) {
  checkDialects(llzkRegisterCoreDialects, {"llzk", "felt", "array"});
}

TEST(DialectRegistration, RegisterCorePasses) {
  checkPasses(
      llzkRegisterCorePasses, {"llzk-duplicate-read-write-elim", "llzk-poly-lowering-pass"}
  );
}

TEST(DialectRegistration, RegisterR1CSDialects) {
  checkDialects(llzkRegisterR1CSDialects, {"r1cs"});
}

TEST(DialectRegistration, RegisterR1CSPasses) {
  checkPasses(llzkRegisterR1CSPasses, {"llzk-r1cs-lowering"});
}

TEST(DialectRegistration, RegisterZKLeanDialects) {
  checkDialects(llzkRegisterZKLeanDialects, {"ZKExpr", "ZKBuilder", "ZKLeanLean", "func"});
}

TEST(DialectRegistration, RegisterZKLeanPasses) {
  checkPasses(llzkRegisterZKLeanPasses, {"convert-llzk-to-zklean", "convert-zklean-to-llzk"});
}

TEST(DialectRegistration, RegisterPCLDialects) {
#if LLZK_WITH_PCL
  checkDialects(llzkRegisterPCLDialects, {"pcl", "func"});
#else
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  llzkRegisterPCLDialects(registry);
  MlirContext context = mlirContextCreateWithRegistry(registry, false);
  EXPECT_TRUE(
      mlirDialectIsNull(mlirContextGetOrLoadDialect(context, mlirStringRefCreateFromCString("pcl")))
  );
  mlirContextDestroy(context);
  mlirDialectRegistryDestroy(registry);
#endif
}

TEST(DialectRegistration, RegisterPCLPasses) {
#if LLZK_WITH_PCL
  checkPasses(llzkRegisterPCLPasses, {"llzk-to-pcl", "pcl-trim-expression-size"});
#else
  checkPasses(llzkRegisterPCLPasses, {});
  EXPECT_EQ(mlir::PassInfo::lookup("llzk-to-pcl"), nullptr);
#endif
}
