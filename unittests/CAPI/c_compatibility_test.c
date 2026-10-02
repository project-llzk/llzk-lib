/*===-- c_compatibility_test.c - Test C API from pure C -----------*- C -*-===//
 *
 * Part of the LLZK Project, under the Apache License v2.0.
 * See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
 * SPDX-License-Identifier: Apache-2.0
 *
 *===----------------------------------------------------------------------===//
 *
 * This file tests that the LLZK C API can be consumed from pure C code.
 * It verifies that all headers are properly wrapped in extern "C" blocks.
 *
 *===----------------------------------------------------------------------===*/

#include "llzk-c/Analysis.h"
#include "llzk-c/Builder.h"
#include "llzk-c/Constants.h"
#include "llzk-c/DialectRegistration.h"
#include "llzk-c/Support.h"
#include "llzk-c/Transforms.h"
#include "llzk-c/Typing.h"
#include "llzk-c/Validators.h"

/* Include all dialect headers */
#include "llzk-c/Dialect/Array.h"
#include "llzk-c/Dialect/Bool.h"
#include "llzk-c/Dialect/Cast.h"
#include "llzk-c/Dialect/Constrain.h"
#include "llzk-c/Dialect/Felt.h"
#include "llzk-c/Dialect/Function.h"
#include "llzk-c/Dialect/Global.h"
#include "llzk-c/Dialect/Include.h"
#include "llzk-c/Dialect/LLZK.h"
#include "llzk-c/Dialect/POD.h"
#include "llzk-c/Dialect/Poly.h"
#include "llzk-c/Dialect/RAM.h"
#include "llzk-c/Dialect/String.h"
#include "llzk-c/Dialect/Struct.h"
#include "llzk-c/Dialect/Verif.h"
#include "llzk-c/Target/PCL.h"
#include "llzk-c/Target/R1CS.h"
#include "llzk-c/Target/SMT.h"
#include "llzk-c/Target/ZKLean.h"

#include "llzk/Config/Config.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/IR.h>
#include <mlir-c/Pass.h>

#include <stdio.h>
#include <stdlib.h>

/// Check every individual dialect handle independently of bulk registration.
static int test_dialect_handles(void) {
  MlirContext context = mlirContextCreate();
  const MlirDialectHandle handles[] = {
      mlirGetDialectHandle__llzk__array__(),
      mlirGetDialectHandle__llzk__boolean__(),
      mlirGetDialectHandle__llzk__cast__(),
      mlirGetDialectHandle__llzk__constrain__(),
      mlirGetDialectHandle__llzk__felt__(),
      mlirGetDialectHandle__llzk__function__(),
      mlirGetDialectHandle__llzk__global__(),
      mlirGetDialectHandle__llzk__include__(),
      mlirGetDialectHandle__llzk__(),
      mlirGetDialectHandle__llzk__pod__(),
      mlirGetDialectHandle__llzk__polymorphic__(),
      mlirGetDialectHandle__llzk__ram__(),
      mlirGetDialectHandle__llzk__string__(),
      mlirGetDialectHandle__llzk__component__(),
      mlirGetDialectHandle__llzk__verif__(),
  };
  int failed = 0;
  for (size_t i = 0; i < sizeof(handles) / sizeof(handles[0]); ++i) {
    MlirStringRef name = mlirDialectHandleGetNamespace(handles[i]);
    if (mlirDialectIsNull(mlirDialectHandleLoadDialect(handles[i], context))) {
      fprintf(stderr, "Failed to load dialect %.*s\n", (int)name.length, name.data);
      failed = 1;
    }
  }
  mlirContextDestroy(context);
  return failed;
}

/// Check all pass group registrations, individual registrations, and constructors.
static int test_passes(MlirContext context) {
  mlirRegisterLLZKAnalysisPasses();
  mlirRegisterLLZKArrayTransformationPasses();
  mlirRegisterLLZKBoolTransformationPasses();
  mlirRegisterLLZKGlobalTransformationPasses();
  mlirRegisterLLZKIncludeTransformationPasses();
  mlirRegisterLLZKPodTransformationPasses();
  mlirRegisterLLZKPolymorphicTransformationPasses();
  mlirRegisterLLZKStructTransformationPasses();
  mlirRegisterLLZKTransformationPasses();
  mlirRegisterLLZKValidationPasses();
#if LLZK_WITH_PCL
  mlirRegisterPCLConversionPasses();
  mlirRegisterPCLTransformationPasses();
#endif
  mlirRegisterR1CSTransformationPasses();
  mlirRegisterSMTConversionPasses();
  mlirRegisterZKLeanConversionPasses();

  /* Keep each registration and constructor paired, with its name for diagnostics. */
  const struct {
    const char *name;
    void (*registerPass)(void);
    MlirPass (*createPass)(void);
    const char *nestedOperation;
  } passes[] = {
#define NESTED_PASS(name, operation) {#name, mlirRegister##name, mlirCreate##name, operation}
#define PASS(name) NESTED_PASS(name, NULL)
      PASS(LLZKAnalysisCallGraphPrinterPass),
      PASS(LLZKAnalysisCallGraphSCCsPrinterPass),
      PASS(LLZKAnalysisConstraintDependencyGraphPrinterPass),
      PASS(LLZKAnalysisIntervalAnalysisPrinterPass),
      PASS(LLZKAnalysisPredecessorPrinterPass),
      PASS(LLZKAnalysisSymbolDefTreePrinterPass),
      PASS(LLZKAnalysisSymbolUseGraphPrinterPass),
      PASS(LLZKArrayTransformationArrayToScalarPass),
      PASS(LLZKArrayTransformationStraightLineStaticArrayPromotionPass),
      PASS(LLZKBoolTransformationLowerBoolQuantifiersPass),
      PASS(LLZKGlobalTransformationConstGlobalPropagationPass),
      PASS(LLZKIncludeTransformationInlineIncludesPass),
      PASS(LLZKPodTransformationPodToScalarPass),
      PASS(LLZKPolymorphicTransformationEmptyTemplateRemovalPass),
      PASS(LLZKPolymorphicTransformationFlatteningPass),
      PASS(LLZKPolymorphicTransformationTypeVarInferencePass),
      PASS(LLZKPolymorphicTransformationWildcardArraySpecializationPass),
      PASS(LLZKStructTransformationInlineStructsPass),
      PASS(LLZKTransformationComputeConstrainToProductPass),
      PASS(LLZKTransformationEnforceNoMemberOverwritePass),
      PASS(LLZKTransformationFuseProductControlFlowPass),
      PASS(LLZKTransformationInlineFreeFunctionsPass),
      PASS(LLZKTransformationPolyLoweringPass),
      PASS(LLZKTransformationRedundantOperationEliminationPass),
      PASS(LLZKTransformationRedundantReadAndWriteEliminationPass),
      PASS(LLZKTransformationRemoveUnusedDiscardableAllocationsPass),
      PASS(LLZKTransformationUnusedDeclarationEliminationPass),
      PASS(LLZKTransformationWhileToForPass),
      PASS(LLZKValidationMemberWriteValidatorPass),
#if LLZK_WITH_PCL
      PASS(PCLConversionPCLLoweringPass),
      NESTED_PASS(PCLTransformationTrimExprSizePass, "func.func"),
#endif
      PASS(R1CSTransformationR1CSLoweringPass),
      PASS(SMTConversionSMTCFLoweringPass),
      PASS(SMTConversionSMTLoweringPass),
      PASS(SMTConversionSMTNaiveLoweringPass),
      PASS(ZKLeanConversionConvertLLZKToZKLeanPass),
      PASS(ZKLeanConversionConvertZKLeanToLLZKPass),
#undef PASS
#undef NESTED_PASS
  };
  MlirPassManager manager = mlirPassManagerCreate(context);
  int failed = 0;
  for (size_t i = 0; i < sizeof(passes) / sizeof(passes[0]); ++i) {
    passes[i].registerPass();
    MlirPass pass = passes[i].createPass();
    if (pass.ptr == NULL) {
      fprintf(stderr, "Failed to create pass %s\n", passes[i].name);
      failed = 1;
    } else if (passes[i].nestedOperation != NULL) {
      /* Operation-specific passes must be added under their matching anchor. */
      MlirOpPassManager nested = mlirPassManagerGetNestedUnder(
          manager, mlirStringRefCreateFromCString(passes[i].nestedOperation)
      );
      mlirOpPassManagerAddOwnedPass(nested, pass);
    } else {
      mlirPassManagerAddOwnedPass(manager, pass);
    }
  }
  mlirPassManagerDestroy(manager);
  return failed;
}

/*
 * Test basic C API functionality
 */
int test_basic_api(void) {
  /* Create context */
  MlirContext context = mlirContextCreate();
  if (mlirContextIsNull(context)) {
    fprintf(stderr, "Failed to create MLIR context\n");
    return 1;
  }

  /* Register dialects */
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  llzkRegisterCoreDialects(registry);
  llzkRegisterPCLDialects(registry);
  llzkRegisterR1CSDialects(registry);
  llzkRegisterSMTDialects(registry);
  llzkRegisterZKLeanDialects(registry);
  llzkRegisterCorePasses(registry);
  llzkRegisterPCLPasses(registry);
  llzkRegisterR1CSPasses(registry);
  llzkRegisterSMTPasses(registry);
  llzkRegisterZKLeanPasses(registry);
  mlirContextAppendDialectRegistry(context, registry);
  mlirContextLoadAllAvailableDialects(context);
  mlirDialectRegistryDestroy(registry);

  /* Check every dialect registered by the core and backend wrappers. */
  const char *dialects[] = {
      "llzk",    "array", "bool", "cast", "constrain", "felt",      "function",   "global",
      "include", "pod",   "poly", "ram",  "smt_info",  "string",    "struct",     "verif",
      "arith",   "scf",   "smt",  "r1cs", "ZKExpr",    "ZKBuilder", "ZKLeanLean", "func",
#if LLZK_WITH_PCL
      "pcl",
#endif
  };
  int missingDialect = 0;
  for (size_t i = 0; i < sizeof(dialects) / sizeof(dialects[0]); ++i) {
    if (mlirDialectIsNull(
            mlirContextGetOrLoadDialect(context, mlirStringRefCreateFromCString(dialects[i]))
        )) {
      fprintf(stderr, "Failed to register dialect %s\n", dialects[i]);
      missingDialect = 1;
    }
  }
  if (missingDialect || test_passes(context)) {
    mlirContextDestroy(context);
    return 1;
  }

  /* Test creating a simple attribute */
  MlirAttribute publicAttr = llzkLlzk_PublicAttrGet(context);
  if (mlirAttributeIsNull(publicAttr)) {
    fprintf(stderr, "Failed to create PublicAttr\n");
    mlirContextDestroy(context);
    return 1;
  }

  /* Test "isa" check */
  int isPublic = llzkAttributeIsA_Llzk_PublicAttr(publicAttr);
  if (!isPublic) {
    fprintf(stderr, "PublicAttr type check failed\n");
    mlirContextDestroy(context);
    return 1;
  }

  /* Clean up */
  mlirContextDestroy(context);

  printf("All C API tests passed!\n");
  return 0;
}

int main(void) {
  if (test_dialect_handles()) {
    return 1;
  }
  int result = test_basic_api();
  return result;
}
