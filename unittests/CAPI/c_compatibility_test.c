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
#include "llzk-c/Dialect/String.h"
#include "llzk-c/Dialect/Struct.h"

/* Include generated backend conversion pass APIs from pure C. */
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/IR.h>
#include <mlir-c/RegisterEverything.h>

#include <stdio.h>
#include <stdlib.h>

#include "smt/Conversions/ConversionPasses.capi.h.inc"
#include "zklean/Conversions/ConversionPasses.capi.h.inc"

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
  mlirRegisterAllDialects(registry);
  llzkRegisterCoreDialects(registry);
  llzkRegisterPCLDialects(registry);
  llzkRegisterR1CSDialects(registry);
  llzkRegisterZKLeanDialects(registry);
  llzkRegisterCorePasses(registry);
  llzkRegisterPCLPasses(registry);
  llzkRegisterR1CSPasses(registry);
  llzkRegisterZKLeanPasses(registry);
  mlirContextAppendDialectRegistry(context, registry);
  mlirContextLoadAllAvailableDialects(context);
  mlirDialectRegistryDestroy(registry);

  /* Verify group registration, individual registration, and all pass constructors link. */
  mlirRegisterSMTConversionPasses();
  mlirRegisterSMTConversionSMTCFLoweringPass();
  mlirRegisterSMTConversionSMTLoweringPass();
  mlirRegisterSMTConversionSMTNaiveLoweringPass();
  mlirRegisterZKLeanConversionPasses();
  mlirRegisterZKLeanConversionConvertLLZKToZKLeanPass();
  mlirRegisterZKLeanConversionConvertZKLeanToLLZKPass();

  MlirPass passes[] = {
      mlirCreateSMTConversionSMTCFLoweringPass(),
      mlirCreateSMTConversionSMTLoweringPass(),
      mlirCreateSMTConversionSMTNaiveLoweringPass(),
      mlirCreateZKLeanConversionConvertLLZKToZKLeanPass(),
      mlirCreateZKLeanConversionConvertZKLeanToLLZKPass(),
  };
  MlirPassManager manager = mlirPassManagerCreate(context);
  int missingPass = 0;
  for (size_t i = 0; i < sizeof(passes) / sizeof(passes[0]); ++i) {
    if (passes[i].ptr == NULL) {
      fprintf(stderr, "Failed to create backend conversion pass %zu\n", i);
      missingPass = 1;
    } else {
      mlirPassManagerAddOwnedPass(manager, passes[i]);
    }
  }
  mlirPassManagerDestroy(manager);
  if (missingPass) {
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
  int result = test_basic_api();
  return result;
}
