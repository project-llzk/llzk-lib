//===-- Validators.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Validators.h"

#include "CAPITestBase.h"

TEST_F(CAPITest, register_validation_passes_and_create) {
  mlirRegisterLLZKValidationPasses();
  auto manager = mlirPassManagerCreate(context);

  auto pass = mlirCreateLLZKValidationMemberWriteValidatorPass();
  mlirPassManagerAddOwnedPass(manager, pass);

  mlirPassManagerDestroy(manager);
}

TEST_F(CAPITest, register_validation_member_write_validator_pass_and_create) {
  mlirRegisterLLZKValidationMemberWriteValidatorPass();
  auto manager = mlirPassManagerCreate(context);

  auto pass = mlirCreateLLZKValidationMemberWriteValidatorPass();
  mlirPassManagerAddOwnedPass(manager, pass);

  mlirPassManagerDestroy(manager);
}
