//===-- Felt.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Felt.h"
#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/Support.h>

#include <llvm/ADT/APInt.h>

#include "../CAPITestBase.h"

TEST_F(CAPITest, mlir_get_dialect_handle_llzk_felt) { (void)mlirGetDialectHandle__llzk__felt__(); }

TEST_F(CAPITest, llzk_felt_const_attr_get) {
  auto attr = llzkFeltConstAttrGet(context, 0);
  EXPECT_NE(attr.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_felt_const_attr_get_with_field) {
  auto str = MlirStringRef {.data = "goldilocks", .length = 10};
  auto attr = llzkFeltConstAttrGetWithField(context, 0, str);
  EXPECT_NE(attr.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzkFeltConstAttrGetWithBits) {
  constexpr auto BITS = 128;
  auto attr = llzkFeltConstAttrGetWithBits(context, BITS, 0);
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto cxx_attr = llvm::dyn_cast<llzk::felt::FeltConstAttr>(unwrap(attr));
  EXPECT_TRUE(cxx_attr);
  EXPECT_EQ(cxx_attr.getFieldName(), nullptr);
  auto value = cxx_attr.getValue();
  EXPECT_EQ(value.getBitWidth(), BITS);
  EXPECT_EQ(value.getZExtValue(), 0);
}

TEST_F(CAPITest, llzkFeltConstAttrGetWithBitsWithField) {
  auto str = MlirStringRef {.data = "babybear", .length = 8};
  constexpr auto BITS = 128;
  auto attr = llzkFeltConstAttrGetWithBitsWithField(context, BITS, 0, str);
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto cxx_attr = llvm::dyn_cast<llzk::felt::FeltConstAttr>(unwrap(attr));
  EXPECT_TRUE(cxx_attr);
  EXPECT_EQ(cxx_attr.getFieldName().getValue(), str.data);
  auto value = cxx_attr.getValue();
  EXPECT_EQ(value.getBitWidth(), BITS);
  EXPECT_EQ(value.getZExtValue(), 0);
}

TEST_F(CAPITest, llzkFeltConstAttrGetFromString) {
  constexpr auto BITS = 64;
  auto str = MlirStringRef {.data = "123", .length = 3};
  auto attr = llzkFeltConstAttrGetFromString(context, BITS, str);
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto expected = llzk::felt::FeltConstAttr::get(
      unwrap(context), llvm::APInt(BITS, llvm::StringRef("123", 3), 10)
  );
  EXPECT_EQ(unwrap(attr), expected);
}

TEST_F(CAPITest, llzkFeltConstAttrGetFromStringWithField) {
  auto fieldName = MlirStringRef {.data = "bn254", .length = 5};
  constexpr auto BITS = 64;
  auto str = MlirStringRef {.data = "123", .length = 3};
  auto attr = llzkFeltConstAttrGetFromStringWithField(context, BITS, str, fieldName);
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto expected = llzk::felt::FeltConstAttr::get(
      unwrap(context), llvm::APInt(BITS, llvm::StringRef("123", 3), 10),
      mlir::StringAttr::get(unwrap(context), "bn254")
  );
  EXPECT_EQ(unwrap(attr), expected);
}

TEST_F(CAPITest, llzkFeltConstAttrGetFromParts) {
  constexpr auto BITS = 254;
  const uint64_t parts[] = {10, 20, 30, 40};
  auto attr = llzkFeltConstAttrGetFromParts(context, BITS, parts, 4);
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto expected =
      llzk::felt::FeltConstAttr::get(unwrap(context), llvm::APInt(BITS, llvm::ArrayRef(parts, 4)));
  EXPECT_EQ(unwrap(attr), expected);
}

TEST_F(CAPITest, llzkFeltConstAttrGetFromPartsWithField) {
  auto fieldName = MlirStringRef {.data = "bn254", .length = 5};
  constexpr auto BITS = 254;
  const uint64_t parts[] = {10, 20, 30, 40};
  auto attr = llzkFeltConstAttrGetFromPartsWithField(context, BITS, parts, 4, fieldName);
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto expected = llzk::felt::FeltConstAttr::get(
      unwrap(context), llvm::APInt(BITS, llvm::ArrayRef(parts, 4)),
      mlir::StringAttr::get(unwrap(context), "bn254")
  );
  EXPECT_EQ(unwrap(attr), expected);
}

TEST_F(CAPITest, llzk_attribute_is_a_felt_const_attr_pass) {
  auto attr = llzkFeltConstAttrGet(context, 0);
  EXPECT_TRUE(llzkAttributeIsAFeltConstAttr(attr));
}

TEST_F(CAPITest, llzk_attribute_is_a_felt_const_attr_fail) {
  auto attr = mlirIntegerAttrGet(mlirIndexTypeGet(context), 0);
  EXPECT_TRUE(!llzkAttributeIsAFeltConstAttr(attr));
}

TEST_F(CAPITest, llzk_felt_type_get) {
  auto type = llzkFeltTypeGet(context);
  EXPECT_NE(type.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_felt_type_get_with_field) {
  auto str = MlirStringRef {.data = "bn128", .length = 5};
  auto type = llzkFeltTypeGetWithField(context, str);
  EXPECT_NE(type.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_type_is_a_felt_type_pass) {
  auto type = llzkFeltTypeGet(context);
  EXPECT_TRUE(llzkTypeIsAFeltType(type));
}

TEST_F(CAPITest, llzk_type_is_a_felt_type_fail) {
  auto type = mlirIndexTypeGet(context);
  EXPECT_TRUE(!llzkTypeIsAFeltType(type));
}
