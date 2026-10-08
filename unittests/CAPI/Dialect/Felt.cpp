//===-- Felt.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Felt.h"

#include "../CAPITestBase.h"

#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Util/DynamicAPIntHelper.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/Support.h>

#include <mlir/CAPI/Support.h>

#include <llvm/ADT/APInt.h>

// Include the auto-generated tests
#include "llzk/Dialect/Felt/IR/Attrs.capi.test.cpp.inc"
#include "llzk/Dialect/Felt/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Felt/IR/Ops.capi.test.cpp.inc"
#include "llzk/Dialect/Felt/IR/Types.capi.test.cpp.inc"

TEST_F(CAPITest, llzk_felt_const_attr_get) {
  auto attr = llzkFelt_FeltConstAttrGet(
      context, mlirStringRefCreateFromCString("9223372036854775807"), wrap(cppGetFeltType("bn254"))
  );
  EXPECT_NE(attr.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_felt_const_attr_get_in_field) {
  auto fieldName = MlirStringRef {.data = "goldilocks", .length = 10};
  auto attr = llzkFelt_FeltConstAttrGetFromInt64InField(context, 0, fieldName);
  EXPECT_NE(attr.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_felt_const_attr_get_unspecified) {
  auto attr = llzkFelt_FeltConstAttrGetFromInt64Unspecified(context, 0);
  EXPECT_NE(attr.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_felt_const_attr_get_from_int64) {
  auto ty = cppGetFeltType("mersenne31");
  auto attr = llzkFelt_FeltConstAttrGetFromInt64(context, 2147483647, wrap(ty));
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto cxx_attr = llvm::dyn_cast<llzk::felt::FeltConstAttr>(unwrap(attr));
  ASSERT_TRUE(cxx_attr);
  EXPECT_EQ(cxx_attr.getFieldName(), ty.getFieldName());
  const auto &value = cxx_attr.getValue();
  EXPECT_EQ(value, 2147483647);
}

TEST_F(CAPITest, llzk_felt_const_attr_get_from_int64_in_field) {
  auto fieldName = MlirStringRef {.data = "babybear", .length = 8};
  auto attr = llzkFelt_FeltConstAttrGetFromInt64InField(context, 0, fieldName);
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto cxx_attr = llvm::dyn_cast<llzk::felt::FeltConstAttr>(unwrap(attr));
  ASSERT_TRUE(cxx_attr);
  EXPECT_EQ(cxx_attr.getFieldName().getValue(), fieldName.data);
  const auto &value = cxx_attr.getValue();
  EXPECT_EQ(value, 0);
}

TEST_F(CAPITest, llzk_felt_const_attr_get_from_int64_unspecified) {
  auto attr = llzkFelt_FeltConstAttrGetFromInt64Unspecified(context, 0);
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto cxx_attr = llvm::dyn_cast<llzk::felt::FeltConstAttr>(unwrap(attr));
  ASSERT_TRUE(cxx_attr);
  EXPECT_EQ(cxx_attr.getFieldName(), nullptr);
  const auto &value = cxx_attr.getValue();
  EXPECT_EQ(value, 0);
}

TEST_F(CAPITest, llzk_felt_const_attr_get_from_string) {
  auto ty = cppGetFeltType("bn254");
  auto str = MlirStringRef {.data = "123", .length = 3};
  auto attr = llzkFelt_FeltConstAttrGetFromString(context, str, wrap(ty));
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto expected = llzk::felt::FeltConstAttr::get(
      unwrap(context), llvm::DynamicAPInt(123), mlir::StringAttr::get(unwrap(context), "bn254")
  );
  EXPECT_EQ(unwrap(attr), expected);
}

TEST_F(CAPITest, llzk_felt_const_attr_get_from_string_in_field) {
  auto fieldName = MlirStringRef {.data = "bn254", .length = 5};
  auto str = MlirStringRef {.data = "123", .length = 3};
  auto attr = llzkFelt_FeltConstAttrGetFromStringInField(context, str, fieldName);
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto expected = llzk::felt::FeltConstAttr::get(
      unwrap(context), llvm::DynamicAPInt(123), mlir::StringAttr::get(unwrap(context), "bn254")
  );
  EXPECT_EQ(unwrap(attr), expected);
}

TEST_F(CAPITest, llzk_felt_const_attr_get_from_string_unspecified) {
  auto str = MlirStringRef {.data = "123", .length = 3};
  auto attr = llzkFelt_FeltConstAttrGetFromStringUnspecified(context, str);
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto expected = llzk::felt::FeltConstAttr::get(unwrap(context), llvm::DynamicAPInt(123));
  EXPECT_EQ(unwrap(attr), expected);
}

TEST_F(CAPITest, llzk_felt_const_attr_get_from_parts) {
  auto ty = cppGetFeltType("bn254");
  const uint64_t parts[] = {10, 20, 30, 40};
  auto attr = llzkFelt_FeltConstAttrGetFromParts(context, parts, 4, wrap(ty));
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto expected = llzk::felt::FeltConstAttr::get(
      unwrap(context), llzk::toDynamicAPInt(llvm::APInt(256, llvm::ArrayRef(parts, 4))),
      mlir::StringAttr::get(unwrap(context), "bn254")
  );
  EXPECT_EQ(unwrap(attr), expected);
}

TEST_F(CAPITest, llzk_felt_const_attr_get_from_parts_in_field) {
  auto fieldName = MlirStringRef {.data = "bn254", .length = 5};
  const uint64_t parts[] = {10, 20, 30, 40};
  auto attr = llzkFelt_FeltConstAttrGetFromPartsInField(context, parts, 4, fieldName);
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto expected = llzk::felt::FeltConstAttr::get(
      unwrap(context), llzk::toDynamicAPInt(llvm::APInt(256, llvm::ArrayRef(parts, 4))),
      mlir::StringAttr::get(unwrap(context), "bn254")
  );
  EXPECT_EQ(unwrap(attr), expected);
}

TEST_F(CAPITest, llzk_felt_const_attr_get_from_parts_unspecified) {
  const uint64_t parts[] = {10, 20, 30, 40};
  auto attr = llzkFelt_FeltConstAttrGetFromPartsUnspecified(context, parts, 4);
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto expected = llzk::felt::FeltConstAttr::get(
      unwrap(context), llzk::toDynamicAPInt(llvm::APInt(256, llvm::ArrayRef(parts, 4)))
  );
  EXPECT_EQ(unwrap(attr), expected);
}

TEST_F(CAPITest, FeltConstructorsPreserveAllBits) {
  auto type = wrap(cppGetFeltType("bn254"));
  auto field = mlirStringRefCreateFromCString("bn254");
  auto text = mlirStringRefCreateFromCString("340282366920938463463374607431768211455");
  const uint64_t parts[] = {UINT64_MAX, UINT64_MAX};
  auto expected = llzk::toDynamicAPInt(unwrap(text));
  for (auto attr : {
           llzkFelt_FeltConstAttrGetFromString(context, text, type),
           llzkFelt_FeltConstAttrGetFromStringInField(context, text, field),
           llzkFelt_FeltConstAttrGetFromStringUnspecified(context, text),
           llzkFelt_FeltConstAttrGetFromParts(context, parts, 2, type),
           llzkFelt_FeltConstAttrGetFromPartsInField(context, parts, 2, field),
           llzkFelt_FeltConstAttrGetFromPartsUnspecified(context, parts, 2),
       }) {
    ASSERT_NE(attr.ptr, nullptr);
    EXPECT_EQ(llvm::cast<llzk::felt::FeltConstAttr>(unwrap(attr)).getValue(), expected);
  }
}

TEST_F(CAPITest, FeltConstructorsPreserveSignedValues) {
  auto type = wrap(cppGetFeltType("bn254"));
  auto field = mlirStringRefCreateFromCString("bn254");
  for (int64_t value : {INT64_MIN, int64_t(-1), int64_t(0), INT64_MAX}) {
    for (auto attr : {
             llzkFelt_FeltConstAttrGetFromInt64(context, value, type),
             llzkFelt_FeltConstAttrGetFromInt64InField(context, value, field),
             llzkFelt_FeltConstAttrGetFromInt64Unspecified(context, value),
         }) {
      ASSERT_NE(attr.ptr, nullptr);
      EXPECT_EQ(llvm::cast<llzk::felt::FeltConstAttr>(unwrap(attr)).getValue(), value);
    }
  }
  auto text = mlirStringRefCreateFromCString("-18446744073709551617");
  for (auto attr : {
           llzkFelt_FeltConstAttrGetFromString(context, text, type),
           llzkFelt_FeltConstAttrGetFromStringInField(context, text, field),
           llzkFelt_FeltConstAttrGetFromStringUnspecified(context, text),
       }) {
    ASSERT_NE(attr.ptr, nullptr);
    EXPECT_EQ(
        llvm::cast<llzk::felt::FeltConstAttr>(unwrap(attr)).getValue(),
        llzk::toDynamicAPInt(unwrap(text))
    );
  }
}

TEST_F(CAPITest, FeltStringConstructorsRejectMalformedInput) {
  auto type = wrap(cppGetFeltType("bn254"));
  auto field = mlirStringRefCreateFromCString("bn254");
  auto name = mlirIdentifierGet(context, field);
  for (auto input : {"", "12x", "--1"}) {
    auto text = mlirStringRefCreateFromCString(input);
    EXPECT_EQ(llzkFelt_FeltConstAttrGetFromString(context, text, type).ptr, nullptr);
    EXPECT_EQ(llzkFelt_FeltConstAttrGetFromStringInField(context, text, field).ptr, nullptr);
    EXPECT_EQ(llzkFelt_FeltConstAttrGetFromStringUnspecified(context, text).ptr, nullptr);
    EXPECT_EQ(llzkFelt_FieldSpecAttrGetFromString(context, name, text).ptr, nullptr);
  }
}

TEST_F(CAPITest, FeltPartsConstructorsAcceptEmptyAndLeadingZeroParts) {
  auto type = wrap(cppGetFeltType("bn254"));
  auto field = mlirStringRefCreateFromCString("bn254");
  for (auto attr : {
           llzkFelt_FeltConstAttrGetFromParts(context, nullptr, 0, type),
           llzkFelt_FeltConstAttrGetFromPartsInField(context, nullptr, 0, field),
           llzkFelt_FeltConstAttrGetFromPartsUnspecified(context, nullptr, 0),
       }) {
    ASSERT_NE(attr.ptr, nullptr);
    EXPECT_EQ(llvm::cast<llzk::felt::FeltConstAttr>(unwrap(attr)).getValue(), 0);
  }
  const uint64_t parts[] = {UINT64_MAX, 0};
  auto attr = llzkFelt_FeltConstAttrGetFromPartsUnspecified(context, parts, 2);
  EXPECT_EQ(
      llvm::cast<llzk::felt::FeltConstAttr>(unwrap(attr)).getValue(),
      llzk::toDynamicAPInt("18446744073709551615")
  );
}

TEST_F(CAPITest, FieldSpecPartsConstructorRejectsModuliBelowTwo) {
  auto name = mlirIdentifierGet(context, mlirStringRefCreateFromCString("custom"));
  EXPECT_EQ(llzkFelt_FieldSpecAttrGetFromParts(context, name, nullptr, 0).ptr, nullptr);
  for (uint64_t value : {0, 1}) {
    const uint64_t parts[] = {value, 0};
    for (intptr_t count : {1, 2}) {
      EXPECT_EQ(llzkFelt_FieldSpecAttrGetFromParts(context, name, parts, count).ptr, nullptr);
    }
  }
  const uint64_t parts[] = {2, 0};
  auto attr = llzkFelt_FieldSpecAttrGetFromParts(context, name, parts, 2);
  ASSERT_NE(attr.ptr, nullptr);
  EXPECT_EQ(llvm::cast<llzk::felt::FieldSpecAttr>(unwrap(attr)).getPrime(), 2);
}

TEST_F(CAPITest, FieldSpecConstructors) {
  auto name = mlirIdentifierGet(context, mlirStringRefCreateFromCString("custom"));
  // A 127-bit prime exercises multiple limbs.
  auto text = mlirStringRefCreateFromCString("170141183460469231731687303715884105727");
  const uint64_t parts[] = {UINT64_MAX, UINT64_MAX >> 1};
  auto expected = llzk::toDynamicAPInt(unwrap(text));
  for (auto attr : {
           llzkFelt_FieldSpecAttrGetFromString(context, name, text),
           llzkFelt_FieldSpecAttrGetFromParts(context, name, parts, 2),
       }) {
    ASSERT_NE(attr.ptr, nullptr);
    auto spec = llvm::cast<llzk::felt::FieldSpecAttr>(unwrap(attr));
    EXPECT_EQ(spec.getFieldName(), unwrap(name));
    EXPECT_EQ(spec.getPrime(), expected);
  }
}

TEST_F(CAPITest, FeltIntegerRoundTrip) {
  auto text = mlirStringRefCreateFromCString("-18446744073709551617");
  auto attr = llzkFelt_FeltConstAttrGet(context, text, wrap(cppGetFeltType("bn254")));
  ASSERT_NE(attr.ptr, nullptr);
  std::string result;
  llzkFelt_FeltConstAttrGetValue(attr, [](MlirStringRef part, void *out) {
    static_cast<std::string *>(out)->append(part.data, part.length);
  }, &result);
  EXPECT_EQ(result, "-18446744073709551617");
}

TEST_F(CAPITest, FeltIntegerRejectsMalformedInput) {
  auto attr = llzkFelt_FeltConstAttrGet(
      context, mlirStringRefCreateFromCString("12x"), wrap(cppGetFeltType("bn254"))
  );
  EXPECT_EQ(attr.ptr, nullptr);
}

TEST_F(CAPITest, llzk_attribute_is_a_felt_const_attr_pass) {
  auto attr = llzkFelt_FeltConstAttrGetFromInt64Unspecified(context, 0);
  EXPECT_TRUE(llzkAttributeIsA_Felt_FeltConstAttr(attr));
}

TEST_F(CAPITest, llzk_felt_type_get) {
  auto type = llzkFelt_FeltTypeGetUnspecified(context);
  EXPECT_NE(type.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_felt_type_get_with_field) {
  auto fieldName = MlirStringRef {.data = "bn128", .length = 5};
  auto type = llzkFelt_FeltTypeGet(context, mlirIdentifierGet(context, fieldName));
  EXPECT_NE(type.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_felt_type_get_with_field_ref) {
  auto fieldName = MlirStringRef {.data = "bn128", .length = 5};
  auto type = llzkFelt_FeltTypeGetFromRef(context, fieldName);
  EXPECT_NE(type.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_type_is_a_felt_type_pass) {
  auto type = llzkFelt_FeltTypeGetUnspecified(context);
  EXPECT_TRUE(llzkTypeIsA_Felt_FeltType(type));
}

// Implementation for `FeltConstantOp_build_pass` test
std::unique_ptr<FeltConstantOpBuildFuncHelper> FeltConstantOpBuildFuncHelper::get() {
  struct Impl : public FeltConstantOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Use C++ API to avoid indirectly testing other LLZK C API functions here.
      auto attr = llzk::felt::FeltConstAttr::get(unwrap(testClass.context), llvm::APInt());
      auto resultType = wrap(testClass.cppGetFeltType(builder));
      return llzkFelt_FeltConstantOpBuild(builder, location, resultType, wrap(attr));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `OrFeltOp_build_pass` test
std::unique_ptr<OrFeltOpBuildFuncHelper> OrFeltOpBuildFuncHelper::get() {
  struct Impl : public OrFeltOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      testClass.setAllowNonNativeFieldOpsAttrOnFuncDef(builder);
      auto val = testClass.cppGenFeltConstant(builder, location);
      auto resultType = wrap(testClass.cppGetFeltType(builder));
      return llzkFelt_OrFeltOpBuild(builder, location, resultType, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `AndFeltOp_build_pass` test
std::unique_ptr<AndFeltOpBuildFuncHelper> AndFeltOpBuildFuncHelper::get() {
  struct Impl : public AndFeltOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      testClass.setAllowNonNativeFieldOpsAttrOnFuncDef(builder);
      auto val = testClass.cppGenFeltConstant(builder, location);
      auto resultType = wrap(testClass.cppGetFeltType(builder));
      return llzkFelt_AndFeltOpBuild(builder, location, resultType, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `XorFeltOp_build_pass` test
std::unique_ptr<XorFeltOpBuildFuncHelper> XorFeltOpBuildFuncHelper::get() {
  struct Impl : public XorFeltOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      testClass.setAllowNonNativeFieldOpsAttrOnFuncDef(builder);
      auto val = testClass.cppGenFeltConstant(builder, location);
      auto resultType = wrap(testClass.cppGetFeltType(builder));
      return llzkFelt_XorFeltOpBuild(builder, location, resultType, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `NotFeltOp_build_pass` test
std::unique_ptr<NotFeltOpBuildFuncHelper> NotFeltOpBuildFuncHelper::get() {
  struct Impl : public NotFeltOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      testClass.setAllowNonNativeFieldOpsAttrOnFuncDef(builder);
      auto val = testClass.cppGenFeltConstant(builder, location);
      auto resultType = wrap(testClass.cppGetFeltType(builder));
      return llzkFelt_NotFeltOpBuild(builder, location, resultType, wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `ShlFeltOp_build_pass` test
std::unique_ptr<ShlFeltOpBuildFuncHelper> ShlFeltOpBuildFuncHelper::get() {
  struct Impl : public ShlFeltOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      testClass.setAllowNonNativeFieldOpsAttrOnFuncDef(builder);
      auto val = testClass.cppGenFeltConstant(builder, location);
      auto resultType = wrap(testClass.cppGetFeltType(builder));
      return llzkFelt_ShlFeltOpBuild(builder, location, resultType, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `ShrFeltOp_build_pass` test
std::unique_ptr<ShrFeltOpBuildFuncHelper> ShrFeltOpBuildFuncHelper::get() {
  struct Impl : public ShrFeltOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      testClass.setAllowNonNativeFieldOpsAttrOnFuncDef(builder);
      auto val = testClass.cppGenFeltConstant(builder, location);
      auto resultType = wrap(testClass.cppGetFeltType(builder));
      return llzkFelt_ShrFeltOpBuild(builder, location, resultType, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `AddFeltOp_build_pass` test
std::unique_ptr<AddFeltOpBuildFuncHelper> AddFeltOpBuildFuncHelper::get() {
  struct Impl : public AddFeltOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      auto val = testClass.cppGenFeltConstant(builder, location);
      auto resultType = wrap(testClass.cppGetFeltType(builder));
      return llzkFelt_AddFeltOpBuild(builder, location, resultType, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `SubFeltOp_build_pass` test
std::unique_ptr<SubFeltOpBuildFuncHelper> SubFeltOpBuildFuncHelper::get() {
  struct Impl : public SubFeltOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      auto val = testClass.cppGenFeltConstant(builder, location);
      auto resultType = wrap(testClass.cppGetFeltType(builder));
      return llzkFelt_SubFeltOpBuild(builder, location, resultType, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `MulFeltOp_build_pass` test
std::unique_ptr<MulFeltOpBuildFuncHelper> MulFeltOpBuildFuncHelper::get() {
  struct Impl : public MulFeltOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      auto val = testClass.cppGenFeltConstant(builder, location);
      auto resultType = wrap(testClass.cppGetFeltType(builder));
      return llzkFelt_MulFeltOpBuild(builder, location, resultType, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `PowFeltOp_build_pass` test
std::unique_ptr<PowFeltOpBuildFuncHelper> PowFeltOpBuildFuncHelper::get() {
  struct Impl : public PowFeltOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      testClass.setAllowNonNativeFieldOpsAttrOnFuncDef(builder);
      auto val = testClass.cppGenFeltConstant(builder, location);
      auto resultType = wrap(testClass.cppGetFeltType(builder));
      return llzkFelt_PowFeltOpBuild(builder, location, resultType, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `DivFeltOp_build_pass` test
std::unique_ptr<DivFeltOpBuildFuncHelper> DivFeltOpBuildFuncHelper::get() {
  struct Impl : public DivFeltOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      auto val = testClass.cppGenFeltConstant(builder, location);
      auto resultType = wrap(testClass.cppGetFeltType(builder));
      return llzkFelt_DivFeltOpBuild(builder, location, resultType, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `UnsignedIntDivFeltOp_build_pass` test
std::unique_ptr<UnsignedIntDivFeltOpBuildFuncHelper> UnsignedIntDivFeltOpBuildFuncHelper::get() {
  struct Impl : public UnsignedIntDivFeltOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      testClass.setAllowNonNativeFieldOpsAttrOnFuncDef(builder);
      auto val = testClass.cppGenFeltConstant(builder, location);
      auto resultType = wrap(testClass.cppGetFeltType(builder));
      return llzkFelt_UnsignedIntDivFeltOpBuild(
          builder, location, resultType, wrap(val), wrap(val)
      );
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `SignedIntDivFeltOp_build_pass` test
std::unique_ptr<SignedIntDivFeltOpBuildFuncHelper> SignedIntDivFeltOpBuildFuncHelper::get() {
  struct Impl : public SignedIntDivFeltOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      testClass.setAllowNonNativeFieldOpsAttrOnFuncDef(builder);
      auto val = testClass.cppGenFeltConstant(builder, location);
      auto resultType = wrap(testClass.cppGetFeltType(builder));
      return llzkFelt_SignedIntDivFeltOpBuild(builder, location, resultType, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `UnsignedModFeltOp_build_pass` test
std::unique_ptr<UnsignedModFeltOpBuildFuncHelper> UnsignedModFeltOpBuildFuncHelper::get() {
  struct Impl : public UnsignedModFeltOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      testClass.setAllowNonNativeFieldOpsAttrOnFuncDef(builder);
      auto val = testClass.cppGenFeltConstant(builder, location);
      auto resultType = wrap(testClass.cppGetFeltType(builder));
      return llzkFelt_UnsignedModFeltOpBuild(builder, location, resultType, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `SignedModFeltOp_build_pass` test
std::unique_ptr<SignedModFeltOpBuildFuncHelper> SignedModFeltOpBuildFuncHelper::get() {
  struct Impl : public SignedModFeltOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      testClass.setAllowNonNativeFieldOpsAttrOnFuncDef(builder);
      auto val = testClass.cppGenFeltConstant(builder, location);
      auto resultType = wrap(testClass.cppGetFeltType(builder));
      return llzkFelt_SignedModFeltOpBuild(builder, location, resultType, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `NegFeltOp_build_pass` test
std::unique_ptr<NegFeltOpBuildFuncHelper> NegFeltOpBuildFuncHelper::get() {
  struct Impl : public NegFeltOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      auto val = testClass.cppGenFeltConstant(builder, location);
      auto resultType = wrap(testClass.cppGetFeltType(builder));
      return llzkFelt_NegFeltOpBuild(builder, location, resultType, wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `InvFeltOp_build_pass` test
std::unique_ptr<InvFeltOpBuildFuncHelper> InvFeltOpBuildFuncHelper::get() {
  struct Impl : public InvFeltOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      testClass.setAllowNonNativeFieldOpsAttrOnFuncDef(builder);
      auto val = testClass.cppGenFeltConstant(builder, location);
      auto resultType = wrap(testClass.cppGetFeltType(builder));
      return llzkFelt_InvFeltOpBuild(builder, location, resultType, wrap(val));
    }
  };
  return std::make_unique<Impl>();
}
