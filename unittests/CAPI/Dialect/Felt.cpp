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

// Include the auto-generated tests
#include "llzk/Dialect/Felt/IR/Attrs.capi.test.cpp.inc"
#include "llzk/Dialect/Felt/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Felt/IR/Ops.capi.test.cpp.inc"
#include "llzk/Dialect/Felt/IR/Types.capi.test.cpp.inc"

TEST_F(CAPITest, llzk_felt_const_attr_get) {
  auto attr = llzkFelt_FeltConstAttrGetUnspecified(context, 0);
  EXPECT_NE(attr.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_felt_const_attr_get_with_field) {
  auto fieldName = MlirStringRef {.data = "goldilocks", .length = 10};
  auto attr = llzkFelt_FeltConstAttrGet(context, 0, mlirIdentifierGet(context, fieldName));
  EXPECT_NE(attr.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_felt_const_attr_get_with_bits_unspecified) {
  constexpr auto BITS = 128;
  auto attr = llzkFelt_FeltConstAttrGetWithBitsUnspecified(context, BITS, 0);
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto cxx_attr = llvm::dyn_cast<llzk::felt::FeltConstAttr>(unwrap(attr));
  EXPECT_TRUE(cxx_attr);
  EXPECT_EQ(cxx_attr.getFieldName(), nullptr);
  auto value = cxx_attr.getValue();
  EXPECT_EQ(value.getBitWidth(), BITS);
  EXPECT_EQ(value.getZExtValue(), 0);
}

TEST_F(CAPITest, llzk_felt_const_attr_get_with_bits) {
  constexpr auto BITS = 128;
  auto fieldName = MlirStringRef {.data = "babybear", .length = 8};
  auto attr =
      llzkFelt_FeltConstAttrGetWithBits(context, BITS, 0, mlirIdentifierGet(context, fieldName));
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto cxx_attr = llvm::dyn_cast<llzk::felt::FeltConstAttr>(unwrap(attr));
  EXPECT_TRUE(cxx_attr);
  EXPECT_EQ(cxx_attr.getFieldName().getValue(), fieldName.data);
  auto value = cxx_attr.getValue();
  EXPECT_EQ(value.getBitWidth(), BITS);
  EXPECT_EQ(value.getZExtValue(), 0);
}

TEST_F(CAPITest, llzk_felt_const_attr_get_from_string_unspecified) {
  constexpr auto BITS = 64;
  auto str = MlirStringRef {.data = "123", .length = 3};
  auto attr = llzkFelt_FeltConstAttrGetFromStringUnspecified(context, BITS, str);
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto expected = llzk::felt::FeltConstAttr::get(
      unwrap(context), llvm::APInt(BITS, llvm::StringRef("123", 3), 10)
  );
  EXPECT_EQ(unwrap(attr), expected);
}

TEST_F(CAPITest, llzk_felt_const_attr_get_from_string) {
  constexpr auto BITS = 64;
  auto fieldName = MlirStringRef {.data = "bn254", .length = 5};
  auto str = MlirStringRef {.data = "123", .length = 3};
  auto attr = llzkFelt_FeltConstAttrGetFromString(
      context, BITS, str, mlirIdentifierGet(context, fieldName)
  );
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto expected = llzk::felt::FeltConstAttr::get(
      unwrap(context), llvm::APInt(BITS, llvm::StringRef("123", 3), 10),
      mlir::StringAttr::get(unwrap(context), "bn254")
  );
  EXPECT_EQ(unwrap(attr), expected);
}

TEST_F(CAPITest, llzk_felt_const_attr_get_from_parts_unspecified) {
  constexpr auto BITS = 254;
  const uint64_t parts[] = {10, 20, 30, 40};
  auto attr = llzkFelt_FeltConstAttrGetFromPartsUnspecified(context, BITS, parts, 4);
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto expected =
      llzk::felt::FeltConstAttr::get(unwrap(context), llvm::APInt(BITS, llvm::ArrayRef(parts, 4)));
  EXPECT_EQ(unwrap(attr), expected);
}

TEST_F(CAPITest, llzk_felt_const_attr_get_from_parts) {
  constexpr auto BITS = 254;
  auto fieldName = MlirStringRef {.data = "bn254", .length = 5};
  const uint64_t parts[] = {10, 20, 30, 40};
  auto attr = llzkFelt_FeltConstAttrGetFromParts(
      context, BITS, parts, 4, mlirIdentifierGet(context, fieldName)
  );
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto expected = llzk::felt::FeltConstAttr::get(
      unwrap(context), llvm::APInt(BITS, llvm::ArrayRef(parts, 4)),
      mlir::StringAttr::get(unwrap(context), "bn254")
  );
  EXPECT_EQ(unwrap(attr), expected);
}

TEST_F(CAPITest, llzk_attribute_is_a_felt_const_attr_pass) {
  auto attr = llzkFelt_FeltConstAttrGetUnspecified(context, 0);
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
      // Use "@compute" function as parent to avoid the following:
      // error: 'felt.bit_or' op only valid within a 'function.def' with 'function.allow_witness'
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
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
      // Use "@compute" function as parent to avoid the following:
      // error: 'felt.bit_and' op only valid within a 'function.def' with 'function.allow_witness'
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
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
      // Use "@compute" function as parent to avoid the following:
      // error: 'felt.bit_xor' op only valid within a 'function.def' with 'function.allow_witness'
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
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
      // Use "@compute" function as parent to avoid the following:
      // error: 'felt.bit_not' op only valid within a 'function.def' with 'function.allow_witness'
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
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
      // Use "@compute" function as parent to avoid the following:
      // error: 'felt.shl' op only valid within a 'function.def' with 'function.allow_witness'
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
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
      // Use "@compute" function as parent to avoid the following:
      // error: 'felt.shr' op only valid within a 'function.def' with 'function.allow_witness'
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
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
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
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
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
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
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
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
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
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
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
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
      // Use "@compute" function as parent to avoid the following:
      // error: 'felt.inv' op only valid within a 'function.def' with 'function.allow_witness'
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      auto val = testClass.cppGenFeltConstant(builder, location);
      auto resultType = wrap(testClass.cppGetFeltType(builder));
      return llzkFelt_InvFeltOpBuild(builder, location, resultType, wrap(val));
    }
  };
  return std::make_unique<Impl>();
}
