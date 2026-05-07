//===-- CallOpTests.cpp - Unit tests for CallOps ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "OpTestBase.h"

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Shared/Builders.h"

#include <mlir/Dialect/Arith/IR/Arith.h>

using namespace mlir;
using namespace llzk;
using namespace llzk::component;
using namespace llzk::function;

//===------------------------------------------------------------------===//
// CallOp::build(..., TypeRange, SymbolRefAttr, ValueRange = {})
//===------------------------------------------------------------------===//

TEST_F(OpTests, testCallNoAffine_GoodNoArgs) {
  ModuleBuilder llzkBldr = newBasicFunctionsExample(0);

  auto funcA = llzkBldr.getFreeFunc(funcNameA);
  ASSERT_TRUE(succeeded(funcA));
  auto funcB = llzkBldr.getFreeFunc(funcNameB);
  ASSERT_TRUE(succeeded(funcB));

  OpBuilder bldr(funcA->getBody());
  CallOp op = bldr.create<CallOp>(loc, funcB->getResultTypes(), funcB->getFullyQualifiedName());
  // module attributes {llzk.lang} {
  //   function.def @FuncA() -> index {
  //     %0 = call @FuncB() : () -> index
  //   }
  //   function.def @FuncB() -> index {
  //   }
  // }
  ASSERT_TRUE(verify(mod.get()));
  ASSERT_TRUE(verify(op, true));
}

TEST_F(OpTests, testCallNoAffine_GoodWithArgs) {
  ModuleBuilder llzkBldr = newBasicFunctionsExample(2);

  auto funcA = llzkBldr.getFreeFunc(funcNameA);
  ASSERT_TRUE(succeeded(funcA));
  auto funcB = llzkBldr.getFreeFunc(funcNameB);
  ASSERT_TRUE(succeeded(funcB));

  OpBuilder bldr(funcA->getBody());
  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 5);
  auto v2 = bldr.create<arith::ConstantIndexOp>(loc, 2);
  CallOp op = bldr.create<CallOp>(
      loc, funcB->getResultTypes(), funcB->getFullyQualifiedName(), ValueRange {v1, v2}
  );
  // module attributes {llzk.lang} {
  //   function.def @FuncA(%arg0: index, %arg1: index) -> index {
  //     %idx5 = arith.constant 5 : index
  //     %idx2 = arith.constant 2 : index
  //     %0 = call @FuncB(%idx5, %idx2) : (index, index) -> index
  //   }
  //   function.def @FuncB(%arg0: index, %arg1: index) -> index {
  //   }
  // }
  ASSERT_TRUE(verify(mod.get()));
  ASSERT_TRUE(verify(op, true));
}

TEST_F(OpTests, testCallNoAffine_TooFewValues) {
  ModuleBuilder llzkBldr = newBasicFunctionsExample(2);

  auto funcA = llzkBldr.getFreeFunc(funcNameA);
  ASSERT_TRUE(succeeded(funcA));
  auto funcB = llzkBldr.getFreeFunc(funcNameB);
  ASSERT_TRUE(succeeded(funcB));

  OpBuilder bldr(funcA->getBody());
  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 5);
  CallOp op = bldr.create<CallOp>(
      loc, funcB->getResultTypes(), funcB->getFullyQualifiedName(), ValueRange {v1}
  );
  // module attributes {llzk.lang} {
  //   function.def @FuncA(%arg0: index, %arg1: index) -> index {
  //     %idx5 = arith.constant 5 : index
  //     %0 = call @FuncB(%idx5) : (index) -> index
  //   }
  //   function.def @FuncB(%arg0: index, %arg1: index) -> index {
  //   }
  // }
  EXPECT_DEATH(
      {
        verifyOrDie(mod.get());
        verifyOrDie(op, true);
      },
      "error: 'function.call' op incorrect number of operands for callee, expected 2"
  );
}

TEST_F(OpTests, testCallNoAffine_WrongRetTy) {
  ModuleBuilder llzkBldr = newBasicFunctionsExample(1);

  auto funcA = llzkBldr.getFreeFunc(funcNameA);
  ASSERT_TRUE(succeeded(funcA));
  auto funcB = llzkBldr.getFreeFunc(funcNameB);
  ASSERT_TRUE(succeeded(funcB));

  OpBuilder bldr(funcA->getBody());
  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 5);
  CallOp op = bldr.create<CallOp>(
      loc, TypeRange {bldr.getI1Type()}, funcB->getFullyQualifiedName(), ValueRange {v1}
  );
  // module attributes {llzk.lang} {
  //   function.def @FuncA(%arg0: index) -> index {
  //     %idx5 = arith.constant 5 : index
  //     %0 = call @FuncB(%idx5) : (index) -> i1
  //   }
  //   function.def @FuncB(%arg0: index) -> index {
  //   }
  // }
  EXPECT_DEATH(
      {
        verifyOrDie(mod.get());
        verifyOrDie(op, true);
      },
      "error: 'function.call' op result type mismatch: expected type 'index', but found 'i1' for "
      "result number 0"
  );
}

TEST_F(OpTests, testCallNoAffine_InvalidCalleeName) {
  ModuleBuilder llzkBldr = newBasicFunctionsExample(0);

  auto funcA = llzkBldr.getFreeFunc(funcNameA);
  ASSERT_TRUE(succeeded(funcA));

  OpBuilder bldr(funcA->getBody());
  CallOp op = bldr.create<CallOp>(loc, TypeRange {}, FlatSymbolRefAttr::get(&ctx, "invalidName"));
  // module attributes {llzk.lang} {
  //   function.def @FuncA() -> index {
  //     call @invalidName() : () -> ()
  //   }
  //   function.def @FuncB() -> index {
  //   }
  // }
  EXPECT_DEATH(
      {
        verifyOrDie(mod.get());
        verifyOrDie(op, true);
      },
      "error: 'function.call' op references unknown symbol \"@invalidName\""
  );
}

TEST_F(OpTests, testCallNoAffine_InvalidTemplateParam) {
  auto [modBldr, tmplBldr] = newTemplateFunctionsExample(1);

  auto funcA = tmplBldr->getFreeFunc(funcNameA);
  ASSERT_TRUE(succeeded(funcA));
  auto funcB = tmplBldr->getFreeFunc(funcNameB);
  ASSERT_TRUE(succeeded(funcB));

  // create an 256-bit IntegerAttr with larger value than IndexType can hold
  APInt bigValue = APInt::getMaxValue(256);
  IntegerAttr a = IntegerAttr::get(IntegerType::get(&ctx, 256), bigValue);

  OpBuilder bldr(funcA->getBody());
  CallOp op = bldr.create<CallOp>(
      loc, TypeRange {bldr.getIndexType()}, funcB->getFullyQualifiedName(), ValueRange {},
      ArrayRef<Attribute> {a}
  );
  // module attributes {llzk.lang} {
  //   poly.template @ExampleTemplate {
  //     poly.param @T0
  //     function.def @FuncA() -> index {
  //       %0 = call @FuncB<[-1 : i256]>() : () -> index
  //     }
  //     function.def @FuncB() -> index {
  //     }
  //   }
  // }
  EXPECT_DEATH(
      {
        verifyOrDie(mod.get());
        verifyOrDie(op, true);
      },
      "'function.call' op value is too large for `index` type: -1"
  );
}

//===------------------------------------------------------------------===//
// CallOp::build(..., TypeRange, SymbolRefAttr, ArrayRef<ValueRange>,
//                    ArrayRef<int32_t>, ValueRange = {})
//===------------------------------------------------------------------===//

TEST_F(OpTests, testCallWithAffine_Good) {
  auto [modBldr, tmplBldr] = newTemplateStructExample(2);

  auto funcComputeA = tmplBldr->getComputeFn(structNameA);
  ASSERT_TRUE(succeeded(funcComputeA));
  auto funcComputeB = tmplBldr->getComputeFn(structNameB);
  ASSERT_TRUE(succeeded(funcComputeB));

  auto structB = tmplBldr->getStruct(structNameB);
  ASSERT_TRUE(succeeded(structB));

  OpBuilder bldr(funcComputeA->getBody());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  StructType affineStructType = StructType::get(
      structB->getFullyQualifiedName(), bldr.getArrayAttr({m, m})
  ); // !struct.type<@StructB<[affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>]>>

  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 2);
  auto v2 = bldr.create<arith::ConstantIndexOp>(loc, 4);
  CallOp op = bldr.create<CallOp>(
      loc, TypeRange {affineStructType}, funcComputeB->getFullyQualifiedName(),
      ArrayRef {ValueRange {v1}, ValueRange {v2}}, ArrayRef<int32_t> {1, 1}
  );
  ASSERT_TRUE(verify(mod.get()));
  ASSERT_TRUE(verify(op, true));
}

TEST_F(OpTests, testCallWithAffine_WrongStructNameInResultType) {
  auto [modBldr, tmplBldr] = newTemplateStructExample(2);

  auto funcComputeA = tmplBldr->getComputeFn(structNameA);
  ASSERT_TRUE(succeeded(funcComputeA));
  auto funcComputeB = tmplBldr->getComputeFn(structNameB);
  ASSERT_TRUE(succeeded(funcComputeB));

  auto structA = tmplBldr->getStruct(structNameA);
  ASSERT_TRUE(succeeded(structA));

  OpBuilder bldr(funcComputeA->getBody());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  StructType affineStructType = StructType::get(
      structA->getFullyQualifiedName(), bldr.getArrayAttr({m, m})
  ); // !struct.type<@StructA<[affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>]>>

  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 2);
  auto v2 = bldr.create<arith::ConstantIndexOp>(loc, 4);
  CallOp op = bldr.create<CallOp>(
      loc, TypeRange {affineStructType}, funcComputeB->getFullyQualifiedName(),
      ArrayRef {ValueRange {v1}, ValueRange {v2}}, ArrayRef<int32_t> {1, 1}
  );
  EXPECT_DEATH(
      {
        verifyOrDie(mod.get());
        verifyOrDie(op, true);
      },
      "error: 'function.call' op result type mismatch: expected type "
      "'!struct.type<@ExampleTemplate::@StructB<\\[@T0, @T1\\]>>', but found "
      "'!struct.type<@ExampleTemplate::@StructA<\\[affine_map<\\(d0\\) -> \\(d0\\)>, "
      "affine_map<\\(d0\\) -> \\(d0\\)>\\]>>' for result number 0"
  );
}

TEST_F(OpTests, testCallWithAffine_TooFewMapsInResultType) {
  auto [modBldr, tmplBldr] = newTemplateStructExample(2);

  auto funcComputeA = tmplBldr->getComputeFn(structNameA);
  ASSERT_TRUE(succeeded(funcComputeA));
  auto funcComputeB = tmplBldr->getComputeFn(structNameB);
  ASSERT_TRUE(succeeded(funcComputeB));

  auto structB = tmplBldr->getStruct(structNameB);
  ASSERT_TRUE(succeeded(structB));

  OpBuilder bldr(funcComputeA->getBody());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  StructType affineStructType = StructType::get(
      structB->getFullyQualifiedName(), bldr.getArrayAttr({m})
  ); // !struct.type<@StructB<[#m]>>

  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 2);
  auto v2 = bldr.create<arith::ConstantIndexOp>(loc, 4);
  CallOp op = bldr.create<CallOp>(
      loc, TypeRange {affineStructType}, funcComputeB->getFullyQualifiedName(),
      ArrayRef {ValueRange {v1}, ValueRange {v2}}, ArrayRef<int32_t> {1, 1}
  );
  EXPECT_DEATH(
      {
        verifyOrDie(mod.get());
        verifyOrDie(op, true);
      },
      "error: 'struct.type' type has 1 parameters but \"StructB\" expects 2"
  );
}

TEST_F(OpTests, testCallWithAffine_TooManyMapsInResultType) {
  auto [modBldr, tmplBldr] = newTemplateStructExample(2);

  auto funcComputeA = tmplBldr->getComputeFn(structNameA);
  ASSERT_TRUE(succeeded(funcComputeA));
  auto funcComputeB = tmplBldr->getComputeFn(structNameB);
  ASSERT_TRUE(succeeded(funcComputeB));

  auto structB = tmplBldr->getStruct(structNameB);
  ASSERT_TRUE(succeeded(structB));

  OpBuilder bldr(funcComputeA->getBody());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  StructType affineStructType = StructType::get(
      structB->getFullyQualifiedName(), bldr.getArrayAttr({m, m, m})
  ); // !struct.type<@StructB<[#m,#m,#m]>>

  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 2);
  auto v2 = bldr.create<arith::ConstantIndexOp>(loc, 4);
  CallOp op = bldr.create<CallOp>(
      loc, TypeRange {affineStructType}, funcComputeB->getFullyQualifiedName(),
      ArrayRef {ValueRange {v1}, ValueRange {v2}}, ArrayRef<int32_t> {1, 1}
  );
  EXPECT_DEATH(
      {
        verifyOrDie(mod.get());
        verifyOrDie(op, true);
      },
      "error: 'struct.type' type has 3 parameters but \"StructB\" expects 2"
  );
}

TEST_F(OpTests, testCallWithAffine_OpGroupCountLessThanDimSizeCount) {
  auto [modBldr, tmplBldr] = newTemplateStructExample(2);

  auto funcComputeA = tmplBldr->getComputeFn(structNameA);
  ASSERT_TRUE(succeeded(funcComputeA));
  auto funcComputeB = tmplBldr->getComputeFn(structNameB);
  ASSERT_TRUE(succeeded(funcComputeB));

  auto structB = tmplBldr->getStruct(structNameB);
  ASSERT_TRUE(succeeded(structB));

  OpBuilder bldr(funcComputeA->getBody());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  StructType affineStructType = StructType::get(
      structB->getFullyQualifiedName(), bldr.getArrayAttr({m, m})
  ); // !struct.type<@StructB<[affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>]>>

  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 2);
  CallOp op = bldr.create<CallOp>(
      loc, TypeRange {affineStructType}, funcComputeB->getFullyQualifiedName(),
      ArrayRef {ValueRange {v1}}, ArrayRef<int32_t> {1, 1}
  );
  EXPECT_DEATH(
      {
        verifyOrDie(mod.get());
        verifyOrDie(op, true);
      },
      "error: 'function.call' op length of 'numDimsPerMap' attribute \\(2\\) does not match with "
      "length of 'mapOpGroupSizes' attribute \\(1\\)"
  );
}

TEST_F(OpTests, testCallWithAffine_OpGroupCountMoreThanDimSizeCount) {
  auto [modBldr, tmplBldr] = newTemplateStructExample(2);

  auto funcComputeA = tmplBldr->getComputeFn(structNameA);
  ASSERT_TRUE(succeeded(funcComputeA));
  auto funcComputeB = tmplBldr->getComputeFn(structNameB);
  ASSERT_TRUE(succeeded(funcComputeB));

  auto structB = tmplBldr->getStruct(structNameB);
  ASSERT_TRUE(succeeded(structB));

  OpBuilder bldr(funcComputeA->getBody());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  StructType affineStructType = StructType::get(
      structB->getFullyQualifiedName(), bldr.getArrayAttr({m, m})
  ); // !struct.type<@StructB<[affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>]>>

  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 2);
  CallOp op = bldr.create<CallOp>(
      loc, TypeRange {affineStructType}, funcComputeB->getFullyQualifiedName(),
      ArrayRef {ValueRange {v1}, ValueRange {v1}, ValueRange {v1}}, ArrayRef<int32_t> {1, 1}
  );
  EXPECT_DEATH(
      {
        verifyOrDie(mod.get());
        verifyOrDie(op, true);
      },
      "error: 'function.call' op length of 'numDimsPerMap' attribute \\(2\\) does not match with "
      "length of 'mapOpGroupSizes' attribute \\(3\\)"
  );
}

TEST_F(OpTests, testCallWithAffine_OpGroupCount0) {
  auto [modBldr, tmplBldr] = newTemplateStructExample(2);

  auto funcComputeA = tmplBldr->getComputeFn(structNameA);
  ASSERT_TRUE(succeeded(funcComputeA));
  auto funcComputeB = tmplBldr->getComputeFn(structNameB);
  ASSERT_TRUE(succeeded(funcComputeB));

  auto structB = tmplBldr->getStruct(structNameB);
  ASSERT_TRUE(succeeded(structB));

  OpBuilder bldr(funcComputeA->getBody());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  StructType affineStructType = StructType::get(
      structB->getFullyQualifiedName(), bldr.getArrayAttr({m, m})
  ); // !struct.type<@StructB<[affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>]>>

  CallOp op = bldr.create<CallOp>(
      loc, TypeRange {affineStructType}, funcComputeB->getFullyQualifiedName(),
      ArrayRef<ValueRange> {}, ArrayRef<int32_t> {1, 1}
  );
  EXPECT_DEATH(
      {
        verifyOrDie(mod.get());
        verifyOrDie(op, true);
      },
      "error: 'function.call' op length of 'numDimsPerMap' attribute \\(2\\) does not match with "
      "length of 'mapOpGroupSizes' attribute \\(0\\)"
  );
}

TEST_F(OpTests, testCallWithAffine_DimSizeCount0) {
  auto [modBldr, tmplBldr] = newTemplateStructExample(2);

  auto funcComputeA = tmplBldr->getComputeFn(structNameA);
  ASSERT_TRUE(succeeded(funcComputeA));
  auto funcComputeB = tmplBldr->getComputeFn(structNameB);
  ASSERT_TRUE(succeeded(funcComputeB));

  auto structB = tmplBldr->getStruct(structNameB);
  ASSERT_TRUE(succeeded(structB));

  OpBuilder bldr(funcComputeA->getBody());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  StructType affineStructType = StructType::get(
      structB->getFullyQualifiedName(), bldr.getArrayAttr({m, m})
  ); // !struct.type<@StructB<[affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>]>>

  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 2);
  CallOp op = bldr.create<CallOp>(
      loc, TypeRange {affineStructType}, funcComputeB->getFullyQualifiedName(),
      ArrayRef {ValueRange {v1}, ValueRange {v1}}, ArrayRef<int32_t> {}
  );
  EXPECT_DEATH(
      {
        verifyOrDie(mod.get());
        verifyOrDie(op, true);
      },
      "error: 'function.call' op length of 'numDimsPerMap' attribute \\(0\\) does not match with "
      "length of 'mapOpGroupSizes' attribute \\(2\\)"
  );
}

TEST_F(OpTests, testCallWithAffine_OpGroupCount0DimSizeCount0) {
  auto [modBldr, tmplBldr] = newTemplateStructExample(2);

  auto funcComputeA = tmplBldr->getComputeFn(structNameA);
  ASSERT_TRUE(succeeded(funcComputeA));
  auto funcComputeB = tmplBldr->getComputeFn(structNameB);
  ASSERT_TRUE(succeeded(funcComputeB));

  auto structB = tmplBldr->getStruct(structNameB);
  ASSERT_TRUE(succeeded(structB));

  OpBuilder bldr(funcComputeA->getBody());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  StructType affineStructType = StructType::get(
      structB->getFullyQualifiedName(), bldr.getArrayAttr({m, m})
  ); // !struct.type<@StructB<[affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>]>>

  CallOp op = bldr.create<CallOp>(
      loc, TypeRange {affineStructType}, funcComputeB->getFullyQualifiedName(),
      ArrayRef<ValueRange> {}, ArrayRef<int32_t> {}
  );
  EXPECT_DEATH(
      {
        verifyOrDie(mod.get());
        verifyOrDie(op, true);
      },
      "error: 'function.call' op map instantiation group count \\(0\\) does not match the number "
      "of affine map instantiations \\(2\\) required by the type"
  );
}

TEST_F(OpTests, testCallWithAffine_OpGroupSizeLessThanDimSize) {
  auto [modBldr, tmplBldr] = newTemplateStructExample(2);

  auto funcComputeA = tmplBldr->getComputeFn(structNameA);
  ASSERT_TRUE(succeeded(funcComputeA));
  auto funcComputeB = tmplBldr->getComputeFn(structNameB);
  ASSERT_TRUE(succeeded(funcComputeB));

  auto structB = tmplBldr->getStruct(structNameB);
  ASSERT_TRUE(succeeded(structB));

  OpBuilder bldr(funcComputeA->getBody());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  StructType affineStructType = StructType::get(
      structB->getFullyQualifiedName(), bldr.getArrayAttr({m, m})
  ); // !struct.type<@StructB<[#m,#m]>>

  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 2);
  CallOp op = bldr.create<CallOp>(
      loc, TypeRange {affineStructType}, funcComputeB->getFullyQualifiedName(),
      ArrayRef {ValueRange {v1}, ValueRange {}}, ArrayRef<int32_t> {1, 1}
  );
  EXPECT_DEATH(
      {
        verifyOrDie(mod.get());
        verifyOrDie(op, true);
      },
      "error: 'function.call' op map instantiation group 1 dimension count \\(1\\) exceeds group 1 "
      "size in 'mapOpGroupSizes' attribute \\(0\\)"
  );
}

TEST_F(OpTests, testCallWithAffine_OpGroupSizeMoreThanDimSize) {
  auto [modBldr, tmplBldr] = newTemplateStructExample(2);

  auto funcComputeA = tmplBldr->getComputeFn(structNameA);
  ASSERT_TRUE(succeeded(funcComputeA));
  auto funcComputeB = tmplBldr->getComputeFn(structNameB);
  ASSERT_TRUE(succeeded(funcComputeB));

  auto structB = tmplBldr->getStruct(structNameB);
  ASSERT_TRUE(succeeded(structB));

  OpBuilder bldr(funcComputeA->getBody());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  StructType affineStructType = StructType::get(
      structB->getFullyQualifiedName(), bldr.getArrayAttr({m, m})
  ); // !struct.type<@StructB<[#m,#m]>>

  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 2);
  auto v2 = bldr.create<arith::ConstantIndexOp>(loc, 4);
  CallOp op = bldr.create<CallOp>(
      loc, TypeRange {affineStructType}, funcComputeB->getFullyQualifiedName(),
      ArrayRef {ValueRange {v1}, ValueRange {v1, v2}}, ArrayRef<int32_t> {1, 1}
  );
  EXPECT_DEATH(
      {
        verifyOrDie(mod.get());
        verifyOrDie(op, true);
      },
      "error: 'function.call' op instantiation of map 1 expected 0 but found 1 symbol values in "
      "\\[\\]"
  );
}

TEST_F(OpTests, testCallWithAffine_OpGroupCountAndDimSizeCountMoreThanType) {
  auto [modBldr, tmplBldr] = newTemplateStructExample(2);

  auto funcComputeA = tmplBldr->getComputeFn(structNameA);
  ASSERT_TRUE(succeeded(funcComputeA));
  auto funcComputeB = tmplBldr->getComputeFn(structNameB);
  ASSERT_TRUE(succeeded(funcComputeB));

  auto structB = tmplBldr->getStruct(structNameB);
  ASSERT_TRUE(succeeded(structB));

  OpBuilder bldr(funcComputeA->getBody());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  StructType affineStructType = StructType::get(
      structB->getFullyQualifiedName(), bldr.getArrayAttr({m, m})
  ); // !struct.type<@StructB<[affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>]>>

  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 2);
  auto v2 = bldr.create<arith::ConstantIndexOp>(loc, 4);
  CallOp op = bldr.create<CallOp>(
      loc, TypeRange {affineStructType}, funcComputeB->getFullyQualifiedName(),
      ArrayRef {ValueRange {v1}, ValueRange {v2}, ValueRange {v2}}, ArrayRef<int32_t> {1, 1, 1}
  );
  EXPECT_DEATH(
      {
        verifyOrDie(mod.get());
        verifyOrDie(op, true);
      },
      "error: 'function.call' op map instantiation group count \\(3\\) does not match the number "
      "of affine map instantiations \\(2\\) required by the type"
  );
}

//===------------------------------------------------------------------===//
// Other
//===------------------------------------------------------------------===//

TEST_F(OpTests, test_calleeIs_withStructCompute) {
  ModuleBuilder llzkBldr = newStructExample();
  llzkBldr.insertComputeCall(structNameA, structNameB);
  // module attributes {llzk.lang} {
  //   struct.def @StructB {
  //     function.def @constrain(%arg0: !struct.type<@StructB>) {
  //     }
  //     function.def @compute() -> !struct.type<@StructB> {
  //     }
  //   }
  //   struct.def @StructA {
  //     function.def @constrain(%arg0: !struct.type<@StructA>) {
  //     }
  //     function.def @compute() -> !struct.type<@StructA> {
  //       %0 = call @StructB::@compute() : () -> !struct.type<@StructB>
  //     }
  //   }
  // }

  auto funcComputeA = llzkBldr.getComputeFn(structNameA);
  ASSERT_TRUE(succeeded(funcComputeA));
  auto ops = funcComputeA->getBody().getOps();
  ASSERT_FALSE(ops.empty());
  CallOp call = llvm::dyn_cast_if_present<CallOp>(*ops.begin());
  ASSERT_FALSE(call == nullptr);

  ASSERT_TRUE(call.calleeIsCompute());
  ASSERT_FALSE(call.calleeIsConstrain());

  ASSERT_TRUE(call.calleeIsStructCompute());
  ASSERT_FALSE(call.calleeIsStructConstrain());
}

TEST_F(OpTests, test_calleeIs_withStructConstrain) {
  ModuleBuilder llzkBldr = newStructExample();
  llzkBldr.insertConstrainCall(structNameA, structNameB);
  // module attributes {llzk.lang} {
  //   struct.def @StructB {
  //     function.def @constrain(%arg0: !struct.type<@StructB>) {
  //     }
  //     function.def @compute() -> !struct.type<@StructB> {
  //     }
  //   }
  //   struct.def @StructA {
  //     member @StructB1 : !struct.type<@StructB>
  //     function.def @constrain(%arg0: !struct.type<@StructA>) {
  //       %0 = readm %arg0[@StructB1] : <@StructA>, !struct.type<@StructB>
  //       call @StructB::@constrain(%0) : (!struct.type<@StructB>) -> ()
  //     }
  //     function.def @compute() -> !struct.type<@StructA> {
  //     }
  //   }
  // }

  auto funcConstrainA = llzkBldr.getConstrainFn(structNameA);
  ASSERT_TRUE(succeeded(funcConstrainA));
  auto ops = funcConstrainA->getBody().getOps();
  ASSERT_FALSE(ops.empty());
  CallOp call = llvm::dyn_cast_if_present<CallOp>(*std::next(ops.begin()));
  ASSERT_FALSE(call == nullptr);

  ASSERT_FALSE(call.calleeIsCompute());
  ASSERT_TRUE(call.calleeIsConstrain());

  ASSERT_FALSE(call.calleeIsStructCompute());
  ASSERT_TRUE(call.calleeIsStructConstrain());
}

TEST_F(OpTests, test_calleeIs_withGlobalCompute) {
  ModuleBuilder llzkBldr = newBasicFunctionsExample(0, {"compute", "entry"});
  auto funcEntry = llzkBldr.getFreeFunc("entry");
  ASSERT_TRUE(succeeded(funcEntry));
  llzkBldr.insertFreeCall(*funcEntry, "compute");
  // module attributes {llzk.lang} {
  //   function.def @entry() -> index {
  //     %0 = call @compute() : () -> index
  //   }
  //   function.def @compute() -> index {
  //   }
  // }

  auto ops = funcEntry->getBody().getOps();
  ASSERT_FALSE(ops.empty());
  CallOp call = llvm::dyn_cast_if_present<CallOp>(*ops.begin());
  ASSERT_FALSE(call == nullptr);

  ASSERT_TRUE(call.calleeIsCompute());
  ASSERT_FALSE(call.calleeIsConstrain());

  ASSERT_FALSE(call.calleeIsStructCompute());
  ASSERT_FALSE(call.calleeIsStructConstrain());
}

TEST_F(OpTests, test_calleeIs_withGlobalConstrain) {
  ModuleBuilder llzkBldr = newBasicFunctionsExample(0, {"constrain", "entry"});
  auto funcEntry = llzkBldr.getFreeFunc("entry");
  ASSERT_TRUE(succeeded(funcEntry));
  llzkBldr.insertFreeCall(*funcEntry, "constrain");
  // module attributes {llzk.lang} {
  //   function.def @entry() -> index {
  //     %0 = call @constrain() : () -> index
  //   }
  //   function.def @constrain() -> index {
  //   }
  // }

  auto ops = funcEntry->getBody().getOps();
  ASSERT_FALSE(ops.empty());
  CallOp call = llvm::dyn_cast_if_present<CallOp>(*ops.begin());
  ASSERT_FALSE(call == nullptr);

  ASSERT_FALSE(call.calleeIsCompute());
  ASSERT_TRUE(call.calleeIsConstrain());

  ASSERT_FALSE(call.calleeIsStructCompute());
  ASSERT_FALSE(call.calleeIsStructConstrain());
}

//===------------------------------------------------------------------===//
// FuncDefOp function.arg_name attribute tests
//===------------------------------------------------------------------===//

TEST_F(OpTests, testFuncDefOpArgNameAccessors) {
  mlir::OpBuilder opBuilder(&ctx);
  opBuilder.setInsertionPointToStart(mod->getBody());

  auto funcType = opBuilder.getFunctionType({opBuilder.getI1Type(), opBuilder.getI1Type()}, {});
  auto func = opBuilder.create<function::FuncDefOp>(loc, "test", funcType);

  ASSERT_FALSE(func.hasArgName(0));
  ASSERT_FALSE(func.getArgNameAttr(0));

  func.setArgName(0, "input");
  func.setArgNameAttr(1, opBuilder.getStringAttr("input/1"));

  ASSERT_TRUE(func.hasArgName(0));
  ASSERT_EQ(func.getArgNameAttr(0)->getValue(), "input");
  ASSERT_TRUE(func.hasArgName(1));
  ASSERT_EQ(func.getArgNameAttr(1)->getValue(), "input/1");
  ASSERT_FALSE(func.hasArgName(2));
  ASSERT_FALSE(func.getArgNameAttr(2));
}
