//===-- TypeConverter.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "TypeConverter.h"

#include "pcl/Dialect/IR/Ops.h"
#include "pcl/Dialect/IR/Types.h"

#include "llzk/Dialect/Felt/IR/Types.h"

using namespace pcl::lowering;
using namespace mlir;

PCLTypeConverter::PCLTypeConverter() {
  // Default conversion.
  addConversion([](Type t) { return t; });

  addConversion([](IntegerType t) -> Type { return BoolType::get(t.getContext()); });

  addConversion([](llzk::felt::FeltType t) { return FeltType::get(t.getContext()); });

  addSourceMaterialization(
      [](OpBuilder &builder, Type t, ValueRange values, Location location) -> Value {
    if (values.size() != 1) {
      return nullptr;
    }
    return builder.create<UnrealizedConversionCastOp>(location, t, values[0]).getResult(0);
  }
  );

  addTargetMaterialization(
      [](OpBuilder &builder, Type t, ValueRange values, Location location) -> Value {
    if (values.size() != 1) {
      return nullptr;
    }

    return builder.create<UnrealizedConversionCastOp>(location, t, values[0]).getResult(0);
  }
  );

  // Handles the conversion from booleans to felts.
  //
  // This conversion may be necessary in situations where a boolean result is used as operand of
  // an operation that expects a felt. For example, the following input IR:
  //
  // ```
  //  %felt_const_1 = felt.const 1 : !F
  //  %felt_const_65536 = felt.const 65536 : !F
  //  %0 = bool.cmp lt(%in, %felt_const_65536) : !F, !F
  //  %1 = cast.tofelt %0 : i1, !F
  //  constrain.eq %1, %felt_const_1 : !F, !F
  // ```
  //
  // Can be represented in PCL as:
  //
  // ```
  // (assert (= (< %in 65536) 1))
  // ```
  //
  // The result of `(< %in 65536)` needs to be converted from a `pcl.bool` to a `pcl.felt` in
  // order for the IR to typecheck.
  addTargetMaterialization(
      [](OpBuilder &builder, FeltType, ValueRange values, Location location) -> Value {
    if (values.size() != 1 || !llvm::isa<BoolType>(values[0].getType())) {
      return nullptr;
    }
    return builder.create<AsFeltOp>(location, values[0]);
  }
  );

  // Handles the conversion from felts to booleans.
  //
  // This conversion is the counterpart of the conversion above and is used in situations where a
  // felt was passed as operand to an op that expects a boolean.
  //
  // The value is converted by testing for equality against the falsy value (0).
  addTargetMaterialization(
      [](OpBuilder &builder, BoolType, ValueRange values, Location location) -> Value {
    if (values.size() != 1 || !llvm::isa<FeltType>(values[0].getType())) {
      return nullptr;
    }

    llvm::APInt zeroValue;
    auto zero = builder.create<ConstOp>(location, FeltAttr::get(builder.getContext(), zeroValue));
    auto eqOp = builder.create<CmpEqOp>(location, values[0], zero);
    return builder.create<NotOp>(location, eqOp);
  }
  );
}
