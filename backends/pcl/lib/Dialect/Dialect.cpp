//===-- Dialect.cpp - PCL dialect implementation ------------*- C++ -*-----===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "pcl/Dialect/IR/Dialect.h"

#include "pcl/Dialect/IR/Ops.h"
#include "pcl/Dialect/IR/Types.h"

#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"

#include <mlir/IR/DialectImplementation.h>

#include <algorithm>

// TableGen'd implementation files
#include "pcl/Dialect/IR/Dialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "pcl/Dialect/IR/Attrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "pcl/Dialect/IR/Types.cpp.inc"

void pcl::PCLDialect::initialize() {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "pcl/Dialect/IR/Ops.cpp.inc"
  >();

  addTypes<
    #define GET_TYPEDEF_LIST
    #include "pcl/Dialect/IR/Types.cpp.inc"
  >();

  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "pcl/Dialect/IR/Attrs.cpp.inc"
  >();
  // clang-format on
}

using namespace pcl;

//===----------------------------------------------------------------------===//
// PCLDialect
//===----------------------------------------------------------------------===//

mlir::LogicalResult
PCLDialect::verifyOperationAttribute(mlir::Operation *op, mlir::NamedAttribute attr) {
  if (attr.getName() == PCL_PRIME_ATTR_NAME) {
    auto prime = llvm::dyn_cast<pcl::PrimeAttr>(attr.getValue());
    if (!prime) {
      return op->emitError() << '\'' << PCL_PRIME_ATTR_NAME << "' must be a #"
                             << PCL_PRIME_ATTR_NAME << "<...>";
    }

    if (!llvm::isa<mlir::ModuleOp>(op)) {
      return op->emitError() << '\'' << PCL_PRIME_ATTR_NAME << "' may only be on builtin.module";
    }

    const llvm::DynamicAPInt &v = prime.getValue();
    if (v < 2) {
      return op->emitError("prime must be at least 2");
    }
  }
  return mlir::success();
}

mlir::Operation *PCLDialect::materializeConstant(
    mlir::OpBuilder &builder, mlir::Attribute value, mlir::Type, mlir::Location loc
) {
  return llvm::TypeSwitch<mlir::Attribute, mlir::Operation *>(value)
      .Case<FeltAttr>([&builder, loc](auto attr) -> mlir::Operation * {
    return ConstOp::create(builder, loc, attr);
  })
      .Case<BoolAttr>([&builder, loc](auto attr) -> mlir::Operation * {
    if (attr.getValue()) {
      return TrueOp::create(builder, loc);
    } else {
      return FalseOp::create(builder, loc);
    }
  }).Default([](auto) {
    llvm_unreachable("unsupported constant attribute");
    return nullptr;
  });
}

//===----------------------------------------------------------------------===//
// PrimeAttr
//===----------------------------------------------------------------------===//

FeltAttr PrimeAttr::reduce(FeltAttr attr) {
  return FeltAttr::get(getContext(), llvm::mod(attr.getValue(), getValue()));
}
