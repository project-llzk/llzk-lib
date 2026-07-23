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

#include <llvm/Support/Debug.h>
#include <algorithm>

// TableGen'd implementation files
#include "pcl/Dialect/IR/Dialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "pcl/Dialect/IR/Attrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "pcl/Dialect/IR/Types.cpp.inc"

void pcl::PCLDialect::initialize() {
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
}

using namespace pcl;

//===----------------------------------------------------------------------===//
// PCLDialect
//===----------------------------------------------------------------------===//

mlir::LogicalResult
PCLDialect::verifyOperationAttribute(mlir::Operation *op, mlir::NamedAttribute attr) {
  if (attr.getName() == "pcl.prime") {
    auto prime = llvm::dyn_cast<pcl::PrimeAttr>(attr.getValue());
    if (!prime) {
      return op->emitError() << "'pcl.prime' must be a #pcl.prime<...>";
    }

    if (!llvm::isa<mlir::ModuleOp>(op)) {
      return op->emitError() << "'pcl.prime' may only be on builtin.module";
    }

    const llvm::APInt &v = prime.getValue();
    if (v.isZero() || v.isNegative()) {
      return op->emitError() << "prime must be positive";
    }
  }
  return mlir::success();
}

mlir::Operation *PCLDialect::materializeConstant(
    mlir::OpBuilder &builder, mlir::Attribute value, mlir::Type, mlir::Location loc
) {
  return llvm::TypeSwitch<mlir::Attribute, mlir::Operation *>(value)
      .Case<FeltAttr>([&builder, loc](auto attr) -> mlir::Operation * {
    return builder.create<ConstOp>(loc, attr);
  })
      .Case<BoolAttr>([&builder, loc](auto attr) -> mlir::Operation * {
    if (attr.getValue()) {
      return builder.create<TrueOp>(loc);
    } else {
      return builder.create<FalseOp>(loc);
    }
  }).Default([](auto) {
    llvm_unreachable("unsupported constant attribute");
    return nullptr;
  });
}

//===----------------------------------------------------------------------===//
// PrimeAttr
//===----------------------------------------------------------------------===//

namespace {
  
}

FeltAttr PrimeAttr::reduce(FeltAttr attr) {
  auto max = std::max({getValue().getBitWidth(), attr.getValue().getBitWidth()}) + 1;
  auto pExt = getValue().zext(max);
  // The incoming value could be negative so we need to sign-extend.
  auto vExt = attr.getValue().sext(max);
  auto value = vExt.srem(pExt);
  if (value.isNegative()) {
    value += pExt;
  }
  return FeltAttr::get(getContext(), value);
}
