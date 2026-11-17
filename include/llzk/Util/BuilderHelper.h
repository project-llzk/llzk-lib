//===-- BuilderHelper.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Util/ErrorHelper.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

namespace llzk {

/// Clone a block's non-terminator operations with the supplied argument bindings and return
/// the remapped forwarded values. A fresh mapping keeps separate invocations independent;
/// captures outside the block retain their original values.
inline llvm::SmallVector<mlir::Value> cloneBlockWithMappedArguments(
    mlir::OpBuilder &builder, mlir::Block &block, mlir::ValueRange arguments,
    mlir::ValueRange forwarded
) {
  mlir::IRMapping mapping;
  mapping.map(block.getArguments(), arguments);
  for (mlir::Operation &op : block.without_terminator()) {
    builder.clone(op, mapping);
  }
  return llvm::map_to_vector(forwarded, [&](mlir::Value value) {
    return mapping.lookupOrDefault(value);
  });
}

template <typename OpClass, typename... Args>
inline OpClass delegate_to_build(mlir::Location location, Args &&...args) {
  mlir::OpBuilder builder(location->getContext());
  return OpClass::create(builder, location, std::forward<Args>(args)...);
}

template <typename OpClass>
void addTemplateParams(
    mlir::OpBuilder &odsBuilder, typename OpClass::Properties &props,
    llvm::ArrayRef<mlir::Attribute> templateParams
) {
  if (!templateParams.empty()) {
    // Must attempt to convert attribute types but `build()` functions do not have a failure path or
    // error reporting. That comes during validation of the constructed op so ignore errors here.
    llvm::FailureOr<llvm::SmallVector<mlir::Attribute>> r =
        llzk::forceIntAttrTypes(templateParams, [&odsBuilder]() {
      return llzk::InFlightDiagnosticWrapper::createSilent(odsBuilder.getContext());
    });
    llvm::ArrayRef<mlir::Attribute> converted = succeeded(r) ? r.value() : templateParams;
    props.setTemplateParams(odsBuilder.getArrayAttr(converted));
  }
}

} // namespace llzk
