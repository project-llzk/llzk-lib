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

#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>

namespace llzk {

template <typename OpClass, typename... Args>
inline OpClass delegate_to_build(mlir::Location location, Args &&...args) {
  mlir::OpBuilder builder(location->getContext());
  return builder.create<OpClass>(location, std::forward<Args>(args)...);
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
