//===-- PCL.cpp - C API for Picus PCL Target --------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/CAPI/Support.h"
#include "llzk/Config/Config.h"

#include <mlir-c/IR.h>
#include <mlir-c/Support.h>

#include <mlir/CAPI/Support.h>
#include <mlir/Support/LogicalResult.h>

#if LLZK_WITH_PCL

#include "llzk-c/Target/PCL.h"

#include <pcl/Target/PCL.h>

#include <mlir/CAPI/Support.h>
#include <mlir/CAPI/Utils.h>

MlirLogicalResult
llzkTranslateModuleToPCL(MlirOperation module, MlirStringCallback callback, void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  auto op = mlir::unwrap_cast<mlir::ModuleOp>(module);
  return wrap(pcl::moduleToPcl(op, stream));
}

#else

MlirLogicalResult mlirTranslateModuleToPCL(MlirOperation, MlirStringCallback) {
  return wrap(mlir::failure());
}

#endif // LLZK_WITH_PCL
