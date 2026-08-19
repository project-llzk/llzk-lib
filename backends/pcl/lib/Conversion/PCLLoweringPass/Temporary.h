//===-- Temporary.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringMap.h>

namespace pcl::lowering {
/// Keeps track of temporary names during conversion.
class Temporaries {
  llvm::DenseMap<mlir::Operation *, mlir::StringAttr> names;

public:
  Temporaries(mlir::ModuleOp root);

  /// Returns, if available, the temporary created for the given operation.
  mlir::FailureOr<mlir::StringAttr> get(mlir::Operation *op) const {
    auto it = names.find(op);
    if (it == names.end()) {
      return op->emitOpError() << "does not emit a temporary";
    }
    return it->getSecond();
  }

  /// Returns true if the op has a temporary name associated to it.
  bool hasTemp(mlir::Operation *op) const { return names.contains(op); }
};
} // namespace pcl::lowering
