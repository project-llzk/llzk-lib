//===-- Dialect.cpp - Global value dialect implementation -------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Global/IR/Dialect.h"

#include "InitializerUtils.h"

#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Versioning.h"

// TableGen'd implementation files
#include "llzk/Dialect/Global/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace llzk;

//===------------------------------------------------------------------===//
// GlobalDialect
//===------------------------------------------------------------------===//

namespace {

class GlobalDialectBytecodeInterface : public LLZKDialectBytecodeInterface<global::GlobalDialect> {
  using Base = LLZKDialectBytecodeInterface<global::GlobalDialect>;

public:
  using Base::Base;

  LogicalResult upgradeFromVersion(
      Operation *root, const LLZKDialectVersion & /*current*/,
      const LLZKDialectVersion & /*requested*/
  ) const final {
    auto res = root->walk([](global::GlobalDefOp global) -> WalkResult {
      if (Attribute initialValue = global.getInitialValueAttr()) {
        Type type = global.getType();
        FailureOr<Attribute> normalized = global::normalizeGlobalInitializer(
            type, initialValue, [context = initialValue.getContext()] {
          return InFlightDiagnosticWrapper::createSilent(context);
        }
        );
        if (failed(normalized)) {
          return global.emitError(
              "contains a legacy initializer that is incompatible with its declared type"
          );
        }
        global.setInitialValueAttr(*normalized);
        global.setTypeAttr(TypeAttr::get(type));
      }
      return WalkResult::advance();
    });
    return failure(res.wasInterrupted());
  }
};

} // namespace

auto global::GlobalDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/Global/IR/Ops.cpp.inc"
  >();
  // clang-format on
  addInterfaces<GlobalDialectBytecodeInterface>();
}
