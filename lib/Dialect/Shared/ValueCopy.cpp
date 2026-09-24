//===-- ValueCopy.cpp - Mutable value-copy materialization ------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Shared/ValueCopy.h"

#include <mlir/IR/BuiltinTypes.h>

using namespace mlir;

namespace llzk {

namespace {

/// Return whether copying a scalar or handle of `type` preserves the same SSA value.
static bool isIdentityPreservingCopy(Type type) {
  if (llvm::isa<IndexType, IntegerType>(type)) {
    return true;
  }

  // These LLZK dialects define immutable scalar values. Struct values are identity-bearing
  // handles: copying one preserves the identity of the referenced mutable component rather than
  // duplicating its member storage. Aggregate-specific identity rules, such as shape-only arrays,
  // belong to their dialect interface. Keep this list explicit so a newly added mutable type cannot
  // silently acquire alias semantics merely because it lacks the interface.
  StringRef dialectNamespace = type.getDialect().getNamespace();
  return dialectNamespace == "bool" || dialectNamespace == "felt" || dialectNamespace == "string" ||
         dialectNamespace == "struct";
}

static const ValueCopyDialectInterface *getValueCopyInterface(Type type) {
  return type.getDialect().getRegisteredInterface<ValueCopyDialectInterface>();
}

} // namespace

bool requiresValueCopy(Type type) { return !isIdentityPreservingCopy(type); }

bool canMaterializeValueCopy(Type type) {
  if (isIdentityPreservingCopy(type)) {
    return true;
  }
  if (const auto *interface = getValueCopyInterface(type)) {
    return interface->canMaterializeValueCopy(type);
  }
  return false;
}

FailureOr<Value> materializeValueCopy(OpBuilder &builder, Location loc, Value source) {
  Type type = source.getType();
  if (isIdentityPreservingCopy(type)) {
    return source;
  }
  if (const auto *interface = getValueCopyInterface(type)) {
    return interface->materializeValueCopy(builder, loc, source);
  }
  return failure();
}

} // namespace llzk
