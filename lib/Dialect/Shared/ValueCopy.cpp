//===-- ValueCopy.cpp - Mutable value-copy materialization ------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Shared/ValueCopy.h"

#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/String/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Types.h"

#include <mlir/IR/BuiltinTypes.h>

using namespace mlir;

namespace llzk {

namespace {

/// Return whether copying a scalar or handle of `type` preserves the same SSA value.
static bool isIdentityPreservingCopy(Type type) {
  // Struct values are identity-bearing handles: copies continue to reference the same component.
  // Keep concrete immutable types explicit so future mutable types cannot inherit this policy.
  return llvm::isa<
      IndexType, IntegerType, felt::FeltType, string::StringType, component::StructType>(type);
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

std::string ValueCopyDialectInterface::getValueCopyFailureReason(Type type) const {
  std::string reason;
  llvm::raw_string_ostream(reason) << "unsupported value-copy type '" << type << "'";
  return reason;
}

std::string getValueCopyFailureReason(Type type) {
  if (canMaterializeValueCopy(type)) {
    return {};
  }
  if (const auto *interface = getValueCopyInterface(type)) {
    return interface->getValueCopyFailureReason(type);
  }
  std::string reason;
  llvm::raw_string_ostream(reason) << "unsupported value-copy type '" << type << "'";
  return reason;
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
