//===-- Attrs.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/IR/Attrs.h"

#include "llzk/Dialect/Struct/IR/Types.h"
#include "llzk/Util/Constants.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/BuiltinAttributes.h>

using namespace mlir;

namespace llzk {

using namespace component;

FailureOr<StructType> getTypeFromLlzkMainAttr(ModuleOp op, Attribute attr) {
  assert(op);   // pre-condition
  assert(attr); // pre-condition

  // If the attribute is present, it must be a TypeAttr of concrete StructType.
  if (TypeAttr ta = llvm::dyn_cast<TypeAttr>(attr)) {
    if (auto st = llvm::dyn_cast<StructType>(ta.getValue())) {
      if (isConcreteType(st)) {
        return success(st);
      }
    }
  }
  return op->emitError().append(
      '"', MAIN_ATTR_NAME, "\" on module must be a concrete '", StructType::name,
      "' attribute. Found: ", attr
  );
}

} // namespace llzk
