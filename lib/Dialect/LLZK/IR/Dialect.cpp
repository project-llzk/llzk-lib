//===-- Dialect.cpp - Dialect method implementations ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/IR/Dialect.h"

#include "llzk/Config/Config.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Versioning.h"
#include "llzk/Dialect/Struct/IR/Types.h"

#include <mlir/Bytecode/BytecodeImplementation.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/TypeSwitch.h>

#include <limits>

// TableGen'd implementation files
#include "llzk/Dialect/LLZK/IR/Dialect.cpp.inc"

// Need a complete declaration of storage classes for below
#define GET_ATTRDEF_CLASSES
#include "llzk/Dialect/LLZK/IR/Attrs.cpp.inc"

using namespace mlir;
using namespace llzk;

//===------------------------------------------------------------------===//
// LLZKDialect
//===------------------------------------------------------------------===//

namespace {

/// Denotes which dialect attribute is serialized.
enum class LLZKAttrEncoding : uint8_t {
  LoopBounds = 0,
};

struct LLZKDialectBytecodeInterfaceImpl : public LLZKDialectBytecodeInterface<LLZKDialect> {
  using LLZKDialectBytecodeInterface::LLZKDialectBytecodeInterface;

  Attribute readAttribute(DialectBytecodeReader &reader) const final {
    uint64_t encoding;
    if (failed(reader.readVarInt(encoding))) {
      return {};
    }
    if (encoding > std::numeric_limits<uint8_t>::max()) {
      reader.emitError() << "unknown LLZK attribute encoding: " << encoding;
      return {};
    }

    switch (static_cast<LLZKAttrEncoding>(encoding)) {
    case LLZKAttrEncoding::LoopBounds: {
      FailureOr<APInt> lower = readAPInt(reader);
      FailureOr<APInt> upper = readAPInt(reader);
      FailureOr<APInt> step = readAPInt(reader);
      if (failed(lower) || failed(upper) || failed(step)) {
        return {};
      }
      return LoopBoundsAttr::get(getContext(), *lower, *upper, *step);
    }
    }

    reader.emitError() << "unknown LLZK attribute encoding: " << encoding;
    return {};
  }

  LogicalResult writeAttribute(Attribute attr, DialectBytecodeWriter &writer) const final {
    if (auto loopBounds = dyn_cast<LoopBoundsAttr>(attr)) {
      writer.writeVarInt(static_cast<uint64_t>(LLZKAttrEncoding::LoopBounds));
      writeAPInt(writer, loopBounds.getLower());
      writeAPInt(writer, loopBounds.getUpper());
      writeAPInt(writer, loopBounds.getStep());
      return success();
    }
    return failure();
  }
};

LogicalResult verifyLlzkMainAttr(Operation *op, Attribute attr) {
  ModuleOp moduleOp = llvm::dyn_cast<ModuleOp>(op);
  if (!moduleOp) {
    return op->emitError().append(
        '"', MAIN_ATTR_NAME, "\" attribute can only be specified on '",
        ModuleOp::getOperationName(), '\''
    );
  }

  FailureOr<component::StructType> mainStructTypeOpt = getTypeFromLlzkMainAttr(moduleOp, attr);
  if (succeeded(mainStructTypeOpt)) {
    if (component::StructType st = mainStructTypeOpt.value()) {
      SymbolTableCollection symbolTables;
      return st.getDefinition(symbolTables, op);
    }
  }
  return failure();
}

} // namespace

LogicalResult LLZKDialect::verifyOperationAttribute(Operation *op, NamedAttribute attr) {
  if (attr.getName() == MAIN_ATTR_NAME) {
    return verifyLlzkMainAttr(op, attr.getValue());
  }
  return success();
}

auto LLZKDialect::initialize() -> void {
  // clang-format off
  // Suppress false positive from `clang-tidy`
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "llzk/Dialect/LLZK/IR/Attrs.cpp.inc"
  >();

  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/LLZK/IR/Ops.cpp.inc"
  >();
  // clang-format on
  addInterfaces<LLZKDialectBytecodeInterfaceImpl>();
}
