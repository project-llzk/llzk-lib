#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanDialect.h"
#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanOps.h"
#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanTypes.h"

#include <mlir/IR/Builders.h>
#include <llvm/ADT/TypeSwitch.h>

#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanTypes.cpp.inc"
#define GET_OP_CLASSES
#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanOps.cpp.inc"

auto mlir::zkleanlean::ZKLeanLeanDialect::initialize() -> void {
  addTypes<
#define GET_TYPEDEF_LIST
#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanOps.cpp.inc"
      >();
}
