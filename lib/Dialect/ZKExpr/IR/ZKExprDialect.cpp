#include "llzk/Dialect/ZKExpr/IR/ZKExprDialect.h"
#include "llzk/Dialect/ZKExpr/IR/ZKExprOps.h"
#include "llzk/Dialect/ZKExpr/IR/ZKExprTypes.h"

#include <mlir/IR/Builders.h>
#include <llvm/ADT/TypeSwitch.h>

#include "llzk/Dialect/ZKExpr/IR/ZKExprDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/ZKExpr/IR/ZKExprTypes.cpp.inc"
#define GET_OP_CLASSES
#include "llzk/Dialect/ZKExpr/IR/ZKExprOps.cpp.inc"

auto mlir::zkexpr::ZKExprDialect::initialize() -> void {
  addTypes<
#define GET_TYPEDEF_LIST
#include "llzk/Dialect/ZKExpr/IR/ZKExprTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "llzk/Dialect/ZKExpr/IR/ZKExprOps.cpp.inc"
      >();
}
