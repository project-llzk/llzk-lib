#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderDialect.h"
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderOps.h"
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderTypes.h"
#include "llzk/Dialect/ZKExpr/IR/ZKExprTypes.h"

#include <mlir/IR/Builders.h>
#include <llvm/ADT/TypeSwitch.h>

#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderTypes.cpp.inc"
#define GET_OP_CLASSES
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderOps.cpp.inc"

namespace mlir {
namespace zkbuilder {

void ZKBuilderDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderOps.cpp.inc"
      >();
}

} // namespace zkbuilder
} // namespace mlir
