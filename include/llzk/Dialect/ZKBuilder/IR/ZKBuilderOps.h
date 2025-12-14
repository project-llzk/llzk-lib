#ifndef LIB_DIALECT_ZKBUILDER_ZKBUILDEROPS_H_
#define LIB_DIALECT_ZKBUILDER_ZKBUILDEROPS_H_

#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderDialect.h"
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderTypes.h"
#include "llzk/Dialect/ZKExpr/IR/ZKExprTypes.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>

#define GET_OP_CLASSES
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderOps.h.inc"

#endif // LIB_DIALECT_ZKBUILDER_ZKBUILDEROPS_H_
