#ifndef LIB_DIALECT_ZKEXPR_ZKEXPROPS_H_
#define LIB_DIALECT_ZKEXPR_ZKEXPROPS_H_

#include "llzk/Dialect/ZKExpr/IR/ZKExprDialect.h"
#include "llzk/Dialect/ZKExpr/IR/ZKExprTypes.h"
#include "llzk/Dialect/Felt/IR/Types.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>

#define GET_OP_CLASSES
#include "llzk/Dialect/ZKExpr/IR/ZKExprOps.h.inc"

#endif // LIB_DIALECT_ZKEXPR_ZKEXPROPS_H_
