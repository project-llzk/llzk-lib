#ifndef LIB_DIALECT_ZKLEANLEAN_ZKLEANLEANOPS_H_
#define LIB_DIALECT_ZKLEANLEAN_ZKLEANLEANOPS_H_

#include "llzk/Dialect/ZKExpr/IR/ZKExprTypes.h"
#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanDialect.h"
#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanTypes.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>

#define GET_OP_CLASSES
#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanOps.h.inc"

#endif  // LIB_DIALECT_ZKLEANLEAN_ZKLEANLEANOPS_H_
