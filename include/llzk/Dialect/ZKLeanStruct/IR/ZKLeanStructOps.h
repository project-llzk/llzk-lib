#ifndef LIB_DIALECT_ZKLEANSTRUCT_ZKLEANSTRUCTOPS_H_
#define LIB_DIALECT_ZKLEANSTRUCT_ZKLEANSTRUCTOPS_H_

#include "llzk/Dialect/ZKExpr/IR/ZKExprTypes.h"
#include "llzk/Dialect/ZKLeanStruct/IR/ZKLeanStructDialect.h"
#include "llzk/Dialect/ZKLeanStruct/IR/ZKLeanStructTypes.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>

#define GET_OP_CLASSES
#include "llzk/Dialect/ZKLeanStruct/IR/ZKLeanStructOps.h.inc"

#endif  // LIB_DIALECT_ZKLEANSTRUCT_ZKLEANSTRUCTOPS_H_
