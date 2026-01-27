#include "llzk/Dialect/ZKLeanStruct/IR/ZKLeanStructDialect.h"
#include "llzk/Dialect/ZKLeanStruct/IR/ZKLeanStructOps.h"
#include "llzk/Dialect/ZKLeanStruct/IR/ZKLeanStructTypes.h"

#include <mlir/IR/Builders.h>
#include <llvm/ADT/TypeSwitch.h>

#include "llzk/Dialect/ZKLeanStruct/IR/ZKLeanStructDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/ZKLeanStruct/IR/ZKLeanStructTypes.cpp.inc"
#define GET_OP_CLASSES
#include "llzk/Dialect/ZKLeanStruct/IR/ZKLeanStructOps.cpp.inc"

auto mlir::zkleanstruct::ZKLeanStructDialect::initialize() -> void {
  addTypes<
#define GET_TYPEDEF_LIST
#include "llzk/Dialect/ZKLeanStruct/IR/ZKLeanStructTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "llzk/Dialect/ZKLeanStruct/IR/ZKLeanStructOps.cpp.inc"
      >();
}
