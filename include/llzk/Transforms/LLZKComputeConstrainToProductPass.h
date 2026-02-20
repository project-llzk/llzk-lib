#pragma once

#include "llzk/Analysis/LightweightSignalEquivalenceAnalysis.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"

#include <mlir/Support/LogicalResult.h>

#include <vector>

namespace llzk {

class ProductAligner {
  mlir::SymbolTableCollection &tables;
  LightweightSignalEquivalenceAnalysis &equivalence;

public:
  std::vector<component::StructDefOp> alignedStructs;
  ProductAligner(
      mlir::SymbolTableCollection &_tables, LightweightSignalEquivalenceAnalysis &_equivalence
  )
      : tables {_tables}, equivalence {_equivalence} {}

  // Given a @product function body, try to match up calls to @A::@compute and @A::@constrain for
  // every sub-struct @A and replace them with a call to @A::@product
  mlir::LogicalResult alignCalls(function::FuncDefOp product);

  // Given a StructDefOp @root, replace the @root::@compute and @root::@constrain functions with a
  // @root::@product
  function::FuncDefOp alignFuncs(
      component::StructDefOp root, function::FuncDefOp compute, function::FuncDefOp constrain
  );
};

mlir::LogicalResult alignStartingAt(
    component::StructDefOp root, mlir::SymbolTableCollection &tables,
    LightweightSignalEquivalenceAnalysis &equivalence
);

} // namespace llzk
