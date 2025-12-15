//===-- Passes.td ------------------------------------------------*- td -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_CONVERSIONS_PASSES_TD
#define LLZK_CONVERSIONS_PASSES_TD

include "llzk/Pass/PassBase.td"

def ConvertLLZKToZKLeanPass
    : Pass<"convert-llzk-to-zklean", "::mlir::ModuleOp"> {
  let summary = "Wrap felt constants in ZKExpr.Literal ops";
  let description = [{
    Converts felt and constrain operations into the ZKLean dialects by
    materializing corresponding ZKExpr.Literal for each felt.const operation
    and replacing felt / constrain operations with respective ZKExpr /
    ZKBuilder operations. The literal operation consumes a felt value so its
    operand is a cloned felt.const while the original result is replaced with
    the ZKExpr literal.
  }];
  let dependentDialects = [
    "llzk::felt::FeltDialect",
    "llzk::constrain::ConstrainDialect",
    "mlir::zkexpr::ZKExprDialect",
    "mlir::zkbuilder::ZKBuilderDialect"
  ];
  let constructor = "llzk::createConvertLLZKToZKLeanPass()";
}

#endif // LLZK_CONVERSIONS_PASSES_TD
