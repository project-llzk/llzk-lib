// RUN: llzk-opt %s > %t

module {
  // CHECK-LABEL: test_op_syntax
  function.def @test_op_syntax(%arg0 : !felt.type,
                               %arg1 : !felt.type,
                               %arg2 : !felt.type
                              ) {

    %1 = ZKExpr.Literal %arg0
    %2 = ZKExpr.Literal %arg1
    %3 = ZKExpr.Literal %arg2
    %4 = ZKExpr.Add %1 %2 
    %5 = ZKExpr.Sub %1 %3
    %6 = ZKExpr.Mul %2 %1

    %7 = ZKBuilder.ConstrainEq %2 %3
    %8 = ZKBuilder.ConstrainR1CS %3 %2 %1

    function.return
  }
}
