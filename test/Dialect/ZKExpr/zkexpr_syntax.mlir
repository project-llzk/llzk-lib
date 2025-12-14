// RUN: llzk-opt %s > %t

module {
  // CHECK-LABEL: test_op_syntax
  function.def @test_op_syntax(%arg0 : !felt.type,
                               %arg1 : !felt.type
                              ) {

    %1 = ZKExpr.Literal %arg0
    %2 = ZKExpr.Literal %arg1
    %3 = ZKExpr.Add %1 %2
    %4 = ZKExpr.Sub %1 %3
    %5 = ZKExpr.Mul %2 %4
    %6 = ZKExpr.Neg %5

    function.return
  }
}
