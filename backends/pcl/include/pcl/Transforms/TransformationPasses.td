//===-- TransformationPasses.td ----------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef PCL_TRANSFORMATION_PASSES_TD
#define PCL_TRANSFORMATION_PASSES_TD

include "mlir/Pass/PassBase.td"

class PCLPass<string passArg, string operation = "::mlir::ModuleOp">
    : Pass<passArg, operation> {
  let dependentDialects = [
      // clang-format off
      "pcl::PCLDialect",
      "mlir::func::FuncDialect"
      // clang-format on
  ];
}

def TrimExprSizePass
    : PCLPass<"pcl-trim-expression-size", "::mlir::func::FuncOp"> {
  let summary = "Trims the size of expressions by introducing intermediates";
  let description = [{
    Trims the size of expressions by introducing intermediate bindings. This transformation 
    is recommended before translating to PCL s-expressions if the circuit is large or 
    it was converted from a circuit with many free functions.

    When translating the MLIR representation of the PCL circuit to the s-expressions representation,
    each expression is expanded at each use point. This could lead to an explosion of the IR 
    size. For example, consider the following IR:

    ```
    %0 = pcl.var "a" false
    %1 = pcl.var "b" false
    %2 = pcl.add %0, %1
    %3 = pcl.mul %2, %2
    %4 = pcl.add %2, %3
    %5 = pcl.var "c" false
    %6 = pcl.eq %5, %4
    pcl.assert %6
    %7 = pcl.var "d" false
    %8 = pcl.eq %7, %4
    pcl.assert %8
    %9 = pcl.var "e" false
    %10 = pcl.eq %9, %4
    pcl.assert %10
    ```
    
    Its s-expressions representation will expand to something roughtly 3 times the size:

    ```
    (assert (= c (+ (+ a b) (* (+ a b) (+ a b)))))
    (assert (= d (+ (+ a b) (* (+ a b) (+ a b)))))
    (assert (= e (+ (+ a b) (* (+ a b) (+ a b)))))
    ```

    This pass will introduce an intermediate binding in the middle of large expressions, resulting in smaller 
    s-expressions in the final representation. The example above would be trimmed as follows.

    ```
    %0 = pcl.var "a" false
    %1 = pcl.var "b" false
    %2 = pcl.add %0, %1
    %3 = pcl.mul %2, %2
    %4 = pcl.add %2, %3
    %5 = pcl.var "t0" false 
    %6 = pcl.eq %5, %4
    pcl.assert %6
    %7 = pcl.var "c" false
    %8 = pcl.eq %7, %5
    pcl.assert %8
    %9 = pcl.var "d" false
    %10 = pcl.eq %9, %5
    pcl.assert %10
    %11 = pcl.var "e" false
    %12 = pcl.eq %11, %5
    pcl.assert %12
    ```

    Which results in the following s-expressions:

    ```
    (assert (= t0 (+ (+ a b) (* (+ a b) (+ a b)))))
    (assert (= c t0))
    (assert (= d t0))
    (assert (= e t0))
    ```
  }];

  let options = [Option<"maxSize", "max-size", "unsigned", /*default=*/"10",
                        "The maximum size expressions are allowed to have">,
  ];
}

#endif // PCL_TRANSFORMATION_PASSES_TD
