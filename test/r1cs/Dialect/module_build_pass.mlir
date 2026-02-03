// RUN: r1cs-opt %s 2>&1 | FileCheck --enable-var-scope %s

module attributes {llzk.lang = "r1cs"} {
  r1cs.circuit @CmpConstraint inputs (%arg0: !r1cs.signal {pub = #r1cs.pub}, %arg1: !r1cs.signal, %arg2: !r1cs.signal) {
    %0 = r1cs.def 0 : !r1cs.signal {pub = #r1cs.pub}
    %1 = r1cs.def 1 : !r1cs.signal
    %2 = r1cs.def 2 : !r1cs.signal
    %3 = r1cs.to_linear %arg1 : !r1cs.signal to !r1cs.linear
    %4 = r1cs.to_linear %arg2 : !r1cs.signal to !r1cs.linear
    %5 = r1cs.to_linear %1 : !r1cs.signal to !r1cs.linear
    %6 = r1cs.mul_const %5,  -1 : i64 : !r1cs.linear
    r1cs.constrain %3, %4, %6 : !r1cs.linear
    %7 = r1cs.to_linear %arg0 : !r1cs.signal to !r1cs.linear
    %8 = r1cs.to_linear %2 : !r1cs.signal to !r1cs.linear
    %9 = r1cs.mul_const %8,  -1 : i64 : !r1cs.linear
    %10 = r1cs.add %6, %9 : !r1cs.linear
    r1cs.constrain %7, %3, %10 : !r1cs.linear
    r1cs.constrain %7, %3, %9 : !r1cs.linear
    %11 = r1cs.to_linear %0 : !r1cs.signal to !r1cs.linear
    %12 = r1cs.mul_const %11,  -1 : i64 : !r1cs.linear
    %13 = r1cs.mul_const %3,  -1 : i64 : !r1cs.linear
    %14 = r1cs.add %12, %13 : !r1cs.linear
    r1cs.constrain %7, %3, %14 : !r1cs.linear
  }
}

// CHECK-LABEL:   r1cs.circuit @CmpConstraint inputs (
// CHECK-SAME:                                        %[[VAL_0:[0-9a-zA-Z_\.]+]]: !r1cs.signal {#r1cs.pub},
// CHECK-SAME:                                        %[[VAL_1:[0-9a-zA-Z_\.]+]]: !r1cs.signal,
// CHECK-SAME:                                        %[[VAL_2:[0-9a-zA-Z_\.]+]]: !r1cs.signal) {
// CHECK:           %[[VAL_3:[0-9a-zA-Z_\.]+]] = r1cs.def 0 : !r1cs.signal {pub = #r1cs.pub}
// CHECK:           %[[VAL_4:[0-9a-zA-Z_\.]+]] = r1cs.def 1 : !r1cs.signal
// CHECK:           %[[VAL_5:[0-9a-zA-Z_\.]+]] = r1cs.def 2 : !r1cs.signal
// CHECK:           %[[VAL_6:[0-9a-zA-Z_\.]+]] = r1cs.to_linear %[[VAL_1]] : !r1cs.signal to !r1cs.linear
// CHECK:           %[[VAL_7:[0-9a-zA-Z_\.]+]] = r1cs.to_linear %[[VAL_2]] : !r1cs.signal to !r1cs.linear
// CHECK:           %[[VAL_8:[0-9a-zA-Z_\.]+]] = r1cs.to_linear %[[VAL_4]] : !r1cs.signal to !r1cs.linear
// CHECK:           %[[VAL_9:[0-9a-zA-Z_\.]+]] = r1cs.mul_const %[[VAL_8]],  -1 : i64 : !r1cs.linear
// CHECK:           r1cs.constrain %[[VAL_6]], %[[VAL_7]], %[[VAL_9]] : !r1cs.linear
// CHECK:           %[[VAL_10:[0-9a-zA-Z_\.]+]] = r1cs.to_linear %[[VAL_0]] : !r1cs.signal to !r1cs.linear
// CHECK:           %[[VAL_11:[0-9a-zA-Z_\.]+]] = r1cs.to_linear %[[VAL_5]] : !r1cs.signal to !r1cs.linear
// CHECK:           %[[VAL_12:[0-9a-zA-Z_\.]+]] = r1cs.mul_const %[[VAL_11]],  -1 : i64 : !r1cs.linear
// CHECK:           %[[VAL_13:[0-9a-zA-Z_\.]+]] = r1cs.add %[[VAL_9]], %[[VAL_12]] : !r1cs.linear
// CHECK:           r1cs.constrain %[[VAL_10]], %[[VAL_6]], %[[VAL_13]] : !r1cs.linear
// CHECK:           r1cs.constrain %[[VAL_10]], %[[VAL_6]], %[[VAL_12]] : !r1cs.linear
// CHECK:           %[[VAL_14:[0-9a-zA-Z_\.]+]] = r1cs.to_linear %[[VAL_3]] : !r1cs.signal to !r1cs.linear
// CHECK:           %[[VAL_15:[0-9a-zA-Z_\.]+]] = r1cs.mul_const %[[VAL_14]],  -1 : i64 : !r1cs.linear
// CHECK:           %[[VAL_16:[0-9a-zA-Z_\.]+]] = r1cs.mul_const %[[VAL_6]],  -1 : i64 : !r1cs.linear
// CHECK:           %[[VAL_17:[0-9a-zA-Z_\.]+]] = r1cs.add %[[VAL_15]], %[[VAL_16]] : !r1cs.linear
// CHECK:           r1cs.constrain %[[VAL_10]], %[[VAL_6]], %[[VAL_17]] : !r1cs.linear
// CHECK:         }
