// RUN: llzk-opt -cse %s | FileCheck %s

// Both unused output and internal wires must remain declared. Repeated linear
// conversions can still be eliminated independently of those declarations.
module attributes {llzk.lang = "r1cs"} {
  r1cs.circuit @UnusedSignals inputs (%input: !r1cs.signal) {
    %unusedOutput = r1cs.def 1 : !r1cs.signal {pub = #r1cs.pub}
    %unusedInternal = r1cs.def 2 : !r1cs.signal
    %usedInternal = r1cs.def 3 : !r1cs.signal
    %lhs = r1cs.to_linear %input : !r1cs.signal to !r1cs.linear
    %rhs = r1cs.to_linear %input : !r1cs.signal to !r1cs.linear
    %result = r1cs.to_linear %usedInternal : !r1cs.signal to !r1cs.linear
    r1cs.constrain %lhs, %rhs, %result : !r1cs.linear
  }
}

// CHECK-LABEL: module attributes {llzk.lang = "r1cs"} {
// CHECK-NEXT:    r1cs.circuit @UnusedSignals inputs (%[[VAL_0:[0-9a-zA-Z_\.]+]]: !r1cs.signal) {
// CHECK-NEXT:      %[[VAL_1:[0-9a-zA-Z_\.]+]] = r1cs.def 1 : !r1cs.signal {pub = #r1cs.pub}
// CHECK-NEXT:      %[[VAL_2:[0-9a-zA-Z_\.]+]] = r1cs.def 2 : !r1cs.signal
// CHECK-NEXT:      %[[VAL_3:[0-9a-zA-Z_\.]+]] = r1cs.def 3 : !r1cs.signal
// CHECK-NEXT:      %[[VAL_4:[0-9a-zA-Z_\.]+]] = r1cs.to_linear %[[VAL_0]] : !r1cs.signal to !r1cs.linear
// CHECK-NEXT:      %[[VAL_5:[0-9a-zA-Z_\.]+]] = r1cs.to_linear %[[VAL_3]] : !r1cs.signal to !r1cs.linear
// CHECK-NEXT:      r1cs.constrain %[[VAL_4]], %[[VAL_4]], %[[VAL_5]] : !r1cs.linear
// CHECK-NEXT:    }
// CHECK-NEXT:  }
