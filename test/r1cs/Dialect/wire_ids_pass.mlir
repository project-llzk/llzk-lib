// RUN: r1cs-opt %s | FileCheck %s

// Explicit IDs use R1CS wire classes without any LLZK layout metadata.
module {
  r1cs.circuit @Main inputs (%private: !r1cs.signal, %public: !r1cs.signal {pub = #r1cs.pub}) {
    %out = r1cs.def 1 : !r1cs.signal {pub = #r1cs.pub}
    %internal = r1cs.def 4 : !r1cs.signal
  } {r1cs.input_labels = array<i64: 3, 2>}
}

// CHECK-LABEL: module {
// CHECK-NEXT:    r1cs.circuit @Main inputs (%[[VAL_0:[0-9a-zA-Z_\.]+]]: !r1cs.signal, %[[VAL_1:[0-9a-zA-Z_\.]+]]: !r1cs.signal {pub = #r1cs.pub}) {
// CHECK-NEXT:      %[[VAL_2:[0-9a-zA-Z_\.]+]] = r1cs.def 1 : !r1cs.signal {pub = #r1cs.pub}
// CHECK-NEXT:      %[[VAL_3:[0-9a-zA-Z_\.]+]] = r1cs.def 4 : !r1cs.signal
// CHECK-NEXT:    } {r1cs.input_labels = array<i64: 3, 2>}
// CHECK-NEXT:  }
