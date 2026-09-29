// RUN: not llzk-translate --r1cs-to-binary --r1cs-prime=17 --llzk-layout-map=%t.layout %s -o %t.r1cs 2>&1 | FileCheck %s

// Root-name metadata is optional, but each physical wire must agree with the
// logical signal id attached to its value in the binary exporter model.
// CHECK: error: 'r1cs.circuit' op cannot export layout map: 'poly.wire_bindings' entry for wire 1 does not match the binary export signal

module attributes {llzk.lang = "r1cs"} {
  r1cs.circuit @Main inputs (%arg0: !r1cs.signal) attributes {
    poly.layout_argument_signals = [1 : i64],
    poly.layout_signals = [
      {id = 0 : i64, path = [0, "out"]},
      {id = 1 : i64, path = [1]}
    ],
    poly.wire_bindings = [{path = [1], public = false, signal = 0 : i64, wire = 1 : i64}]
  } {
  }
}
