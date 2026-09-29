// RUN: llzk-translate --r1cs-to-binary --r1cs-prime=17 --llzk-layout-map=%t.layout %s -o %t.r1cs
// RUN: FileCheck %s --check-prefix=NAMED --input-file=%t.layout
// RUN: sed '/^    poly.layout_root_names =/d' %s > %t.unnamed.mlir
// RUN: llzk-translate --r1cs-to-binary --r1cs-prime=17 --llzk-layout-map=%t.unnamed.layout %t.unnamed.mlir -o %t.unnamed.r1cs
// RUN: FileCheck %s --check-prefix=UNNAMED --input-file=%t.unnamed.layout
// RUN: cmp %t.r1cs %t.unnamed.r1cs

// NAMED: signal 0{{[[:space:]]+}}main["out"]
// NAMED-NEXT: signal 1{{[[:space:]]+}}arg["main"]
// NAMED-NEXT: signal 2{{[[:space:]]+}}arg["a\22b[0]"]
// UNNAMED: signal 0{{[[:space:]]+}}main["out"]
// UNNAMED-NEXT: signal 1{{[[:space:]]+}}arg1
// UNNAMED-NEXT: signal 2{{[[:space:]]+}}arg2
// UNNAMED-NEXT: # r1cs
// UNNAMED-NEXT: wire 0{{[[:space:]]+}}<one>
// UNNAMED-NEXT: wire 1{{[[:space:]]+}}signal 0
// UNNAMED-NEXT: wire 2{{[[:space:]]+}}signal 1
// UNNAMED-NEXT: wire 3{{[[:space:]]+}}signal 2

module attributes {llzk.lang = "r1cs"} {
  r1cs.circuit @Main inputs (%a: !r1cs.signal, %b: !r1cs.signal) attributes {
    poly.layout_root_names = {"0" = "ignored", "1" = "main", "2" = "a\22b[0]"},
    poly.layout_argument_signals = [1 : i64, 2 : i64],
    poly.layout_signals = [{id = 0 : i64, path = [0, "out"]}, {id = 1 : i64, path = [1]}, {id = 2 : i64, path = [2]}],
    poly.wire_bindings = [{path = [0, "out"], public = true, signal = 0 : i64, wire = 1 : i64}, {path = [1], public = false, signal = 1 : i64, wire = 2 : i64}, {path = [2], public = false, signal = 2 : i64, wire = 3 : i64}]
  } {
    %out = r1cs.def 1 : !r1cs.signal {poly.layout_signal = 0 : i64, pub = #r1cs.pub}
  }
}
