// RUN: r1cs-opt --r1cs-export-binary='output-file=%t.bin prime=17' %s -o /dev/null
// RUN: test -f %t.bin
// RUN: od -An -v -t x1 %t.bin | tr -d ' \n' | FileCheck %s --check-prefix=HEX

// HEX: 723163730100000003000000010000002800000000000000080000001100000000000000050000000100000001000000010000000b0000000000000002000000020000006c000000000000000200000002000000010000000000000004000000010000000000000001000000030000000100000000000000020000000000000003000000000000000100000001000000000000000000000001000000010000000100000000000000010000000100000001000000000000000300000028000000000000000000000000000000050000000000000009000000000000000a000000000000000800000000000000

module attributes {llzk.lang = "r1cs"} {
  r1cs.circuit @Main inputs (%arg0: !r1cs.signal {pub = #r1cs.pub}, %arg1: !r1cs.signal) {
    %0 = r1cs.def 5 : !r1cs.signal {pub = #r1cs.pub}
    %1 = r1cs.def 8 : !r1cs.signal

    %2 = r1cs.to_linear %arg0 : !r1cs.signal to !r1cs.linear
    %3 = r1cs.to_linear %arg1 : !r1cs.signal to !r1cs.linear
    %4 = r1cs.to_linear %0 : !r1cs.signal to !r1cs.linear
    %5 = r1cs.to_linear %1 : !r1cs.signal to !r1cs.linear
    %6 = r1cs.const 3 : i64 : !r1cs.linear

    %7 = r1cs.add %5, %2 : !r1cs.linear
    %8 = r1cs.add %4, %6 : !r1cs.linear
    r1cs.constrain %7, %3, %8 : !r1cs.linear

    %9 = r1cs.mul_const %2, -1 : i64 : !r1cs.linear
    %10 = r1cs.add %2, %9 : !r1cs.linear
    r1cs.constrain %10, %4, %4 : !r1cs.linear
  }
}
