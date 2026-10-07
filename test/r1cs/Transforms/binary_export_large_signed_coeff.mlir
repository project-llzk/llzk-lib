// RUN: llzk-translate --r1cs-to-binary --r1cs-prime=17 %s -o %t.bin
// RUN: od -An -v -t x1 %t.bin | tr -d ' \n' | FileCheck %s

// -(2^128 + 1) reduces to 15 modulo 17.
// Check the one-term linear combination: wire 2, coefficient 15.
// CHECK: 01000000020000000f00000000000000

module attributes {llzk.lang = "r1cs"} {
  r1cs.circuit @Main inputs (%arg0: !r1cs.signal) {
    %0 = r1cs.def 2 : !r1cs.signal {pub = #r1cs.pub}
    %1 = r1cs.to_linear %arg0 : !r1cs.signal to !r1cs.linear
    %2 = r1cs.mul_const %1, -340282366920938463463374607431768211457 : !r1cs.linear
    %3 = r1cs.to_linear %0 : !r1cs.signal to !r1cs.linear
    r1cs.constrain %2, %3, %3 : !r1cs.linear
  }
}
