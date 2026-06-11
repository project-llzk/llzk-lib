// RUN: r1cs-opt --r1cs-export-binary='output-file=%t.bin prime=17' %s -o /dev/null
// RUN: od -An -v -t x1 %t.bin | tr -d ' \n' | FileCheck %s --check-prefix=NEG

// NEG: 01000000020000001000000000000000

module attributes {llzk.lang = "r1cs"} {
  r1cs.circuit @Main inputs (%arg0: !r1cs.signal) {
    %0 = r1cs.def 2 : !r1cs.signal {pub = #r1cs.pub}
    %1 = r1cs.to_linear %arg0 : !r1cs.signal to !r1cs.linear
    %2 = r1cs.mul_const %1, -1 : i64 : !r1cs.linear
    %3 = r1cs.to_linear %0 : !r1cs.signal to !r1cs.linear
    r1cs.constrain %2, %3, %3 : !r1cs.linear
  }
}
