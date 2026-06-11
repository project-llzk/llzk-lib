// RUN: not r1cs-opt --r1cs-export-binary='output-file=%t.bin prime=17' %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: 'builtin.unrealized_conversion_cast' op cannot be exported as a .r1cs linear combination; expected one of r1cs.to_linear, r1cs.const, r1cs.add, r1cs.mul_const, or r1cs.neg

module attributes {llzk.lang = "r1cs"} {
  r1cs.circuit @Main inputs (%arg0: !r1cs.signal) {
    %0 = r1cs.def 1 : !r1cs.signal {pub = #r1cs.pub}
    %1 = r1cs.to_linear %arg0 : !r1cs.signal to !r1cs.linear
    %2 = builtin.unrealized_conversion_cast %1 : !r1cs.linear to !r1cs.linear
    %3 = r1cs.to_linear %0 : !r1cs.signal to !r1cs.linear
    r1cs.constrain %2, %1, %3 : !r1cs.linear
  }
}
