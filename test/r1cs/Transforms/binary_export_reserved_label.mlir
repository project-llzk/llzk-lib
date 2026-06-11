// RUN: not r1cs-opt --r1cs-export-binary='output-file=%t.bin prime=17' %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: 'r1cs.def' op label 0 is reserved for the implicit one wire in .r1cs

module attributes {llzk.lang = "r1cs"} {
  r1cs.circuit @Main inputs (%arg0: !r1cs.signal) {
    %0 = r1cs.def 0 : !r1cs.signal {pub = #r1cs.pub}
    %1 = r1cs.to_linear %arg0 : !r1cs.signal to !r1cs.linear
    %2 = r1cs.to_linear %0 : !r1cs.signal to !r1cs.linear
    r1cs.constrain %1, %1, %2 : !r1cs.linear
  }
}
