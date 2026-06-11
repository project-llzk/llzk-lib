// RUN: not r1cs-opt --r1cs-export-binary='output-file=%t.bin' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISSING-PRIME
// RUN: not r1cs-opt --r1cs-export-binary='prime=17' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISSING-OUTPUT
// RUN: not r1cs-opt --r1cs-export-binary='output-file=%t.bin prime=17 circuit-name=@Missing' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISSING-CIRCUIT
// RUN: not r1cs-opt --r1cs-export-binary='output-file=%t.bin prime=abc' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=BAD-PRIME
// RUN: not r1cs-opt --r1cs-export-binary='output-file=%t.bin prime=1' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=SMALL-PRIME

// MISSING-PRIME: error: 'builtin.module' op R1CS binary export requires a non-empty 'prime' option
// MISSING-OUTPUT: error: 'builtin.module' op R1CS binary export requires a non-empty 'output-file'
// MISSING-CIRCUIT: error: 'builtin.module' op could not find r1cs.circuit @Missing
// BAD-PRIME: error: 'builtin.module' op 'prime' must be a base-10 integer
// SMALL-PRIME: error: 'builtin.module' op 'prime' must be greater than 1

module attributes {llzk.lang = "r1cs"} {
  r1cs.circuit @Main inputs (%arg0: !r1cs.signal) {
    %0 = r1cs.def 1 : !r1cs.signal {pub = #r1cs.pub}
    %1 = r1cs.to_linear %arg0 : !r1cs.signal to !r1cs.linear
    %2 = r1cs.to_linear %0 : !r1cs.signal to !r1cs.linear
    r1cs.constrain %1, %1, %2 : !r1cs.linear
  }
}
