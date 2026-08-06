// RUN: not llzk-translate --r1cs-to-binary %s -o %t.bin 2>&1 | FileCheck %s --check-prefix=MISSING-PRIME
// RUN: not llzk-translate --r1cs-to-binary --r1cs-prime=17 --r1cs-circuit-name=@Missing %s -o %t.bin 2>&1 | FileCheck %s --check-prefix=MISSING-CIRCUIT
// RUN: not llzk-translate --r1cs-to-binary --r1cs-prime=abc %s -o %t.bin 2>&1 | FileCheck %s --check-prefix=BAD-PRIME
// RUN: not llzk-translate --r1cs-to-binary --r1cs-prime=1 %s -o %t.bin 2>&1 | FileCheck %s --check-prefix=SMALL-PRIME

// MISSING-PRIME: error: 'builtin.module' op R1CS binary export requires a non-empty '--r1cs-prime' option
// MISSING-CIRCUIT: error: 'builtin.module' op could not find r1cs.circuit @Missing
// BAD-PRIME: error: 'builtin.module' op '--r1cs-prime' must be a base-10 integer
// SMALL-PRIME: error: 'builtin.module' op '--r1cs-prime' must be greater than 1

module attributes {llzk.lang = "r1cs"} {
  r1cs.circuit @Main inputs (%arg0: !r1cs.signal) {
    %0 = r1cs.def 1 : !r1cs.signal {pub = #r1cs.pub}
    %1 = r1cs.to_linear %arg0 : !r1cs.signal to !r1cs.linear
    %2 = r1cs.to_linear %0 : !r1cs.signal to !r1cs.linear
    r1cs.constrain %1, %1, %2 : !r1cs.linear
  }
}
