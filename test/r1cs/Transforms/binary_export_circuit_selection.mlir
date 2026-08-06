// RUN: not llzk-translate --r1cs-to-binary --r1cs-prime=17 %s -o %t.bin 2>&1 | FileCheck %s --check-prefix=AMBIGUOUS
// RUN: llzk-translate --r1cs-to-binary --r1cs-prime=17 --r1cs-circuit-name=@Second %s -o %t.selected.bin
// RUN: test -f %t.selected.bin

// AMBIGUOUS: error: 'builtin.module' op contains multiple r1cs.circuit ops; specify '--r1cs-circuit-name' (available: @First @Second)

module attributes {llzk.lang = "r1cs"} {
  r1cs.circuit @First inputs (%arg0: !r1cs.signal) {
    %0 = r1cs.def 1 : !r1cs.signal {pub = #r1cs.pub}
    %1 = r1cs.to_linear %arg0 : !r1cs.signal to !r1cs.linear
    %2 = r1cs.to_linear %0 : !r1cs.signal to !r1cs.linear
    r1cs.constrain %1, %1, %2 : !r1cs.linear
  }

  r1cs.circuit @Second inputs (%arg0: !r1cs.signal) {
    %0 = r1cs.def 1 : !r1cs.signal {pub = #r1cs.pub}
    %1 = r1cs.to_linear %arg0 : !r1cs.signal to !r1cs.linear
    %2 = r1cs.to_linear %0 : !r1cs.signal to !r1cs.linear
    r1cs.constrain %1, %1, %2 : !r1cs.linear
  }
}
