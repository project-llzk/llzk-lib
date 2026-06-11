// RUN: not r1cs-opt --r1cs-export-binary='output-file=%t.bin prime=17' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=AMBIGUOUS
// RUN: r1cs-opt --r1cs-export-binary='output-file=%t.selected.bin prime=17 circuit-name=@Second' %s -o /dev/null
// RUN: test -f %t.selected.bin

// AMBIGUOUS: error: 'builtin.module' op contains multiple r1cs.circuit ops; specify 'circuit-name' (available: @First @Second)

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
