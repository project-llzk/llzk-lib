// RUN: r1cs-opt --r1cs-export-binary='output-file=%t.bin prime=17' %s -o /dev/null
// RUN: dd if=%t.bin bs=1 skip=24 count=40 2>/dev/null | od -An -v -t x1 | tr -d ' \n' | FileCheck %s --check-prefix=HEADER
// RUN: tail -c 40 %t.bin | od -An -v -t x1 | tr -d ' \n' | FileCheck %s --check-prefix=WIREMAP

// HEADER: 080000001100000000000000050000000100000001000000010000000b0000000000000002000000
// WIREMAP: 0000000000000000050000000000000009000000000000000a000000000000000800000000000000

module attributes {llzk.lang = "r1cs"} {
  r1cs.circuit @Main inputs (%arg0: !r1cs.signal {pub = #r1cs.pub}, %arg1: !r1cs.signal) {
    %0 = r1cs.def 5 : !r1cs.signal {pub = #r1cs.pub}
    %1 = r1cs.def 8 : !r1cs.signal

    %2 = r1cs.to_linear %arg0 : !r1cs.signal to !r1cs.linear
    %3 = r1cs.to_linear %arg1 : !r1cs.signal to !r1cs.linear
    %4 = r1cs.to_linear %0 : !r1cs.signal to !r1cs.linear
    %5 = r1cs.to_linear %1 : !r1cs.signal to !r1cs.linear
    %6 = r1cs.add %5, %2 : !r1cs.linear
    r1cs.constrain %6, %3, %4 : !r1cs.linear
    r1cs.constrain %2, %2, %4 : !r1cs.linear
  }
}
