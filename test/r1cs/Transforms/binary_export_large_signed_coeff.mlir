// RUN: llzk-translate --r1cs-to-binary --r1cs-prime=17 %s -o %t.large
// RUN: sed 's/-340282366920938463463374607431768211457/-2/g' %s | llzk-translate --r1cs-to-binary --r1cs-prime=17 -o %t.reduced
// RUN: cmp %t.large %t.reduced

// -(2^128 + 1) is congruent to -2 modulo 17.
module attributes {llzk.lang = "r1cs"} {
  r1cs.circuit @Main inputs (%arg0: !r1cs.signal) {
    %0 = r1cs.def 2 : !r1cs.signal {pub = #r1cs.pub}
    %1 = r1cs.to_linear %arg0 : !r1cs.signal to !r1cs.linear
    %2 = r1cs.mul_const %1, -340282366920938463463374607431768211457 : !r1cs.linear
    %3 = r1cs.to_linear %0 : !r1cs.signal to !r1cs.linear
    r1cs.constrain %2, %3, %3 : !r1cs.linear
  }
}
