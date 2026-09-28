// RUN: llzk-opt %s -o /dev/null
// RUN: llzk-translate --r1cs-to-binary --r1cs-prime=17 %s -o %t.r1cs
// RUN: test -s %t.r1cs
// No public input or output is needed for a private satisfiability statement.
module attributes {llzk.lang = "r1cs"} {
  r1cs.circuit @Private inputs (%input: !r1cs.signal) {
    %output = r1cs.def 1 : !r1cs.signal
    %a = r1cs.to_linear %input : !r1cs.signal to !r1cs.linear
    %b = r1cs.to_linear %output : !r1cs.signal to !r1cs.linear
    r1cs.constrain %a, %a, %b : !r1cs.linear
  }
}
