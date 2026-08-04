; RUN: llzk-smt-check %s | FileCheck %s

(set-logic ALL)
(check-sat)
(push 1)
(assert false)
(check-sat)
(pop 1)

; CHECK: check[0]: sat
; CHECK-NEXT: check[1]: unsat
