; RUN: llzk-smt-check %s | FileCheck %s

(set-logic ALL)
(set-info :status sat)
(check-sat)

; CHECK: check[0]: sat (expected sat)
