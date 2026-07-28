; RUN: not llzk-smt-check %s 2>&1 | FileCheck %s

(set-logic ALL)
(assert (= 0 0)
(check-sat)

; CHECK: llzk-smt-check: solver exited with code
