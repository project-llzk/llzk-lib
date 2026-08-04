; RUN: not llzk-smt-check %s 2>&1 | FileCheck %s

(set-logic ALL)
(set-info :status maybe)
(check-sat)

; CHECK: llzk-smt-check: invalid set-info annotation: '(set-info :status maybe)'
