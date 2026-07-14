; RUN: not llzk-smt-check %s 2>&1 | FileCheck %s

(set-logic ALL)

(set-info :llzk-stage "pre")
(set-info :status sat)
(check-sat)
(check-sat)

; CHECK: llzk-smt-check: check metadata count (1) does not match check-sat count (2)
