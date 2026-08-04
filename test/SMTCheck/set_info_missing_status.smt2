; RUN: not llzk-smt-check %s 2>&1 | FileCheck %s

(set-logic ALL)
(set-info :llzk-stage "pre")
(check-sat)

; CHECK: llzk-smt-check: missing expected result before check-sat for stage 'pre'
