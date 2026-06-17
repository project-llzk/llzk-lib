; RUN: not llzk-smt-check %s 2>&1 | FileCheck %s

(set-logic ALL)

; check-sat stage=pre expect=unsat
(check-sat)

; CHECK: pre: sat (expected unsat)
; CHECK: llzk-smt-check: stage result mismatch:
; CHECK-NEXT: pre: got sat, expected unsat
