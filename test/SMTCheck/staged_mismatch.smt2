; RUN: not llzk-smt-check %s 2>&1 | FileCheck %s

(set-logic ALL)

(set-info :llzk-root "smt_verif_Sample_entry")
(set-info :llzk-stage "pre")
(set-info :status unsat)
(check-sat)

; CHECK: smt_verif_Sample_entry/pre: sat (expected unsat)
; CHECK: llzk-smt-check: stage result mismatch:
; CHECK-NEXT: smt_verif_Sample_entry/pre: got sat, expected unsat
