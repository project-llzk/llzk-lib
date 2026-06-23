; RUN: not llzk-smt-check %s 2>&1 | FileCheck %s

(set-logic ALL)

; root: smt_verif_Sample_entry
; check-sat stage=pre expect=unsat
(check-sat)

; CHECK: smt_verif_Sample_entry/pre: sat (expected unsat)
; CHECK: llzk-smt-check: stage result mismatch:
; CHECK-NEXT: smt_verif_Sample_entry/pre: got sat, expected unsat
