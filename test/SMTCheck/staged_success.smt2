; RUN: llzk-smt-check %s | FileCheck %s

(set-logic ALL)

; root: smt_verif_Sample_entry
; check-sat stage=pre expect=unsat
(push 1)
(assert false)
(check-sat)
(pop 1)

; check-sat stage=target expect=unsat
(push 1)
(assert (= 0 1))
(check-sat)
(pop 1)

; check-sat stage=post expect=unsat
(push 1)
(assert (and true false))
(check-sat)
(pop 1)

; CHECK: smt_verif_Sample_entry/pre: unsat (expected unsat)
; CHECK-NEXT: smt_verif_Sample_entry/target: unsat (expected unsat)
; CHECK-NEXT: smt_verif_Sample_entry/post: unsat (expected unsat)
