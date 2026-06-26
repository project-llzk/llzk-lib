; RUN: llzk-smt-check %s | FileCheck %s

(set-logic ALL)
(set-info :llzk-root "smt_verif_CheckGate_entry")
(set-info :llzk-stage "target")
(set-info :status unsat)
(push 1)
(assert false)
(check-sat)
(pop 1)

(reset)
(set-logic ALL)
(set-info :llzk-root "smt_verif_CheckGate_entry")
(set-info :llzk-stage "post")
(set-info :status unsat)
(push 1)
(assert (= 0 1))
(check-sat)
(pop 1)

; CHECK: smt_verif_CheckGate_entry/target: unsat (expected unsat)
; CHECK-NEXT: smt_verif_CheckGate_entry/post: unsat (expected unsat)
