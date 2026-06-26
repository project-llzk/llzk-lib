; RUN: llzk-smt-check %s | FileCheck %s

(set-logic ALL)
; root: legacy_root
; check-sat stage=legacy expect=sat
(set-info :llzk-root "smt_verif_Preferred_entry")
(set-info :llzk-stage "preferred")
(set-info :status unsat)
(push 1)
(assert false)
(check-sat)
(pop 1)

; CHECK: smt_verif_Preferred_entry/preferred: unsat (expected unsat)
