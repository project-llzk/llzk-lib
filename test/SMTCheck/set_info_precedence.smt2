; RUN: llzk-smt-check %s | FileCheck %s

(set-logic ALL)
(set-info :llzk-root "smt_verif_Preferred_entry")
(set-info :llzk-root "ignored_root")
(set-info :llzk-stage "ignored")
(set-info :status sat)
(set-info :llzk-stage "preferred")
(set-info :status unsat)
(push 1)
(assert false)
(check-sat)
(pop 1)

; CHECK: ignored_root/preferred: unsat (expected unsat)
