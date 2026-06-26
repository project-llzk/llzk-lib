; RUN: not llzk-smt-check %s 2>&1 | FileCheck %s

(set-logic ALL)

; check-sat stage=pre expect=sat
(check-sat)
(check-sat)

; CHECK: llzk-smt-check: stage annotation count (1) does not match check-sat count (2)
