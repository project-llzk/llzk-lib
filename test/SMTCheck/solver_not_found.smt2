; RUN: not llzk-smt-check %s --solver-binary=does-not-exist 2>&1 | FileCheck %s

(set-logic ALL)
(check-sat)

; CHECK: llzk-smt-check: failed to find solver binary 'does-not-exist'
