; RUN: llzk-smt-check %s | FileCheck %s

(set-logic ALL)
(set-info :source "demo")
(set-info :llzk-stage "pre")
(set-info :status sat)
(check-sat)

; CHECK: pre: sat (expected sat)
