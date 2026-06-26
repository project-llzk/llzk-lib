# llzk-scalar-to-smt restructuring

We need to restructure the SMT output of the verif to smt pass so that it can
be more easily consumed and used downstream.

## Includes function breakdown

Currently, a contract broken down into several components, including a "includes"
function that contains logic for all verif.include operations in the contract.

Instead of aggregating these into a single "includes" function, a unique "include" function
should be created for each `verif.include` operation in the contract. This will allow
for better composition in the next task.

## Includes function logic changes

Currently, the "includes" function computes "(preconditions && target_semantics) => postconditions".
This is not precisely what we want to prove, as we want to demonstrate:
- The preconditions hold
- Given the preconditions, the target semantics hold.
- Given the preconditions and the target semantics, the postconditions hold

I believe to prove this sequence of information, we want to use some more advanced SMT operations
(you can use this page https://mlir.llvm.org/docs/Dialects/SMT/ for reference).

In particular, I believe that in the includes function, we want to do the following sequence of operations:

- check satisfiability of preconditions given caller-given context
- if the preconditions are satisfied, push a solver context and assert the preconditions and the target semantics
- check satisfiability of postconditions given the preconditions and target semantics
- if satisfied, pop the context and assert preconditions, target semantics, and postconditions in the caller context

## Full Contract Functions

Currently, we break each contract into:
- A precondition function
- Functions for verif.include ops
- A postcondition function

We should create a high-level function for each contract that treats it like the entry point. This function would:

- Assert the preconditions (assume they hold)
- check if the target semantics hold (check if satisfiable given the preconditions). If so, assert them
- check if the postconditions hold given the preconditions and semantics.
