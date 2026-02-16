# Overview {#mainpage}

LLZK is an open-source Intermediate Representation (IR) for Zero Knowledge (ZK)
circuit languages.
The LLZK project provides a flexible framework, inspired by LLVM, designed to
unify diverse ZK front-end languages and backend ZK architectures.
From an implementation perspective, the LLZK IR is a composition of multiple
[MLIR *dialects*][mlir-dialects] that represent different features that may be present
in the source ZK language.

You can read more about the motivation and design of the project [on our blog][llzk-post].
You can also view [our Ethereum Foundation grant proposal][proposal] that helped fund this project.

## Site Organization

This site contains both user documentation and internal developer documentation
for the LLZK library and related tooling.

User Documentation:
- \ref setup "Setup and Development Tips"
- \ref tools "Tool Guides"

Advanced Documentation:
- \ref dialects "LLZK Dialect Language Reference"

How to Contribute:
- First, read our \ref code-of-conduct "Code of Conduct".
- Then, read our \ref contribution-guide "Contribution Guide".

**Are you a maintainer?** If so, read the \ref maintanence "Maintenance Guide".

\subpage license "View the LLZK License."

## Versions

This site documents the current state of LLZK version \llzkVersion.
For specific release version documentation, refer to the below sub-sites:

\todo Coming soon after release.

[llzk-post]: https://medium.com/veridise/veridise-secures-ethereum-foundation-grant-to-develop-llzk-a-new-intermediate-representation-ir-224c0e71f4d5
[mlir-dialects]: https://mlir.llvm.org/docs/DefiningDialects/
[proposal]: https://drive.google.com/file/d/1tAIjAPJX5cGZT_ASFf7A2OiZaEgeWUx8/view?usp=sharing
