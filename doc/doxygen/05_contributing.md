# Contribution Guide {#contribution-guide}

\tableofcontents

Thank you for investing your time in contributing to the LLZK project!

Before getting involved, read our \subpage code-of-conduct "Code of Conduct" to keep our community approachable and respectable.

In this guide you will get an overview of our overarching philosophy, as well as the contribution workflow; from opening an issue, creating a PR, reviewing, and merging the PR.

## Contribution Philosophy

To keep the LLZK codebase maintainable and approachable, we follow these guiding principles for all contributions:

1. **Testing, testing, testing.** Every PR should come with thorough tests (especially lit tests) that demonstrate the behavior of the new feature/changes. Good tests make your PR easier to review and maintain, as tests serve as both demonstrations of correctness and working examples.
2. **Document thoroughly.** New features should be documented clearly (e.g., with TableGen docs for passes). Documentation should describe both the *intent* and the *operation* of the feature.
3. **Small PRs are better.** Large PRs are difficult to review. We would prefer to review a sequence of smaller PRs that gradually extend functionality rather than a single large PR. For example, start with a PR handling the simplest cases (with more complex cases as "TODOs"), and then extend support in subsequent PRs.
4. **Readable over clever.** Code should be simple and understandable. If you must choose between a highly optimized solution and a readable one, choose readability. Optimizations can always come later.
5. **Avoid overengineering.** Avoid adding generalized abstractions until they're needed. For example, avoid templates or callback-based designs if only one specialization or behavior exists. Over-generalizing makes code harder to understand and maintain.

## Getting started

### Issues

#### Create a new issue

If you spot a problem, encounter a bug, or want to request a new feature, first
[search for an existing issue](https://github.com/project-llzk/llzk-lib/issues).
Please contribute to an existing issue when possible instead of opening a duplicate.

If a related issue does not exist, open a new issue using the relevant
[issue form](https://github.com/project-llzk/llzk-lib/issues/new/choose). A new
issue must clearly describe and motivate the bug or feature, including its
impact or use case, expected behavior, and enough detail for a core developer
to evaluate it.

New issues begin with the `under review` label. This means the proposal
is still being evaluated or refined. A maintainer may ask for more
information, narrow the scope, suggest an alternative, or decline the proposal.
Do not begin implementation while an issue is under review.

A maintainer will replace that label with `approved` when the issue is
sufficiently motivated, appropriately scoped, and suitable for implementation.
Approval does not assign the issue to a contributor or guarantee that a
resulting PR will be merged.

#### Solve an issue

External contributors should start by scanning our
[existing issues](https://github.com/project-llzk/llzk-lib/issues) for one that
interests them. Issues labeled `approved` are ready for
implementation; issues also labeled `good first issue` are intended to be
approachable for newcomers. You can narrow the search using labels as filters.

As a general rule, we do not assign issues. Before starting work, comment on the
issue to share your intended approach and coordinate with other contributors.
Every external contribution must resolve one or more existing issues, and those
issues must be marked `approved` before implementation begins.


### Make Changes

1. Fork the repository.
  - Using GitHub Desktop:
    - [Getting started with GitHub Desktop](https://docs.github.com/en/desktop/installing-and-configuring-github-desktop/getting-started-with-github-desktop) will guide you through setting up Desktop.
    - Once Desktop is set up, you can use it to [fork the repo](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/cloning-and-forking-repositories-from-github-desktop)!
  - Using the command line:
    - [Fork the repo](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo#fork-an-example-repository) so that you can make your changes without affecting the original project until you're ready to merge them.
2. Install or update LLZK development dependencies. For more information, see \ref setup "the development guide"
3. Create a working branch and start with your changes!

### Commit your update

Commit the changes once you are happy with them.
Remember that all code must pass a format check before it can be merged, so consider running clang format
before pushing your commits. For more information, see \ref dev-workflow "the development workflow section".

### Pull Request

When you are finished with the changes, create a _pull request_, also known as a
PR. External PRs must link to the approved issue or issues they resolve.

#### Best Practices Checklist

Before submitting your PR, ensure you have:

- [ ] Added lit and/or unit tests covering all relevant cases.
- [ ] Updated TableGen and/or other documentation to describe new features.
- [ ] Added a changelog. All PRs require a changelog describing what user-level changes have been made in the PR. To create a template changelog, run `create-changelog` from the nix shell.
- [ ] [Linked the PR to the existing issue or issues it resolves](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue). For external contributions, each issue must be marked `approved`.
- [ ] Disclosed whether AI tools contributed to the PR. If they did, identified the tools, described how they contributed, and explained how their output was verified.
- [ ] Enabled the checkbox to [allow maintainer edits](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/allowing-changes-to-a-pull-request-branch-created-from-a-fork) so the branch can be updated for a merge.

#### Pull Request Review

Once you submit your PR, a maintainer will first review its linked issues. We
may defer implementation review until every issue addressed by an external PR
is marked `approved`. If the motivation or scope is unclear, we will continue
that discussion on the issue before reviewing the implementation.

After the linked issues are approved, a maintainer will review the PR. We
may ask questions or request additional information.
- We may ask for changes to be made before a PR can be merged, either using [suggested changes](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/incorporating-feedback-in-your-pull-request) or pull request comments.
You can apply suggested changes directly through the UI.
You can make any other changes in your fork, then commit them to your branch.
- As you update your PR and apply changes, mark each conversation as [resolved](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/commenting-on-a-pull-request#resolving-conversations).
- If you run into any merge issues, checkout this [git tutorial](https://github.com/skills/resolve-merge-conflicts) to help you resolve merge conflicts and other issues.

### Your PR is merged!

Congratulations! The LLZK team thanks you.

Once your PR is merged, your contributions will be publicly available in the [LLZK-lib repository](https://github.com/project-llzk/llzk-lib).

<div class="section_buttons">
| Previous          |                              Next |
|:------------------|----------------------------------:|
| \ref tools | \ref dialects |
</div>
