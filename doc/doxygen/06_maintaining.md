# Maintenance Guide {#maintanence}

Here we document a few processes relevant to the maintainers of LLZK-lib and the
LLZK Project repositories in general.

## Tracking a New Version

We use [GitHub milestones][about-milestones] for tracking tasks associated with
a version release.

### Release version

Before creating a new release milestone, decide on the release version.

The release process uses [semantic versioning](https://semver.org/),
so the release version must match the format `v[0-9]+.[0-9]+.[0-9]+`.

## Releasing a New Version

Once a release milestone is met, the release can be created.

The general procedure is:
- Run the "Prepare release" workflow and clean up the changelog in the new pre-release branch.
- Run the "Create Release Candidate" workflow and ensure the RC is ready for release.
- Run the "Create Release PR" workflow when the RC is ready.

### Preparing a new release

A new release starts via a manually triggered GitHub workflow.

1. Go to component's GitHub repository
2. Go to `Actions` page
3. Select the `Prepare release` workflow on the left sidebar.
4. Select `Run workflow` on the top right corner of the workflow’s runs page
5. Run the workflow from the `main` branch
6. Provide the git ref used to create the release:
    - If making a new release, this is the `main` branch
    - If making a patch on an existing release, this is a fix branch
7. Provide the version of the new release
8. Click `Run workflow`. This workflow will:
    1. Create required temporary pre-release files
    2. Run scripts from `release-helpers` to update `CHANGELOG.md` from files in `/changelogs/unreleased/*.yaml`
    3. Create a `vx.y.z-pre-release` branch
    4. Commit and push changes to the new branch

From this point, any cleanup that needs to be performed (e.g., cleaning up the `CHANGELOG.md`) should
be performed by creating PRs against the aforementioned pre-release branch.

### Creating the Release Candidate

Once the changelog is updated and the necessary changes have been cherry-picked into the pre-release branch,
run the "Create Release Candidate" workflow. This creates and tags a new release candidate for the desired version.
If there are any issues that need to be addressed at this state, PRs can be opened against
this branch and new release candidates can be created by re-running the "Create Release Candidate" workflow.

### Create the Release

Once a release candidate has been tested and is ready to become a release, run the
"Create Release PR" workflow.

This workflow will create and auto-merge a PR to main that contains all the changes
from the pre-release branch and has been tagged with the release version, assuming
there are no issues.

When this PR is merged, the "Release" workflow will be called, which then creates
the GitHub release. All releases are viewable [on GitHub][llzk-releases].


[about-milestones]: https://docs.github.com/en/issues/using-labels-and-milestones-to-track-work/about-milestones
[llzk-releases]: https://github.com/project-llzk/llzk-lib/releases