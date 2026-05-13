# AGENTS.md

This repository expects coding agents to use `nix` for builds and command execution so all necessary dependencies are available.

## Build and command guidance

- Preferred build command: `nix build -L`
- When you need to run a project command inside the development environment, use:
  `nix develop --command bash -c "[command]"`

## Completion requirement

- Always ensure the build is successful before you stop working.
- If you make changes, finish by running a successful build with `nix`.
