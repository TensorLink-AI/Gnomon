# CI/CD and release operations

## Continuous integration

`.github/workflows/ci.yml` runs on pushes to `main`, pull requests, and manual
dispatch. It:

- tests Python 3.11, 3.12, and 3.13;
- compiles source and tests;
- builds the source distribution and wheel;
- installs the wheel in the runner;
- smoke-tests the installed CLI; and
- uploads distributions as a workflow artifact.

Enable branch protection for `main` and require the CI jobs before merging.

## PyPI releases

`.github/workflows/release.yml` runs only for `v*` tags. It verifies that the tag
matches `project.version`, builds and smoke-tests distributions, and publishes
through PyPI Trusted Publishing. No long-lived PyPI token is stored in GitHub.

One-time configuration is still required:

1. Create or reserve the `headwater-forecast` project on PyPI.
2. Add a GitHub Actions trusted publisher with owner `TensorLink-AI`, repository
   `headwater`, workflow `release.yml`, and environment `pypi`.
3. Create a GitHub environment named `pypi`; adding required reviewers is
   recommended.
4. Update `project.version`, merge it, and create a matching tag such as
   `v0.1.0`.

The publishing job alone receives `id-token: write`; all other jobs use
read-only repository permissions.

## Container delivery

`.github/workflows/container.yml` builds the Dockerfile for pull requests and
publishes images to GitHub Container Registry on `main` and version tags. It
uses Docker Buildx caching, provenance, and SBOM generation.

GitHub's `GITHUB_TOKEN` supplies short-lived registry authentication. The
repository or organization must allow Actions to write packages.

## Release checklist

1. Update the package version and release notes.
2. Open a pull request and wait for CI and container builds.
3. Merge to protected `main`.
4. Create and push the matching signed or annotated version tag.
5. Approve the `pypi` environment deployment if required.
6. Verify the PyPI files, attestations, installed CLI, and GHCR image.

