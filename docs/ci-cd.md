# CI and releases

## Validation

`.github/workflows/ci.yml` runs production tests on Python 3.11, 3.12 and 3.13,
plus current evaluation-harness tests. A separate job builds locked ordinary
software and the actual Gnomon service image and checks their isolation/contracts.
No CI test authorizes paid agent or inference-service calls.

The source of version truth is `src/gnomon/product_contract.py::__version__`.
The builder, CLI, MCP and artifacts consume it. Do not set a second static
`project.version`.

## PyPI publishing

`.github/workflows/release.yml` runs only on a stable tag of the form
`vMAJOR.MINOR.PATCH` (for example `v1.2.0`). Tags carrying `.dev`, `rc` or any
other suffix do not trigger it, and a second guard inside the job rejects them.
It checks the version, runs regressions, builds wheel/sdist, checks package
metadata and proves installed-wheel operation before uploading through PyPI
Trusted Publishing.

Publish to PyPI only for user-visible changes. Intermediate work is tagged,
not published.

The publisher is bound to repository `TensorLink-AI/Gnomon`, workflow
`release.yml` and GitHub environment `pypi`. Only the publishing job receives
`id-token: write`; GitHub release creation separately receives `contents: write`.
Environment approval may still be required. Do not fall back to extracting a key
or publishing from a different account.

Stable releases use a PEP 440 version such as `1.2.0` and a matching `v1.2.0`
tag. A prerelease version needs a manual GitHub release if one is wanted; the
workflow does not build or publish it.
Publication is irreversible in the sense that a PyPI version/file cannot simply
be overwritten; use a new version for corrections.

## Checklist

1. Review the PR and documented limitations.
2. Verify production/harness tests, a fresh wheel and clean installed examples.
3. Push the exact version tag only with explicit release authorization.
4. Wait for build, publisher and any environment approval.
5. Verify the version and file hashes on PyPI and install that exact version.

Release from reviewed main so the tag, source archive and published package all
identify the same accepted commit.

## Containers

`container.yml` builds PR images and publishes on main/version tags. Registry
publication is separate from PyPI. See [containers](containers.md) and never
describe a successful build as a successful registry or package-index upload.
