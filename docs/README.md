# Gnomon documentation

Gnomon provides explicit time-series calculations, provider-neutral forecasting and
revision-aware evidence. Start with an offline baseline, plug in your model or an
optional remote connector, and add a ledger only
when you need persistent history and scoring.

## Start here

1. [Install](installation.md) and [run a forecast](getting-started.md).
2. [Connect an MCP agent](quickstart-mcp.md) or use the [Python API](python-api.md).
3. [Configure providers](production/INFERENCE.md), including Ephemeris and user callables.
4. [Operate the ledger](production/OPERATIONS.md) or use [temporal calculations](production/TEMPORAL.md).

## Reference

| Guide | Covers |
| --- | --- |
| [CLI](cli-reference.md) | Commands, with advanced legacy workflows identified. |
| [Data format](data-format.md) | Input formats, timestamps, quantities and grids. |
| [Concepts](concepts.md) | Inference, evaluation, time and action boundaries. |
| [Results](results-and-artifacts.md) | Stored outputs and advanced evaluated artifacts. |
| [Covariates](covariates.md) | Additional series and advanced enrichment. |
| [Publication modes](publication-modes.md) | Retained advanced evaluated-workflow modes. |
| [Troubleshooting](troubleshooting.md) | Errors and unsupported requests. |
| [Agent skill](agent-skill-and-feedback.md) | Packaged guidance and optional feedback. |
| [Offline installation](offline-installation.md) | Network-free installation and execution. |
| [Containers](containers.md) | Container use and isolation limits. |
| [Validation](agent-evaluation.md) | Tested behavior and unmeasured claims. |
| [Development](development.md) | Tests, layout and contributions. |
| [Releases](ci-cd.md) | CI, versioning and PyPI publishing. |

Regular subdaily grids use `<N>s`, `<N>min`, or `<N>h`; see the data guide for
daily, weekly and monthly conventions. Do not invent a calendar for indexed data.

The [delivery checkpoint](production/HANDOFF.md) records outstanding acceptance
work. Removed proposals and experiments remain in Git history, not current docs.
