# Gnomon documentation

- [Discovery and listings](discovery-and-listings.md): task discovery, registry metadata and publication prerequisites.

Start with the task you need to complete. The [main agent workflow](../skills/use-gnomon/SKILL.md)
covers tool use; specialist skills add Python, ledger, trading or connector details.
These pages describe the checkout. For an installed release, check `gnomon environment`
and discover the current schemas through `gnomon capabilities` or MCP `tools/list`.

- [Shared hosting and Ditto](hosting/README.md)


## Start and connect

| Need | Guide |
| --- | --- |
| Understand the product | [README](../README.md) |
| Install in the right Python environment | [Installation](installation.md) |
| Make a first forecast and score its outcome | [First run](getting-started.md) |
| Connect optional Ephemeris models | [Secure connection setup](ephemeris-onboarding.md) |
| Connect an agent over stdio | [MCP quickstart](quickstart-mcp.md) |
| Choose/install an agent skill | [Skill directory](agent-skill.md) |
| Read the common result overview and discover exact schemas | [Agent operations](agent-operations.md) |

## Work with data and models

- [Data format and time semantics](data-format.md): columns, units, snapshots and revisions.
- [Python API](python-api.md) and [CLI reference](cli-reference.md): executable interfaces.
- [Provider integration and evaluation](production/INFERENCE.md): custom models, configuration and budgets.
- [Complete local evidence workflow](local-evidence-workflow.md): inspect, compare, route and score.
- [MCP evidence workflow](mcp-evidence-workflow.md): the same work over a persistent stdio session.
- [Adaptive routing](adaptive-routing.md): configured model selection using recorded outcomes and episodic memory.
- [Revised-vintage rescoring](revised-vintage-workflow.md): score saved predictions against revised actuals.
- [Final-answer preservation](final-selection.md): bind an agent's answer to the forecast it actually ran.

## Review outcomes and remember evidence

- [Ledger skill](../skills/use-gnomon-ledger/SKILL.md): find, compare and review recorded evidence.
- [Ledger operations](production/OPERATIONS.md): persistence, permissions and backup.
- [Prospective comparison example](production-history-comparison.md): matched production forecasts.
- [Scoring and recovery](scoring-and-recovery.md): pending/partial/complete outcomes, coverage and errors.
- [Decision memory](decision-memory.md): recorded rationale, outcome reviews and versioned lessons.
- [Memory bridge and recall](memory-bridge.md): bounded read-only evidence for an agent's context.
- [Hermes integration](hermes-ledger.md): optional host integration and automatic recall.

Router episodic memory selects among configured forecasting models. Agent memory
recalls evidence and hypotheses for interpretation. Neither requires an LLM judge,
and storing a lesson does not establish that its explanation is correct.

## Operate and troubleshoot

- [Troubleshooting](troubleshooting.md): locate the right schema or recovery path.
- [Date/time utilities](production/TEMPORAL.md): explicit calendar and interval calculations.
- [Containers](containers.md) and [offline installation](offline-installation.md).
- [Validation and limitations](agent-evaluation.md): what checks establish and what remains unverified.

## Maintainers and historical records

Use the guides above for current operation. The following records explain development
and past acceptance work; their dated counts and checkpoints are not live configuration.

- [Development](development.md) · [CI and release process](ci-cd.md) · [Changelog](../CHANGELOG.md).
- [Production delivery plan](production/PLAN.md) · [Delivery checkpoint](production/HANDOFF.md).
- [Agent feedback](agent-feedback-status.md) · [1.1.6 feedback](feedback-116-disposition.md) · [1.1.7 feedback](feedback-117-disposition.md).
- [Development notes on comparison metrics](ledger-comparison-metrics.md).
