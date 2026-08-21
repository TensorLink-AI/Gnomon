# Product position

Status: public positioning contract, 2026-08-12.

## The category

Gnomon is the trusted temporal execution boundary for agents.

It is not primarily a model zoo, a chat analyst, or an observability dashboard.
It is the layer between an agent's intent and any numerical temporal claim the
agent may act on. It determines what was knowable, evaluates what should
produce the answer, computes every published number, states the evidence tier,
and leaves an immutable record that can be checked and scored later.

## The first job

The beachhead is operational threshold risk:

> Which metric may cross a meaningful threshold, when, and does the evidence
> justify intervention?

This is concrete enough to test against an existing capacity, alerting, or
planning workflow. Forecast, investigate, detect, decide, and monitor are
governed views over that job, not five unrelated product promises.

The installer is usually an infrastructure, platform, or data engineer. The
first paying team, if a hosted model backend is used, is the operational group
that already pays for forecast maintenance, overprovisioning, missed demand,
or alert fatigue. Audit and compliance can strengthen an enterprise case, but
they are not the default pitch and this repository does not claim regulatory
certification.

## The promise

Gnomon owns every number it returns:

- point-in-time reads make future leakage structurally unavailable;
- the exact executable candidate that earns publication produces the final
  path;
- support tiers, limitations, repairs, and recovery actions travel with the
  result;
- deterministic headlines give agents a quotable answer without asking them
  to recompute or reinterpret the artifact;
- forecasts and decisions can be scored against realised outcomes.

The agent continues to own intent, authorized data discovery, orchestration,
and explanation. It may quote Gnomon; it may not invent, edit, or silently
upgrade Gnomon's numerical claims.

## The deployed boundary

This repository ships a zero-dependency classical runtime with optional local
TSFM sandboxes, bitemporal data, CLI and Python APIs, a stdio MCP server,
immutable artifacts, and outcome tracking. MCP is the sole agent-facing
contract.

It does not ship a hosted inference service, a live benchmark service, an
enterprise artifact store, or a model-training network. Those can be adjacent
parts of a broader Cascade stack, but Gnomon must remain useful without them.
If a hosted router or future model backend is configured, it supplies a
candidate; it does not inherit authority over publication or support.

## Claims discipline

Public claims follow the same rule as forecast claims: measured or absent.

- The leakage result may be quoted with its population and method: Gnomon
  leaked on 0/40 trap tasks; the LLM control leaked on 13/35 answered tasks.
- “One trusted answer” describes the response contract, not guaranteed
  forecast accuracy.
- “Local-first” means built-in execution needs no network or API key; optional
  downloads and configured remote backends must be named as such.
- “Cheap” is stated with its measurement, never as an absolute. No
  workflow-cost measurement is currently citable: the raw output of the
  2026-08 workflow experiment was not preserved, so its per-case figures are
  withdrawn. Provider, model, corpus, replicate count, and a committed
  artifact must travel with any future cost claim.
- “Audit-ready” may describe traceable artifacts. “Compliant” requires a named
  standard, deployment controls, and evidence outside this repository.
- Foundation models are optional candidates. Gnomon does not claim that one
  family wins universally; candidates must beat baselines on the caller's
  permitted evaluation window.

## Product metrics

Engineering completion is necessary but not evidence of demand. The product
should be evaluated on:

1. first successful run on non-demo data;
2. answer yield and accuracy, split by evidence tier;
3. leakage, misquotation, and caveat-survival rates;
4. cumulative conversation tokens and calls per completed task;
5. forecasts later scored against actuals and week-four retained projects;
6. conversion from local adoption to a paid backend or managed history, if
   those products are offered.

These metrics keep the strategy attached to a problem people already pay to
solve, rather than treating installs, tools, or benchmark wins as the product.
