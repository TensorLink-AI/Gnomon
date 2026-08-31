# Product position

Status: public positioning contract, 2026-08-12.

## The category

Gnomon is the trusted temporal execution boundary for agents.

It is not primarily a model zoo, a chat analyst, or an observability dashboard.
It is the layer between an agent's intent and any numerical temporal claim the
agent may act on. It determines what was knowable, evaluates what should
produce the answer, computes every published number, states the evidence tier,
and leaves an immutable record that can be checked and scored later.

## The first deployment wedge

The first buyer is a security-sensitive or regulated team putting agents near
consequential temporal data. Gnomon's unusual advantage there is deployable
rather than aspirational: the built-in runtime works offline, point-in-time
replay prevents later revisions from leaking backward, and integrity-sealed
artifacts retain what the agent was allowed to claim.

The first workflow inside that market is operational threshold risk:

> Which metric may cross a meaningful threshold, when, and does the evidence
> justify intervention?

This is concrete enough to test against an existing capacity, alerting, or
planning workflow. Forecast, investigate, detect, decide, and monitor are
governed views over that job, not five unrelated product promises.

The installer is usually an infrastructure, platform, data, model-risk, or
governance engineer. The operational group still owns the concrete cost of
overprovisioning, missed demand, or alert fatigue; governance is the reason an
agent is permitted into that workflow. This repository provides traceability
and offline controls, not regulatory certification.

## The promise

Gnomon owns the publication authority and provenance for every number it
returns:

- point-in-time reads make future leakage structurally unavailable;
- the exact executable candidate that earns publication produces the final
  path;
- support tiers, limitations, repairs, and recovery actions travel with the
  result;
- deterministic headlines give agents a quotable answer without asking them
  to recompute or reinterpret the artifact;
- forecasts and decisions can be scored against realised outcomes.

The governed primary is always computed by the runtime. A host may submit a
conditional scenario, but Gnomon keeps it attributed, labelled, non-primary,
and non-automatable unless independent historical evidence later admits it.

The agent continues to own intent, authorized data discovery, orchestration,
and explanation. It may quote Gnomon; it may not invent, edit, or silently
upgrade Gnomon's numerical claims.

## The deployed boundary

This repository ships a zero-dependency classical runtime with optional
governed StatsForecast candidates and local TSFM sandboxes, bitemporal data,
CLI and Python APIs, a stdio MCP server, integrity-sealed artifacts, and
outcome tracking. MCP is the sole supported structured agent-facing contract;
the packaged agent skill is usage guidance over that contract, not a second
execution surface.

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
- “Cheap” is stated only with a retained, reproducible measurement, never as
  an absolute. Provider, model, corpus, evaluated commit, harness commit, and
  replicate count must travel with any external comparison. No current
  workflow result meets that publication bar.
- “Audit-ready” may describe traceable artifacts. “Compliant” requires a named
  standard, deployment controls, and evidence outside this repository.
- Foundation models are optional candidates. Gnomon does not claim that one
  family wins universally; candidates must beat baselines on the caller's
  permitted evaluation window.

## Deployment acceptance journey

A buyer should not have to construct the governance case from implementation
details. A controlled deployment is ready for a local pilot only after it can:

1. install a checksum-verified wheel without a package index;
2. record `gnomon capabilities`, including the version, default profile, and
   explicit non-claims;
3. run a representative local series through the CLI and inspect the sealed
   JSON, Markdown, and HTML artifact;
4. make the same forecast through the packaged MCP server and preserve its
   `tier_floor`, limitations, and artifact identity in the final agent answer;
5. replay a revision-bearing dataset at an earlier `--as-of` cutoff; and
6. retain the result until actuals can score it.

CI owns the first four structural steps. The organization owns representative
data, access policy, retention, and acceptance criteria; no synthetic test can
certify those deployment controls.

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
