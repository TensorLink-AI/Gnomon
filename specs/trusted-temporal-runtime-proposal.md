# Proposal: Gnomon as the trusted temporal runtime for agents

**Status:** Proposal  
**Date:** 2026-08-11  
**Scope:** Product direction, public surface, architecture, and migration

## Executive summary

Gnomon's strongest idea is not that it can perform every temporal task. Its
strongest idea is that an agent may frame and explain a temporal question,
while a deterministic runtime remains authoritative for data visibility,
model evaluation, forecast values, uncertainty, support, and provenance.

This proposal narrows Gnomon around that advantage:

> Give Gnomon time-series data and a question. It returns the strongest
> defensible forecast package, or explains precisely why it cannot.

The product should become a small, difficult-to-misuse numerical authority
behind agents. Forecasting remains the numerical center. Investigation,
monitoring, and decision support become governed views over the same run
contract instead of independently growing subsystems.

The proposed default agent surface has three tools:

1. `gnomon_inspect` diagnoses data and determines what questions are
   supportable.
2. `gnomon_run` compiles a typed temporal question into one verified execution.
3. `gnomon_track` closes the outcome loop with actuals, performance, coverage,
   and regret.

Low-level tools and experimental modelling remain available in expert and
research profiles, but they do not enlarge the default trusted path.

## 1. Problem statement

Gnomon currently spans forecasting, investigation, anomaly detection,
decisions, monitoring, temporal storage, tracking, model routing, experiment
planning, TSFM management, and several agent integration surfaces. Much of
this work is valuable, but the breadth creates four product risks.

### 1.1 The differentiator becomes difficult to see

The important promise is that Gnomon owns every published number and can prove
what was knowable when it was computed. A large feature surface makes Gnomon
look like a general temporal analytics suite, where it competes on breadth and
model count rather than trustworthiness.

### 1.2 Safety can displace usefulness

Leakage prevention, calibrated uncertainty, disclosed degradation, and honest
abstention are necessary. They are not sufficient if the system rarely
answers, costs too much to operate, or does not improve the decisions made
from its output.

Gnomon must measure safety and usefulness together:

- completion rate;
- error conditional on answering;
- total utility with abstention priced;
- interval coverage and width;
- decision regret;
- leakage and misquotation rates;
- latency and token cost.

### 1.3 The agent must understand too much machinery

An agent should not need to discover and sequence a large collection of
forecast, monitor, decide, route, artifact, tracking, and repair tools. The
normal workflow should be one typed execution after optional inspection.

### 1.4 Features can cross implementation boundaries inconsistently

The highest-risk defects tend to occur where one component hands authority to
another: configuration to evaluation, evaluation to final prediction, runtime
to artifact, or environment to persistent state. The architecture should make
it structurally difficult to evaluate one executable forecast and publish
another.

## 2. Proposed product identity

Gnomon should be positioned as:

> A leakage-resistant, evidence-producing temporal runtime that gives agents
> defensible forecasts and prevents unsupported numerical claims.

It should not present itself primarily as:

- a general-purpose temporal intelligence agent;
- a model marketplace;
- an autonomous data-science system;
- a complete anomaly, monitoring, or optimization platform;
- a guarantee that complicated models outperform strong baselines.

The desired relationship is:

```text
agent discovers intent and explains the result
                         |
                         v
Gnomon validates, evaluates, computes, verifies, and records
                         |
                         v
immutable evidence-linked answer with an explicit support state
```

## 3. Product principles

### 3.1 One numerical authority

Every published number must come from the executable candidate that earned its
support during evaluation. Model name, member set, weights, strategy, package
versions, weight revisions, and fallback policy are part of that executable's
identity.

### 3.2 One run contract

Forecast, threshold risk, monitoring, and decision support should compile into
one execution graph and one support contract. A downstream view cannot increase
the support earned by the forecast evidence it consumes.

### 3.3 Compact by default, complete by reference

Normal responses should contain enough information to answer safely without a
second tool call. Large arrays, full lineage, and diagnostics belong in the
immutable artifact referenced by the response.

### 3.4 Fewer controls with stronger semantics

Every accepted option must alter behavior, evidence, or output as documented.
Unsupported options fail at load time. Missing configuration dependencies fail
loudly whenever a configuration file exists.

### 3.5 Experimental work cannot silently enter the trusted path

Research features may be available and measurable without being enabled in the
default product. Graduation requires explicit evidence and end-to-end contract
tests.

## 4. Proposed public surface

### 4.1 `gnomon_inspect`

`gnomon_inspect` answers:

- What temporal data is present?
- Is its timestamp geometry valid?
- What repairs would be required?
- Which columns and frequencies are unambiguous?
- What horizons are evaluable?
- Which question types can the history support?

Inspection performs no forecast and writes no consequential tracking state.

Example:

```json
{
  "input": "requests.csv",
  "target": "requests"
}
```

Response:

```json
{
  "status": "valid",
  "frequency": "D",
  "observations": 180,
  "supportable_horizons": {
    "fully_evaluated": 14,
    "limited": 30
  },
  "repairs": [],
  "suggested_questions": [
    "forecast",
    "threshold_risk",
    "change_investigation"
  ]
}
```

### 4.2 `gnomon_run`

`gnomon_run` is the primary execution tool. It accepts observations or a
bitemporal dataset reference plus one typed question.

Example:

```json
{
  "data": "store:api_requests",
  "as_of": "2026-08-11T00:00:00+10:00",
  "question": {
    "kind": "threshold_risk",
    "target": "requests",
    "horizon": 14,
    "threshold": 340
  },
  "policy": {
    "minimum_support": "best_effort"
  }
}
```

Internally this compiles to:

```text
resolve point-in-time snapshot
    -> validate and disclose repair
    -> construct admissible candidates
    -> rolling selection
    -> held-out calibration
    -> report-only test
    -> final refit of the selected executable
    -> question-specific calculation
    -> deterministic verification
    -> immutable artifact
```

The agent does not manually sequence forecast, monitor, decide, explain, and
artifact calls.

### 4.3 `gnomon_track`

`gnomon_track` handles the outcome loop:

- register a project or recurring question;
- submit actual outcomes;
- inspect model performance and interval coverage;
- resolve decisions;
- report regret and degradation;
- compare current behavior with prior runs.

Example:

```json
{
  "action": "submit_actuals",
  "project": "api-capacity",
  "forecast_id": "fc_...",
  "observations": []
}
```

### 4.4 Profiles

The proposed MCP profiles are:

| Profile | Purpose | Surface |
| --- | --- | --- |
| `core` | Normal agent use | inspect, run, track |
| `expert` | Manual composition and diagnostics | Existing lower-level tools |
| `research` | Evaluation of ungraduated features | Experimental models, planner, context experiments |

The CLI and Python API may retain ergonomic commands such as `forecast` and
`monitor`, but they should compile to the same canonical run request.

## 5. Result contract

Every completed run should return one compact envelope:

```json
{
  "status": "complete",
  "run_id": "run_...",
  "answer": {
    "headline": "Threshold crossing is unlikely during the next 14 days.",
    "support": "limited",
    "values": [
      {
        "timestamp": "2026-08-12",
        "point": 318.2,
        "lower": 296.1,
        "upper": 347.8,
        "tier": "limited"
      }
    ]
  },
  "limitations": [
    "Only three independent evaluation folds were available.",
    "Uncertainty after day 8 borrows pooled residuals."
  ],
  "evidence": {
    "selected_model": "seasonal_naive",
    "strongest_baseline": "last_value",
    "baseline_improvement": 0.084,
    "measured_coverage": 0.78
  },
  "artifact": "..."
}
```

The response must preserve:

- the support state and headline;
- all warnings and material assumptions;
- the values required to answer the question;
- model and baseline identity;
- calibration quality;
- a reference to the full artifact.

Bulk diagnostics and complete arrays may be trimmed from the response, but
never from the artifact.

## 6. Public support language

The public surface should prefer three immediately understandable states:

| State | Meaning |
| --- | --- |
| `supported` | The result passed the full evidence requirements. |
| `limited` | A result exists, but material evidence limitations must travel with it. |
| `abstained` | Gnomon cannot publish a defensible answer at the requested policy floor. |

Internally, Gnomon may preserve richer reasons and per-row tiers such as
conditional, split-horizon, or best-effort. The compact state is a projection,
not a deletion of the detailed contract.

## 7. What remains core

The following capabilities are Gnomon's durable foundation and should receive
most engineering investment:

- bitemporal snapshots and `as_of` replay;
- structural leakage prevention;
- immutable content-addressed artifacts;
- mandatory strong baselines;
- separated selection, calibration, and report-only test partitions;
- measured uncertainty with disclosed calibration limitations;
- per-row support provenance;
- explicit abstention and actionable recovery;
- deterministic claim verification;
- outcome submission and realised scoring;
- disclosed deterministic repair;
- pinned model packages and weight revisions;
- equivalent behavior across Python, CLI, and MCP surfaces.

## 8. Governed question types

The five current verbs need not disappear, but they should stop behaving like
five independent products.

### Forecast

The numerical center: a baseline-tested point path and uncertainty distribution.

### Threshold risk and monitor

Derived from:

```text
forecast distribution + threshold + intervention costs
```

Monitoring must not recalculate an incompatible probability model or increase
forecast support.

### Decide

Derived from:

```text
forecast scenarios + caller-supplied actions + caller-supplied utility
```

Without meaningful utilities, Gnomon may report feasibility and scenarios but
must not pretend to have optimized the decision.

### Investigate

Derived from validated historical diagnostics and explicitly associational
evidence. It may rank changes and correlations but cannot produce causal claims.

### Detect

Anomaly detection should remain a question type only where detector choice has
been validated against representative labels or a clearly disclosed proxy.
Synthetic-injection performance alone should not imply real-world support.

## 9. Default model policy

The standard profile should use a deliberately small contest:

- last value;
- seasonal naive;
- drift;
- one robust local or statistical model;
- optionally one graduated TSFM.

Additional candidates increase selection variance, runtime, dependency risk,
and the amount of evidence required to prove reproducibility. A candidate
graduates to the default pool only if it improves overall utility on the target
task distribution, not merely because an adapter exists.

## 10. Experimental scope

The following should live behind the research profile until each satisfies a
graduation criterion:

- meta-model stacking;
- multiple configurable ensemble strategies;
- experimental plan compilation and execution;
- LLM-proposed context effects;
- structural-event transformations;
- automatic model routing;
- broad anomaly-detector tournaments;
- unproven TSFM adapters;
- proposer-skill and event-effect learning.

Experimental artifacts must identify the feature and version that affected the
answer. A research feature cannot silently alter a core-profile result.

### Graduation criteria

A feature may enter the core profile when it has:

1. a precise behavioral contract;
2. a single executable path shared by evaluation and final prediction;
3. leakage and identity tests;
4. cross-surface equivalence tests;
5. measured benefit on representative tasks;
6. no material regression in completion, calibration, latency, or cost;
7. a stated condition under which it would be removed again.

## 11. Proposed architecture

The codebase should converge toward five stable domains:

```text
gnomon/
|-- temporal/       # snapshots, vintages, timestamps, frequency, leakage
|-- forecast/       # candidates, evaluation, calibration, final refit
|-- contract/       # tasks, artifacts, support, claims, verification
|-- tracking/       # actuals, scoring, projects, decisions, regret
`-- interfaces/     # Python, CLI, MCP adapters
```

Research components may be separated conceptually or physically:

```text
gnomon_experimental/
|-- ensembles/
|-- context/
|-- anomaly/
|-- planner/
`-- tsfm/
```

This does not require an immediate directory rewrite. The first goal is clear
dependency direction and ownership; file moves can follow when they reduce
risk rather than merely improve appearance.

### 11.1 Executable candidate contract

Evaluation should return a fitted executable candidate, not only a model name.

Conceptually:

```python
class ExecutableCandidate(Protocol):
    identity: CandidateIdentity

    def predict(self, history, horizon) -> ForecastPath: ...
```

`CandidateIdentity` includes:

- candidate kind and implementation version;
- exact member set for an ensemble or stack;
- fitted weights and strategy;
- configuration affecting prediction;
- dependency and weight revisions;
- fallback policy;
- training/evaluation data fingerprints.

The same executable contract is used for:

- selection-fold prediction;
- calibration-fold prediction;
- report-only test prediction;
- final prediction on all visible history.

If an executable cannot be reconstructed at final prediction, Gnomon abstains
or explicitly evaluates and calibrates a fallback. It never silently assembles
a different candidate under the same name.

### 11.2 Configuration authority

Configuration loading should obey:

```text
explicit request configuration
    -> explicit configuration file
    -> current environment
    -> fresh built-in defaults
```

Requirements:

- return a fresh configuration object per load;
- do not read mutable environment configuration only at module import;
- fail if an explicitly named environment or config path does not exist;
- fail if a discovered config cannot be parsed;
- reject unknown or inert options;
- include all answer-changing options in artifact identity;
- emit the resolved behavioral configuration in evidence.

## 12. Simplified configuration

The normal configuration should be small:

```yaml
evaluation:
  minimum_baseline_improvement: 0.02
  interval_coverage: 0.80
  minimum_support: best_effort

models:
  profile: standard

output:
  directory: ./gnomon-output
```

Advanced configuration belongs to explicit expert or research sections. It
should not be necessary to understand ensemble weighting, context gates, fold
stride, TSFM sandboxes, or artifact storage to obtain a normal forecast.

## 13. User experience

### CLI

```bash
gnomon inspect data.csv --target requests
gnomon run data.csv --target requests --horizon 14
gnomon run data.csv --target requests --horizon 14 --threshold 340
gnomon track actuals fc_123 actuals.csv
```

Compatibility aliases may remain:

```bash
gnomon forecast ...
gnomon monitor ...
gnomon decide ...
```

They should compile to the same `RunRequest`, not maintain separate numerical
implementations.

### Agent flow

```text
user question
    -> inspect only when input is ambiguous
    -> one gnomon_run call
    -> agent relays Gnomon's headline, support, limitations, and answer
```

The normal flow should not require an artifact fetch. An artifact fetch is for
audit, bulk data, or deeper analysis.

## 14. Evaluation framework

Every release should publish a paired outcome table:

| Measure | Purpose |
| --- | --- |
| Completion rate | Whether Gnomon answers useful tasks |
| Error when answered | Quality of published values |
| Utility with abstention priced | Whether caution improves the total outcome |
| Interval coverage | Whether stated uncertainty is calibrated |
| Interval width | Whether calibration is useful rather than trivially broad |
| Catastrophic-error rate | Whether the harness limits tail failures |
| Leakage violations | Whether point-in-time safety holds |
| Decision regret | Whether downstream choices improve |
| Agent misquotation rate | Whether results survive agent explanation |
| Token use and latency | Whether the integration is operationally viable |

The desired claim need not be “Gnomon has the lowest average forecasting
error.” A defensible and valuable target is:

> Gnomon achieves competitive median accuracy, substantially fewer
> catastrophic failures, zero detected temporal leakage, calibrated useful
> intervals, and a high completion rate at bounded operational cost.

Results should always report the number of tasks attempted, answered, scored,
degraded, and abstained. Accuracy conditional on a small answered subset is not
sufficient.

## 15. Migration plan

### Phase 0: correctness freeze

Do not add new default models, tools, question types, or configuration options.

Complete a boundary audit of:

- evaluation to final prediction;
- configuration to behavior;
- runtime result to artifact identity;
- MCP schema to Python runner;
- environment to persistent storage.

Add invariant tests before refactoring:

- the final candidate equals the evaluated candidate;
- changing any prediction-relevant option changes identity;
- every accepted configuration field has an observable effect;
- CLI, Python, and MCP return equivalent artifacts;
- fallback changes model identity and calibration provenance together.

### Phase 1: canonical run request

Introduce `RunRequest` and `RunResult` as the internal contract. Adapt the
existing forecast path first. Existing CLI, Python, and MCP entry points compile
into this contract.

No compatibility command is removed in this phase.

### Phase 2: executable candidates

Replace name-based final dispatch with executable candidates. Start with
built-in models, then ensembles and optional TSFMs. Disable any candidate that
cannot satisfy the contract.

### Phase 3: compact core tools

Add `gnomon_run` and `gnomon_track`. Make the three-tool core profile the
recommended agent integration. Preserve the expert profile for existing users.

### Phase 4: governed views

Compile threshold risk, monitor, decide, and investigate requests into the
canonical run graph. Remove duplicated numerical logic where a view can consume
the canonical forecast distribution.

### Phase 5: evidence-based graduation

Re-evaluate experimental components against the shared outcome table. Graduate,
retain as research, or remove each component based on measured value.

## 16. Acceptance criteria

This proposal is successful when:

1. A new agent can answer the README scenario using at most one inspection and
   one execution call.
2. The default MCP profile exposes no more than three primary tools.
3. Evaluation and final prediction share the exact executable candidate.
4. No accepted configuration field is inert.
5. A present but unreadable configuration fails loudly.
6. Every answer-changing dependency, weight, strategy, member, and option is
   covered by identity and evidence.
7. Forecast, threshold, monitor, and decision views cannot disagree about the
   underlying distribution.
8. A full post-change evaluation reports safety, usefulness, completion, cost,
   and latency together.
9. Core-profile results do not depend on experimental features being installed.
10. Existing compatibility surfaces either compile into the canonical contract
    or clearly report their deprecation path.

## 17. Non-goals

This proposal does not require:

- deleting the current five verbs;
- abandoning TSFMs;
- removing research infrastructure or historical evidence;
- replacing deterministic repair or support tiers;
- immediately reorganizing every source file;
- optimizing solely for benchmark coverage;
- forcing Gnomon always to answer.

It does require that the default product be smaller than the complete research
system and that every published number be produced by the exact mechanism that
earned the right to publish it.

## 18. Final position

Gnomon should not try to be the agent's temporal brain. It should be the
agent's temporal court of record:

- the agent asks;
- Gnomon determines what was knowable;
- Gnomon evaluates and computes;
- Gnomon states what the evidence permits;
- the artifact proves it;
- later outcomes hold it accountable.

Narrowing the default surface does not reduce Gnomon's ambition. It concentrates
that ambition on a defensible category: the trusted temporal execution boundary
for agents.
