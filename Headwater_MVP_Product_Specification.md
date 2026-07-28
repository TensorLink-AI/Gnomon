# Headwater MVP Product Specification

**A deployable forecasting capability by Cascade**

**Version:** 0.1  
**Date:** 28 July 2026  
**Status:** Working product specification for validation and implementation

> **Product decision:** Headwater is a specialist forecasting capability that technical users, Hermes Agent, and other AI agents can run locally. General-purpose agents may reason about the task, discover permitted context, and interpret results. Headwater remains authoritative for temporal validation, numerical forecasting, backtesting, uncertainty, model selection, abstention, and evidence.

## Purpose of this document

This document defines the product being built, who it is for, the first useful workflows, the MVP boundary, the launch and retention loops, the assumptions that must be tested, and the criteria for deciding whether Headwater has earned further investment.

It intentionally does not specify every internal module or API. Those details are contained in the companion **Headwater System Design**.

## Brand architecture

| Layer | Name |
| --- | --- |
| Parent research/network brand | Cascade |
| Product | Headwater |
| Endorsement | Headwater by Cascade |
| CLI | `headwater` |
| MCP integration | Headwater MCP |
| Hermes integration | Headwater for Hermes |
| Optional hosted service | Headwater Cloud |

## Executive decision

> **Recommendation:** Build a narrow, trustworthy forecasting runtime with an optional agentic layer. Make Hermes the hero integration, while keeping Headwater independently deployable through its CLI, MCP server, Python API, and containers.

The MVP should not attempt to be a general-purpose data-science agent. Its job is to transform a measurable temporal question into a defensible forecast and a structured statement of what another system may safely conclude from it.
The LLM layer should improve task formulation, data mapping, context discovery, bounded experiment selection, and interpretation. It must not generate or modify numerical forecasts. The numerical runtime remains authoritative for validation, backtesting, model selection, quantiles, calibration, and abstention.
The launch proposition is broad enough to attract technical users — “forecast anything you can measure” — but the runtime underneath must be disciplined enough to support recurring, production-like use.

| Decision | MVP choice |
| --- | --- |
| Product category | Deployable temporal intelligence and forecasting capability for agents and technical users |
| Primary integration | Hermes Agent running Headwater locally through MCP or the CLI |
| Canonical interface | Typed runtime library with CLI and MCP adapters |
| Agent boundary | LLM reasons about intent, context, experiments, and explanation; deterministic tools own numbers |
| Initial promise | Produce a backtested forecast, compare it with baselines, quantify uncertainty, and abstain when evidence is inadequate |
| Primary growth motion | Shareable technical forecasts and agent integrations |
| Primary retention motion | Persistent forecast projects, scheduled reruns, actual submission, and realised scoring |

## Product thesis and positioning

### Product thesis

> Existing tools return predictions. Headwater determines whether the forecasting task is valid, whether a candidate beats defensible baselines, how uncertain the result is, and whether an agent should trust or reject it.

### Positioning hierarchy

| Layer | Message |
| --- | --- |
| Homepage headline | Give your agent the ability to forecast. |
| Broad acquisition line | Forecast anything you can measure. |
| Technical category | A temporal intelligence CLI and runtime for developers and agents. |
| Trust differentiator | It does not just predict. It tests whether the prediction deserves to be trusted. |
| Hermes-specific line | Hermes manages the task. Headwater validates the data, runs the models, and returns evidence-backed forecasts. |

### What Headwater is not

- Not an LLM generating future values from serialised arrays.
- Not a replacement for Hermes or another general-purpose agent.
- Not initially an enterprise planning suite, dashboard product, or optimisation engine.
- Not a promise that a complex model will always beat a simple baseline.
- Not a system that always returns a forecast; abstention is a supported outcome.

## Audience, jobs, and initial use patterns

### Primary audience

The first audience is technical and distribution-oriented: developers, open-source maintainers, ML/data engineers, agent builders, SRE/FinOps practitioners, and technically sophisticated founders. They are comfortable with terminals, files, APIs, JSON, automation, and public sharing.

### Jobs to be done

| Job | User statement |
| --- | --- |
| One-off insight | “I have a metric or dataset. Tell me what is likely to happen next and how much confidence I should place in it.” |
| Agent capability | “I want Hermes or another agent to perform forecasting as part of a broader task without inventing numbers.” |
| Recurring project | “Rerun the same forecast when data changes and show whether previous forecasts were accurate.” |
| Threshold decision | “Tell me whether traffic, cost, capacity, backlog, or demand is likely to cross a meaningful limit.” |
| Contextual analysis | “Check whether releases, incidents, launches, promotions, or other events improve the forecast.” |
| Public comparison | “Forecast and compare public technical projects, packages, repositories, or usage trends in a shareable format.” |

### Initial templates

| Template | Question class |
| --- | --- |
| github-growth | Repository stars, forks, contributors, issues, and milestone dates |
| issue-backlog | Open issue or ticket volume and maintainer/service capacity |
| ai-spend | Token consumption, API cost, credit exhaustion, and budget risk |
| api-traffic | Requests, throughput, latency events, and capacity thresholds |
| cloud-cost | Cloud spend and resource consumption |
| gpu-demand | GPU-hours, job queues, inference volume, and reserved capacity |
| package-growth | Package downloads and ecosystem momentum |
| custom-csv | A general timestamp/target/series input path |

## Product principles

| Principle | Implication |
| --- | --- |
| Thin agent, thick tools | LLMs propose and explain; deterministic tools validate and calculate. |
| The numerical core owns numbers | No LLM may alter point forecasts, quantiles, model scores, calibration, or support status. |
| Fail loudly | A structured error is better than a plausible forecast at the wrong frequency or with leaked context. |
| Baselines are mandatory | Every candidate is compared with strong naive and seasonal baselines. |
| Context must earn admission | Context affects forecasts only after temporally valid ablation demonstrates stable out-of-sample value. |
| Abstention is a first-class result | The system can return unsupported or weakly supported, with actionable recovery guidance. |
| One protocol, many surfaces | CLI, MCP, Python, HTTP, and wrappers share the same contracts and runtime. |
| Local first, hosted later | The first useful workflow should not require an account or remote service. |
| Evidence travels with the result | Every material explanation resolves to machine-readable evidence and provenance. |
| Broad acquisition, narrow execution | Users can try many questions, but the MVP supports a deliberately restricted set of temporal inputs and tasks. |

## MVP scope

### MVP product promise

1. Ingest a regular univariate or panel time-series dataset from CSV or Parquet.
2. Preserve true frequency, timezone, timestamp geometry, and configured missingness policy.
3. Infer or request the time, target, series, frequency, horizon, and threshold fields.
4. Run mandatory diagnostics and timestamp-aware rolling-origin evaluation.
5. Compare last-value, seasonal-naive, one statistical model, and one TSFM adapter.
6. Select per series or abstain when evidence is inadequate.
7. Generate point and quantile forecasts with an explicit uncertainty method.
8. Return a human summary plus immutable machine-readable artifacts.
9. Allow later submission of actuals and realised scoring.
10. Expose the workflow through a one-shot CLI and a local MCP server.
11. Provide a Hermes skill/plugin that teaches Hermes to operate Headwater safely.

### In-scope agentic components

| Component | MVP behaviour |
| --- | --- |
| Task compiler | Translate a natural-language request and discovered metadata into a typed ForecastTask; expose assumptions and missing material fields. |
| Data mapper | Propose timestamp, target, series, covariate, and threshold mappings from local files. |
| Context investigator | Search explicitly permitted files/sources and propose typed ContextEvents. Proposals do not automatically enter the model. |
| Experiment planner | Propose a small bounded set of diagnostic or ablation experiments when they can change the decision. |
| Interpretation agent | Explain the immutable artifact, cite evidence IDs, surface uncertainty and abstention, and relate forecasts to user thresholds. |
| Deterministic fallback | Produce useful templates and summaries when no LLM is configured. |

### Explicit non-goals

- General autonomous web research.
- Open-ended multi-agent conversations.
- Anomaly detection, classification, optimisation, or automated business actions.
- A full hosted GUI or enterprise collaboration suite.
- Dozens of model integrations or a model marketplace.
- Customer-specific training or fine-tuning infrastructure.
- Arbitrary mixed-frequency or irregular event-stream forecasting.
- Causal claims from correlational context.
- A scalar probability-like “reliability score” without outcome calibration.

## Core user experiences

### One-command standalone forecast

```bash
headwater forecast observations.csv \
  --time timestamp \
  --target requests \
  --series service_id \
  --horizon 168 \
  --frequency h \
  --threshold 4500000
```

```text
Forecast completed for 6 series

Support: weakly supported
Selection: per-series
Strongest baseline: seasonal_naive
Median improvement over baseline: 7.4%
80% interval threshold crossing: likely on 2 series

Warnings:
  - Recent level shift detected on api-prod
  - 80% interval coverage is 76%, below target
  - One series retained the seasonal baseline

Artifacts:
  ./headwater-output/forecast.csv
  ./headwater-output/summary.md
  ./headwater-output/artifact.json
  ./headwater-output/evidence.jsonl
```

### Hermes-operated forecast

A user asks Hermes a broad question. Hermes searches the allowed workspace, identifies candidate data and context, calls typed Headwater tools, and explains the returned evidence. Headwater never depends on Hermes for numerical correctness.

### Persistent project mode

```bash
headwater init
headwater run
headwater status
headwater actuals submit actuals.csv
headwater score
headwater compare latest previous
```

Project mode stores configuration, data/version fingerprints, prior forecasts, model decisions, actual outcomes, and evaluation history. It is the primary retention mechanism.

### Shareable forecast

```bash
headwater share forecast_01K...
headwater badge forecast_01K...
```

The share output contains a chart, headline prediction, uncertainty range, historical backtest result, support level, generated-at time, and reproducible command. Public sharing is opt-in; local data is never uploaded implicitly.

## Product validation and success metrics

### Assumptions to validate

| Assumption | Test | Signal |
| --- | --- | --- |
| Technical users have broad forecasting curiosity | Public and personal technical metrics generate enough initial installs and first forecasts. | Activation: successful meaningful forecast within first session. |
| A subset has recurring use | Users rerun the same forecast, initialise projects, or schedule execution. | Four-week repeat rate and projects with two or more runs. |
| Hermes integration creates adoption | Users value adding forecasting as a specialist capability to an existing agent. | Hermes plugin installs, MCP calls, and completed agent workflows. |
| Trust changes behaviour | Baseline comparison, support, and abstention influence whether users act or revise data. | Users fix inputs, reduce horizon, retain baseline, or delay decisions based on warnings. |
| Actual scoring creates a loop | Users submit outcomes and compare forecast versions. | Share of recurring projects with actuals and score events. |
| Shareable outputs drive discovery | Public forecasts cause other users to reproduce or adapt the command. | Share-to-install/reproduce conversion. |

### MVP funnel

| Stage | Observable behaviour |
| --- | --- |
| Discovery | Visits documentation, sees a shared forecast, or installs from a Hermes workflow. |
| Activation | Completes a valid forecast and understands the support result. |
| Value | Uses the output to answer a threshold, milestone, comparison, or planning question. |
| Retention | Reruns, initialises a project, submits actuals, or embeds Headwater in an agent/workflow. |
| Advocacy | Shares a forecast, publishes a template, or integrates Headwater into an open-source project. |
| Commercial signal | Requests hosted runs, connectors, private deployment, higher throughput, governance, or support. |

### Initial success criteria

- At least 60% of qualified installers complete one meaningful forecast.
- At least 30% of activated users run a second forecast within four weeks.
- At least 15% of activated users create a persistent project or integrate MCP/CLI into another workflow.
- At least 10 real users submit actuals or score a previous forecast.
- At least five external users complete a Hermes-operated workflow without developer assistance.
- A meaningful portion of unsupported results lead to a successful remediation and rerun rather than abandonment.
- Public examples produce reproducible commands that other users run on their own data or projects.

## Risks, non-goals, and open decisions

### Principal risks

| Risk | Failure mode | Mitigation |
| --- | --- | --- |
| Novelty without retention | Broad public forecasts attract one-off experimentation but no repeat workflow. | Make project mode, update, actual submission, scoring, and thresholds visible early; instrument repeat use. |
| Platform theatre | The CLI exposes many commands that do not perform real work. | Public capabilities must be generated from tested runtime features; hide experimental tools. |
| Agent overreach | LLM silently changes assumptions, introduces leaked context, or narrates excessive certainty. | Typed proposals, deterministic gates, evidence-linked explanations, budgets, and immutable forecast values. |
| Weak differentiation | Agent-native and MCP become commodity features. | Own the trust contract: temporal correctness, baseline comparison, context admission, calibration, abstention, and realised scoring. |
| Poor first-run experience | Frequency, schema, or dependency friction prevents technical users reaching value. | Provide automatic inspection, clear remediation, example templates, local zero-key path, and one-command success. |
| Abstention feels broken | Users interpret unsupported as tool failure. | Return actionable reasons, safe fallbacks, minimum data requirements, and rerun commands. |
| Context demos overstate value | Agent-discovered context looks impressive but leaks future information or degrades performance. | Require known-at time and identical-fold ablation; keep autonomous discovery experimental until proven. |
| Local resource variability | TSFM adapters are slow or incompatible on user machines. | Use capability estimates, optional remote adapters, preserved baselines, partial artifacts, and explicit runtime budgets. |

### Deferred decisions

- Final package and command naming: headwater, headwater-forecast, or namespaced subcommand.
- Which lightweight statistical and TSFM adapters are reliable enough for the default local install.
- Whether charts are generated locally with a static renderer or through an optional web share service.
- The public/private projection format for shareable artifacts.
- How context source permissions are represented consistently across Hermes, CLI, and standalone agent modes.
- The first public-data connector after local files: GitHub, package registries, or generic CSV/JSON URL.
- Whether scheduling belongs in Headwater project mode or remains delegated to Hermes/system schedulers.
- The minimum anonymised telemetry needed to understand adoption without undermining local-first trust.

### Launch demonstration

> **Hero demo:** A Hermes user asks a broad capacity question. Hermes finds a usage dataset, capacity configuration, releases, and incidents. Headwater validates the time grid, evaluates context, compares baselines and models, and returns a threshold-risk conclusion with uncertainty and support. The complete run is reproducible through one command and one MCP workflow.

### Definition of an externally credible MVP

- One canonical runtime is used by the CLI, MCP server, and tests.
- A fresh user can install and complete the documented local quick start.
- Frequency, timezone, missing timestamps, duplicates, and future indexes are correct end to end.
- Backtests are timestamp-aware and reconstruct historical context availability.
- Baselines are mandatory, selection is per series, and unsupported forecasts abstain.
- Uncertainty method and measured coverage are explicit.
- Every material claim resolves to evidence and provenance.
- Actual submission and realised scoring work end to end.
- Hermes can operate Headwater through typed MCP tools and faithfully communicate warnings.
- The public capabilities response contains no mocked analytical behaviour.
- The product has at least three external repeat users, not only one-off demos.

## Closing recommendation

> Build a delightful forecasting capability that Hermes and other agents can operate, backed by a narrow deterministic runtime that treats forecasts as evidence-backed claims. Lead with broad technical curiosity; earn retention through repeatable projects, scoring, and trustworthy decisions.

The LLM layer is not decorative. It can materially improve usability and, when evaluated correctly, the inputs to a forecast. But the product earns trust because every numerical conclusion remains owned by deterministic temporal tools and every contextual improvement must prove itself historically.

The most important implementation discipline is to resist expanding the command surface before one end-to-end path is exceptionally correct: inspect data, formulate task, forecast, evaluate, explain, update, and score.


## Product decision summary

Build Headwater as a technically serious but approachable forecasting product:

- Broad and interesting enough for technical users to try on many measurable questions.
- Simple enough to deliver a meaningful first result from one command.
- Trustworthy enough to compare models with baselines, quantify uncertainty, and abstain.
- Persistent enough to support reruns, actual submission, realised scoring, and repeat use.
- Agent-native enough for Hermes and other systems to operate locally without surrendering numerical control to an LLM.
- Distinct from Cascade while clearly belonging to the same product family.
