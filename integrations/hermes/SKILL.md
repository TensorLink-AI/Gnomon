---
name: forecasting
description: Evidence-backed forecasting with Aion, honestly
---

# Forecasting with Aion

Aion turns time-series data into a backtested forecast, compares every
candidate against naive baselines, measures uncertainty, and abstains when
the evidence is inadequate. You formulate the question and explain the
evidence; Aion owns every number.

## Workflow

1. **Detect capabilities.** Call `aion_capabilities` once per session if you
   are unsure what the installed runtime supports. Never assume a feature
   from documentation.
2. **Inspect before forecasting.** When column mappings, frequency, or data
   quality are uncertain, call `aion_inspect` first and resolve any
   ambiguity with the user rather than guessing.
3. **Forecast.** Call `aion_forecast` with an explicit horizon. The response
   is compact: per-series support status, selected model, warnings, and the
   artifact directory.
4. **Read the numbers from the artifact.** Forecast values live in
   `forecast.csv` and `summary.md` inside the returned `artifact_path`.
   Quote them verbatim. `artifact.json` holds full provenance when the user
   asks for detail.

## Hard rules

- **Never invent, adjust, round-trip, or extrapolate forecast values,
  intervals, scores, or coverage.** If a number is not in an Aion artifact,
  it does not exist.
- **Preserve abstention.** `unsupported` means Aion declined to forecast
  that series. Report it as abstention with Aion's reasons — never
  paraphrase it as "low confidence" and never substitute your own estimate.
- **Surface every warning** and the support status in your final answer,
  even when the user only asked for the numbers.
- **Do not infer business thresholds.** If the user's decision needs a
  threshold (capacity, budget, SLA) and none was given, ask or omit
  threshold analysis.
- **Errors are actionable.** Aion returns structured errors with
  `suggested_actions` (for example reduce the horizon, provide more
  history). Prefer following a suggested action or asking the user over
  improvising around the error.
- **Data stays local.** Never upload the user's data anywhere as part of a
  forecasting task unless the user explicitly requests sharing.

## Interpreting support

| Status | Meaning | What you say |
| --- | --- | --- |
| `supported` | Selected forecast beat or appropriately retained a baseline on valid folds | Report values with intervals and warnings |
| `weakly_supported` | Forecast exists but evidence is limited or risky | Report values with the qualifiers Aion gives |
| `unsupported` | Evidence inadequate; Aion abstained | Report the abstention and its reasons; offer the suggested actions |
