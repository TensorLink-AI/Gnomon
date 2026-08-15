---
name: forecasting
description: Trusted temporal execution with Gnomon — forecast, investigate, detect, decide, and monitor honestly
---

# Temporal reasoning with Gnomon

Gnomon is the trusted execution boundary for temporal questions: you formulate the
objective and explain the evidence; Gnomon validates, computes, and owns every
number, interval, probability, selection, warning, and support status.

## The five verbs

| Question | Tool | Notes |
| --- | --- | --- |
| What changed? | `gnomon_investigate_change` | Changepoints, regime vs transient, ranked *associational* explanations — never causes |
| What happens next? | `gnomon_forecast` | Backtested selection with abstention |
| What is abnormal? | `gnomon_detect_anomalies` | Competing detectors graded before the winner flags observations |
| What should we do? | `gnomon_decide` | Needs actions; utilities optional — without them you get the feasible-action comparison, not a choice |
| When should we intervene? | `gnomon_monitor` | Sequential risk; supply alert/miss costs for a cost-optimal rule |

Workflow: `gnomon_capabilities` once if unsure what is installed →
`gnomon_inspect` when mappings or quality are uncertain → the macro → quote
numbers directly from its response. The full profile also offers
`gnomon_get_artifact`; `gnomon_explain_run`, `forecast.csv`, and `summary.md`
remain available for deeper audit.
Track consequential runs with a `project`; close the loop with
`gnomon_submit_actuals` and the decision record/resolve tools.

## Hard rules (all five verbs)

- **Never invent, adjust, or extrapolate any number.** If it is not in a
  Gnomon artifact, it does not exist.
- **Preserve abstention.** `unsupported` and `inconclusive` are conclusions.
  Report them with Gnomon's reasons; never soften to "low confidence", never
  substitute your own estimate.
- **Respect the support status.** Every result carries a
  `support_assessment` (`supported` / `conditionally_supported` /
  `inconclusive` / `unsupported` / `invalid`) with reasons, assumptions, and
  recovery actions. Surface it — and every warning — even when the user only
  asked for numbers.
- **Never upgrade claim classes.** Investigation explanations are
  associational rankings; a lead correlation is predictive precedence.
  Do not present either as a cause. Performance history is observational.
- **Degraded is not failed.** A decision without utilities returns the
  comparison and probabilities, `conditionally_supported: missing utility
  inputs`. Report exactly that; ask for utilities if the user wants a choice.
- **Errors are actionable.** Structured errors carry `repair_options`;
  follow one or ask the user rather than improvising.
- **Never hand-clean data yourself.** For messy files, call `gnomon_inspect`
  first: its `data_quality` report names the needed repair level, and
  `repair: "aggressive"` on `gnomon_forecast` applies capped, disclosed fixes
  (gaps, jitter, conflicts) that surface as warnings. Preserve those
  `repaired_data:` warnings when reporting. Writing your own cleanup code
  hides the repairs from the audit trail.
- **Do not infer business thresholds or costs.** Ask, or omit the analysis.
- **Data stays local** unless the user explicitly requests sharing.

## Context events and covariates (evidence-gated)

Propose context events only from files the user permitted; Gnomon validates,
grounds, and admits them only on demonstrated backtest lift — report
rejections honestly. For external covariates: `gnomon_covariate_guide` before
fetching, preserve issued vintages, `gnomon_validate_covariates` before
proposing. Never substitute realised outcomes for the vintage available at
a cutoff; the store's `--as-of` replay exists to prove this.

## Reporting issues (do this instead of working around a bug)

If Gnomon behaves wrongly — a crash, a result that contradicts its own
evidence, an error whose `repair_options` don't apply — do not silently
work around it. Gather the reproducible report and give it to the user
(or file it at https://github.com/TensorLink-AI/Gnomon/issues if they ask):

1. the exact command or tool call with its arguments;
2. the complete JSON error envelope, or the run's `artifact.json`;
3. the output of `gnomon_capabilities` (runtime version and installed extras);
4. if data can't be shared, the `data_quality` section from `gnomon_inspect`.

Those four items make nearly every report reproducible without the
original file. A wrong number with evidence attached is a bug; report it —
never patch over it with your own estimate.
