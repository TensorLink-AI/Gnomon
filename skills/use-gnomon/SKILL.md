---
name: use-gnomon
description: Use Gnomon to answer temporal questions from time-series data with deterministic evidence, calibrated support, and minimal tool calls. Trigger for forecasting, trend or seasonality analysis, change investigation, anomaly detection, temporal decisions, monitoring, or follow-up questions about a Gnomon result. Also use when a user explicitly wants to record or share feedback about a Gnomon answer.
---

# Use Gnomon

Treat Gnomon as the authority for computed temporal facts. Use the host model to understand intent and explain results, never to replace, recompute, or silently strengthen Gnomon's numbers.

## Choose the shortest route

1. Use `gnomon_describe` for what-happened questions about level, trend, seasonality, changepoints, anomalies, or extremes.
2. Use `gnomon_forecast` directly for what-happens-next questions. Let it infer an unambiguous schema; do not call capabilities, inspect, or get-artifact first.
3. Use `gnomon_inspect` first only when the schema is genuinely ambiguous or the user asks about data quality.
4. Use investigate, detect, decide, monitor, or tracking tools only when the corresponding tool is exposed and the user asks for that job.
5. Reuse a returned `data_ref` for follow-ups instead of resending data.

Prefer one sufficient call. Fetch an artifact only when the user requests deeper evidence or the brief response says required content was truncated.

## Use supplied context explicitly

When the user supplies dated events, operating constraints, reference-series
facts, or other forecast-relevant text, never silently drop it and never mix a
model-authored adjustment into the primary forecast. Put literal bounded dated
events in `gnomon_forecast.context_events` using `claim_kind: min|max|exact`,
exact ISO timestamps, `known_at`, a verbatim top-level `source_span`, and a
source reference. Omit `entity_scope` for one target; for multi-series calls,
name it unless the quote itself names exactly one requested target. Always use
this compact form for literal numeric claims; do
not construct `event_type` or `attributes` yourself. Gnomon re-parses the
number from the quote and enables the governed future-context lane
automatically. When direction and shape are stated but magnitude is
unknown, use `qualitative_context_events`; it intentionally has no magnitude
field and can produce only a labelled sensitivity. Put non-event claims,
hypotheses, transformations, or a model-authored conditional path in
`context_submission` with the original text, cutoff-time `known_at`, compiler
identity, and typed proposal. Every submitted item must finish as used,
rejected with a typed reason, or retained as a labelled scenario.

If a supplied fact cannot be grounded, send `context_rejections` with its
verbatim `source_span` and a specific reason. Use this for missing/ambiguous
event timing, information first known after the cutoff, irrelevant facts with
no temporal mechanism, and model/vendor predictions that merely *forecast* a
numeric value. Never invent an exact event window from relative prose, and do
not encode somebody else's forecast as a `constraint:` or `override:`. A
`qualitative_context_events.effective_start` calendar date must be stated in
its `source_span`; otherwise reject the fact instead of resolving the date.

Choose publication mode from the user's intent:

- `strict` for automation or when only historically admitted effects may lead;
- `best_effort` when a human wants the best bounded recommendation despite
  weak evidence; the immutable primary remains visible in this mode, so a
  request to retain or disclose it does not by itself imply `scenario`; and
- `scenario` when the user wants the immutable primary beside several explicit
  what-if paths.

If the host can independently sample its model, use Gnomon's provider-neutral
sampled-prior prompt/parser integration and submit the sealed dossier rather
than averaging values in prose. Fewer than three valid sampled paths are
scenario-only. Sampling agreement is not historical skill, and no
model-authored path may authorize automation. Use `gnomon_select_scenario` only
to rank already sealed scenario IDs with cited claim IDs; it cannot edit their
numbers, support, primary, or automation status.

## Preserve the answer contract

- Relay `headline` verbatim when possible.
- Keep each point value with its `tier`; state `tier_floor`, material limitations, staleness, and repairs.
- Preserve provenance and `artifact_id` when the answer relies on an artifact.
- For multi-series answers, preserve `ranking_rule`, notable series, and `remainder_preserved` instead of inventing a new ranking.
- Distinguish the immutable primary forecast from conditional context scenarios. Never present a context-conditioned scenario as though it replaced the primary answer.
- Explain deterministic choice projections; do not override them with model intuition.

Do not turn `best_effort` into confidence. It is an answer with an explicit weak-support warning. If Gnomon abstains, say why and offer its literal recovery action. Issue a repaired call only when the user requested the answer and the recovery does not change their intent.

## Handle feedback with consent

Record feedback only after the user explicitly asks or agrees. If local command execution is available:

1. Create a structured local receipt with `gnomon-feedback create`. Include only the minimum task metadata needed to reproduce the behavior.
2. Put sensitive detail in `--private-note`; it never enters a shareable payload. Put text in `--public-summary` only when the user approves sharing that exact text.
3. Run `gnomon-feedback preview <receipt-id>` and show the preview before any export or submission.
4. Export or submit only after separate explicit consent. Never add `--consent` on the user's behalf.

Never record raw series, prompts, messages, credentials, local paths, or full tool arguments. Do not reward or optimize for call volume. A report can become reward-eligible only after independent verification and duplicate checking; the local claim secret lets a contributor redeem without attaching an identity to the report.
