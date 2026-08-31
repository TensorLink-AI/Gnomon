---
name: use-gnomon
description: Answer forecasting, description, detection, decision, monitoring, and Gnomon follow-up questions with deterministic temporal evidence and minimal calls. Also handle explicit Gnomon feedback requests.
---

# Use Gnomon

Gnomon owns computed facts; the host understands intent and explains them. Never replace, recompute, or strengthen its numbers.

## Choose the shortest route

1. Use `gnomon_describe` for level, trend, seasonality, changepoint, anomaly, or extreme questions.
2. Use `gnomon_forecast` directly for what-happens-next questions. Let it infer an unambiguous schema; do not call capabilities, inspect, or get-artifact first.
3. Use `gnomon_inspect` only for genuine schema ambiguity or requested data quality.
4. Use investigate, detect, decide, monitor, or tracking tools only when the corresponding tool is exposed and the user asks for that job.
5. Reuse returned `data_ref` instead of resending data.

Prefer one sufficient call. Fetch an artifact only when the user requests deeper evidence or the brief response says required content was truncated.
Map windows to grid periods (`next week`: daily 7, hourly 168). Use 1 only for
an explicit next-value question; otherwise omit it and keep the disclosed
seasonal default. For breach questions pass `threshold` and `horizon`; never
infer the decision by eyeballing forecast rows.

## Use supplied context explicitly

Never drop supplied context or mix it into the primary forecast. Put literal,
dated numeric claims in `context_events` with `claim_kind: min|max|exact`, ISO
times, `known_at`, verbatim `source_span`, and source reference. Omit scope for
one target; for multiple targets name it unless the quote uniquely names one.
Do not invent `event_type` or `attributes`; Gnomon re-parses quoted numbers.
Pass the source message as `context_source_text` to verify its quotes and
semantics.
Use `qualitative_context_events` for dated direction/shape with unknown
magnitude; it has no magnitude field and produces only a labelled sensitivity.
Use `context_submission` for other claims, hypotheses, transformations, or
model-authored conditional paths. Every item must be used, rejected with a
typed reason, or retained as a labelled scenario.

Send ungroundable facts to `context_rejections` with verbatim `source_span` and
a specific reason: ambiguous timing, post-cutoff knowledge, no temporal
mechanism, or somebody else's numeric forecast. Never invent dates or encode a
prediction as a constraint. A qualitative event's start date must appear in
its quote.

Choose publication mode from the user's intent:

- `strict` for automation or when only historically admitted effects may lead;
- `best_effort` when a human wants the best bounded recommendation despite
  weak evidence; the immutable primary remains visible in this mode, so a
  request to retain or disclose it does not by itself imply `scenario`; and
- `scenario` when the user wants the immutable primary beside several explicit
  what-if paths.

If the host independently samples its model, submit Gnomon's sealed sampled
prior rather than averaging in prose. Fewer than three valid paths are
scenario-only; agreement is not historical skill. `gnomon_select_scenario`
may rank sealed IDs with cited claims, never edit numbers/support or authorize
automation.

## Preserve the answer contract

- Relay `headline` verbatim when possible.
- Keep each point value with its `tier`; state `tier_floor`, material limitations, staleness, and repairs.
- Treat `tier_floor` as authoritative for the whole response, including series summarized under `triage.remainder_tiers`; never recompute it from only the visible top series.
- Preserve provenance and `artifact_id` when the answer relies on an artifact.
- For multi-series answers, preserve `ranking_rule`, notable series, and `remainder_preserved` instead of inventing a new ranking.
- Distinguish the immutable primary forecast from conditional context scenarios. Never present a context-conditioned scenario as though it replaced the primary answer.
- Explain deterministic choice projections; do not override them with model intuition.

`best_effort` is not confidence. Preserve its warning. On abstention, explain
why and copy the recovery action; retry only when it preserves user intent.

## Handle feedback with consent

Record feedback only after explicit agreement. With local execution:

1. Create a structured local receipt with `gnomon-feedback create`. Include only the minimum task metadata needed to reproduce the behavior.
2. Put sensitive detail in `--private-note`; it never enters a shareable payload. Put text in `--public-summary` only when the user approves sharing that exact text.
3. Run `gnomon-feedback preview <receipt-id>` and show the preview before any export or submission.
4. Export or submit only after separate explicit consent. Never add `--consent` on the user's behalf.

Never record raw series, prompts, messages, credentials, paths, or full tool
arguments. Never reward call volume. Shared reports require independent
verification and duplicate checking.
