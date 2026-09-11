# Development checkpoint — accumulated experience

**Evaluation implemented; improvement target not established.** Work is on
`dev/ledger-optimization`. Main and PyPI are unchanged. This is a new workflow
objective; the earlier negative forecast-selection confirmation remains intact.

## Objective

At least **20% fewer billed tokens per correctly completed evidence checkpoint**
than a capable persistent SQLite control, including failed attempts. Require
>=95% exact completion, completion noninferiority, forecast RMSLE noninferiority
(2% margin), independent safety/visibility checks, and the preregistered paired
uncertainty gates. See [the full protocol](PROTOCOL.md).

The comparison charges discovery, query construction, retries and model calls.
Both arms receive the same arrived raw event stream and forecasting tools. The
SQL control has indexes, a complete reusable reference query, persistent notes,
saved queries and schema/query discovery. Ingestion is automated for both, so
this version measures evidence retrieval, revision review and decision handoff;
it does not measure manual collection/ingestion labor.

## Current checks

* **15 regression tests pass.** Includes both storage implementations, independent
  arithmetic, immutable historical answers, exact ties, partial horizons, SQL
  read boundaries, metric recovery, failed-attempt costs, and confirmation guards.
* **864 retrieval/replay checks and 864 forecast-input visibility checks pass**
  across four full 24-round development worlds. Source/recording visibility,
  unit, revision, context and no-memory fault injections all cause detectable
  failures. These are finite checks, not a general safety proof.
* The latest adapter retains the effective metric and query fields. Omitting
  `metric` correctly discloses the public MAE default; submitting MAE for this
  RMSLE task returns specific correction guidance.
* Validation seeds 200–203 and confirmation seeds 9000–9023 remain unopened.
  Confirmation requires passing full validation, frozen hashes and a one-use
  cohort claim. Its complete grid is 2,304 decisions, clustered by 24 worlds.

The [latest offline audit](evidence/audit-007.json) records exact source hashes.
Its repeated requests/feature checks are not independent statistical samples.

## Experiments retained, including failures

| Run | Source | Outcome |
| --- | --- | --- |
| Pilot 001 | c2119a5 | Interrupted: shared submission schema too vague; vintage queries used different origin ranges. |
| Pilot 002 | 8e7b3ab | Stopped cooperatively: forecast features could use unavailable revisions; control schema discovery was lost after chat reset. |
| Pilot 003 | bd75d50 | All 64 decisions processed, but the compact wrapper omitted metric labels. Retained as harness-development evidence. |
| Focused replay 004 | 5fa5d65 | The known failing Gnomon case completed with correct RMSLE and no retries; SQL replay failed on registry discovery, subsequently improved. |

Pilot 003 had **20/32 correct Gnomon checkpoints versus 26/32 SQLite**, and token
cost per correct checkpoint of **31,427.95 versus 28,471.08** (ratio **1.10386**).
All 12 failed Gnomon checkpoints contained a default-MAE query. Gnomon returned
the metric in its public result, but our compact adapter dropped it, and the
benchmark schema did not explain the default. This is a material evaluation
confound. **Do not use these numbers as a fair estimate of product advantage or
disadvantage.** No original score was rewritten after fixing the adapter.

Engine execution and task completion are separate: Gnomon obtained typed
executions on 32/32 checkpoints; SQLite on 27/32. Pilot 003 used 305 API calls,
1,368,807 billed tokens and no API errors. Its exploratory four-world intervals
and failed gates are retained in [the original progress report](evidence/pilot-003-progress.json).

The focused replay is post-selected and starts with preloaded history and fresh
notebooks. It verifies a correction, not comparative performance: Gnomon completed
the task in four API calls, while SQL exhausted its eight-turn budget. The final
runner adds saved-query inspection for the control. That final discovery change
has regression coverage but has not had a full longitudinal live rerun.

Across these runs, 505 API requests were started; 501 responses and **2,266,527
billed tokens** are retained. Four interrupted requests have unknown usage;
the true total and dollar cost are unknown. Earlier SQLite live query timings
were not populated correctly; the final runner fixes that instrumentation.
Do not compare those earlier zero timing values as measured performance.

## Next iteration and confirmation

Run a fresh full development iteration with the final runner, both agent seeds
and all 24 rounds. Classify remaining failures before changing infrastructure.
Track cold-start and mature rounds separately. Only after the candidate meets
the validation criteria should the guarded confirmation run open its reserved
worlds. Do not retune on confirmation or reuse a consumed cohort under a new name.

The benchmark now makes the target measurable; it has **not** demonstrated that
Gnomon meets it. Forecast-error improvement is separately reported. A later
real-data longitudinal replication is needed for claims about retail usefulness.

## Reproduction and evidence

[README](README.md) has executable audit, pilot, progress and freeze commands.
[JOURNAL](JOURNAL.md) preserves the iteration rationale.
[Artifact index](evidence/artifact-index.json) hashes every retained raw file and
the compressed bundles below. Bundles contain synthetic databases, notebook
states, per-checkpoint decisions and API wire transcripts; no credentials or
virtual environments. The API key was checked absent before export.

* [Pilot 001](evidence/pilot-001-evidence.tar.gz)
* [Pilot 002](evidence/pilot-002-evidence.tar.gz)
* [Pilot 003](evidence/pilot-003-evidence.tar.gz)
* [Focused replay 004](evidence/replay-004-evidence.tar.gz)

Use the listed source commit to inspect an old run. Use a fresh output directory
to rerun; the seeded event stream is reproducible, but API answers and execution
IDs can differ. Keep the original evidence untouched.
