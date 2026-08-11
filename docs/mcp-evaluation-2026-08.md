# Gnomon MCP Evaluation — 2026-08 (Verified)

Two parts, then a disposition. Part 1 condenses the external evaluation of
Gnomon 0.5.0's MCP surface submitted by the Hermes Agent (Nous Research)
on 2026-08-11 — 49 tool calls over raw JSON-RPC against `gnomon mcp serve`
stdio, across the repo examples, six FRED macroeconomic datasets, and a
seeded synthetic server-metrics set. Part 2 is an independent in-repo
verification of every checkable claim, with corrections. Part 3 records
what was changed in response, on this branch.

- **Evaluated head:** the `claude/gnomon-main-evaluation-smtc1d` branch
  (`511e634` atop `464b387`) — i.e. `main` *plus* the anomaly-speed /
  mapping-objects / `suspected_cause` follow-up, which the evaluation's
  own findings exercise. The report's header says "atop main, 464b387",
  which understates its actual head.
- **Report date:** 2026-08-11, FRED data retrieved the same day.

---

## Part 1 — External Evaluation (condensed)

*Summarised from the submission. Numbers here are the evaluator's; see
Part 2 before quoting them.*

Every test produced a valid result: most passed on first attempt, the
rest returned structured errors whose `repair_options` led to a clean
retry after data cleaning (missing dates, blank targets, explicit
frequency). Headline positives:

- **Structured errors with machine-readable repair options** on every
  failure encountered (`AMBIGUOUS_FREQUENCY`, `IRREGULAR_TIME_GRID`,
  `INVALID_TARGET`, `DUPLICATE_TIMESTAMPS`) — "the agent never had to
  guess."
- **Epistemic disclosure held up in the field**: degraded /
  conditionally_supported / underpowered states, structured warnings and
  assumptions, non-influence disclosure on `suspected_cause` (verified
  by the evaluator as byte-identical results with and without it).
- **Model selection behaved credibly on real data**: `ets` on CPI,
  `theta` on housing starts, `drift` on industrial production, and —
  the evaluator's controlled check — `linear_trend` correctly selected
  on a synthetic memory series with a seeded 0.02/step leak.
- **Investigations found real structure**: the 1997 regime shift in 107
  years of industrial production (80% of variance), the 2008-02 CPI
  shift, the 2002 vehicle-miles shift.
- **The anomaly grading-window fix landed**: 943-row UNRATE anomaly
  detection completed in ~18 s.

Reported friction, in the evaluator's priority order:

| # | Item | Evaluator's severity |
|---|---|---|
| 1 | TSFM models disclosed by `gnomon_capabilities` but installable only from a shell — no MCP tool | HIGH |
| 2 | Anomaly-detector selection still slow beyond ~10k rows (23,594-row daily series not attempted within a 300 s interactive timeout; `investigate` on it was also killed) | MEDIUM |
| 3 | Monthly first-of-month data needed an explicit `frequency="MS"` | friction |
| 4 | Business-day data (Treasury yields) needed weekend forward-filling before the daily grid was accepted | LOW |
| 5 | Multi-target response-budget truncation "invisible" — agent must compare returned steps against the requested horizon | friction |
| 6 | `suspected_cause` has no effect on results (by design; may surprise users) | LOW |
| 7 | Flat/near-flat series select `last_value`, disclosed as `selection_underpowered` but "often wrong for trending data" | friction |

---

## Part 2 — Verification

Checked against this repo at the evaluated head. ✓ = reproduces; ✗ =
wrong; ~ = not checkable in-repo (depends on the evaluator's data or
timing environment).

| Claim | Verdict | Evidence |
|---|---|---|
| Gnomon version 0.5.0 | ✓ | `gnomon.__version__` |
| 17 tools on the default (`full`) surface at that head | ✓ | `toolspec.TOOLS`; 24 pre-merge only under `GNOMON_V02_COMPAT=1` |
| Profiles narrow 17 → 7 (`core`) | ✓ | `toolspec.PROFILES`; core is exactly 7 tools |
| 7 eligible TSFMs disclosed, install CLI-only | ✓ | `available_tsfms()` returns 7; no install tool existed on the surface |
| Anomaly grading windowed to trailing 1,024 obs, ≥4 seasons | ✓ | `anomaly.MAX_GRADING_HISTORY = 1024`, stretched to 4 seasonal periods |
| `suspected_cause` recorded verbatim, influence "none", identical results | ✓ | contract tests pin byte-identical results when omitted |
| Monthly first-of-month data with any gap demands explicit `frequency="MS"` | ✓ | `infer_frequency` accepted only *contiguous* month-start series; one hole fell through to `AMBIGUOUS_FREQUENCY` |
| Daily grid rejects weekend gaps; forward-fill required | ✓ | `validate_and_group` requires a continuous calendar grid for `D` |
| Multi-target truncation is invisible | ✗ | Trimmed responses set top-level `truncated: true` plus a `truncation` object recording every trimmed path with its full length, and each forecast result carries `forecast_rows` (the untrimmed count) next to the trimmed array — pinned by `test_multi_series_forecast_respects_the_budget`. The signal exists; the evaluator's own Phase-3 row cites `truncated:true`. |
| "42 tool calls … 34 passed first attempt, 8 after cleaning" (exec summary) | ✗ | The report's own aggregate table sums to 51 tests (49 passed, 2 interrupted); its per-tool table sums to 49 calls. The exec-summary figures don't reconcile with either. |
| "100% pass rate" including DGS10 | ~ | Two DGS10 calls were killed at the evaluator's own 300 s timeout; counting a series with 2 of 4 tests interrupted as 100% is generous bookkeeping, not a Gnomon result either way. |
| Anomaly counts / regime-shift dates on FRED series | ~ | Requires the evaluator's retrieved data; not reproducible in-repo. Internally consistent with the tools' disclosed selection bases. |
| Structure of the report | note | The table of contents promises 13 sections including "Phase 4: Feature Testing — Newly Landed Code" and "Data Quality Incidents"; neither exists as a section in the body, and the appendix lists three datasets (car sales, daily minimum temperatures, daily births) no phase reports on — though pain-point 7 references the car-sales run. |

---

## Part 3 — Disposition (this branch)

| Item | Action |
|---|---|
| TSFM install from MCP (HIGH) | **Done.** New `gnomon_install_tsfm` tool (default surface, `full` profile). Because `ensure_sandbox` legitimately runs for minutes (torch dominates) and a blocking MCP call reads as a hang, the tool starts a *detached* `gnomon tsfm install` process and reports state — `absent` / `installing` / `ready` / `failed` (with log tail) — on each call; `status_only` polls without side effects. The `.gnomon-sandbox-ready` marker remains the single source of truth, shared with the CLI path. `gnomon_capabilities` now names the tool under `models.tsfm_install_tool`. |
| Month-start needs explicit `frequency="MS"` | **Done.** `infer_frequency` now recognises a month-start grid *with holes*: every timestamp on the first of a month at one shared time of day, single-month steps a strict majority (so yearly and quarterly data are not mistaken for gappy monthly). The series then reaches the grid validator, which names the missing month with repair options — the same path an explicit `MS` always took, now without the caller having to know the incantation. |
| Business-day daily data (LOW) | **Diagnosed, not regridded.** When a daily-grid refusal's gap covers only weekend days, `IRREGULAR_TIME_GRID` now says so in the message, sets `details.gap_weekend_only: true`, and `repair_options` carries a `fill_business_days` action (forward-fill non-trading days upstream, `repair=aggressive`, or resample weekly). A true business-day (`B`) grid — with its own season and holiday calendar — remains future work; the refusal now teaches the caller the shape of their data instead of just its first hole. |
| Anomaly speed beyond ~10k rows (MEDIUM) | **Open.** The 1,024-observation grading window covers the 1–5k range the evaluation measured; very long daily series remain impractical for detector *selection*. Candidate next step: grade on a decimated or windowed replica for selection only, keeping full-series scoring for the chosen detector. |
| Multi-target truncation visibility | **No change — already visible.** See Part 2: `truncated`, `truncation.trimmed` (per-path full lengths), and per-result `forecast_rows` all disclose the trim. |
| `suspected_cause` is documentary (LOW) | **Standing design.** A hypothesis must not steer detection; a *dated* suspicion belongs in `context_events`, where it competes as a concurrent-event explanation and is ranked. The parameter descriptions already say this. |
| `last_value` on near-flat series | **Standing design.** Selection is a backtested competition; when nothing beats persistence within the evaluated folds, publishing persistence *labelled underpowered* is the honest output. A trend the folds cannot distinguish from noise is not one Gnomon should assert. |
