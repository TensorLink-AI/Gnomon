# Gnomon `main` Branch Evaluation — 2026-08 (Verified)

Two parts. Part 1 is an external evaluation of the changes that landed on
`main` via the merge of `claude/gnomon-harness-issues-hgerrj`, submitted by
the Hermes Agent (Nous Research) on 2026-08-11. Part 2 is an independent
in-repo verification of every checkable claim in that evaluation, run
against the same head commit, with corrections where the evaluation was
wrong.

- **Baseline:** `20d7981` (pre-merge)
- **Head:** `464b387` (`origin/main`, fast-forward merge)
- **Diff:** 124 files changed, 15,888 insertions, 714 deletions

---

## Part 1 — External Evaluation (Hermes Agent, Nous Research)

*Reproduced as submitted. Two factual errors in this section are corrected
in Part 2 — see the verification table before quoting numbers from here.*

### What Landed — By Category

#### 1. Graduated Support (New System Feature)

| Aspect | Detail |
|---|---|
| **Parameter** | `minimum_support` — values `"best_effort"` (default) or `"supported"` |
| **Behaviour** | Forecast returns tiers per row: `supported` / `conditionally_supported` / `best_effort` |
| **New field** | `headline` — deterministic plain-language sentence naming the weakest tier present |
| **Verifier** | New component rejects any claim quoting a sub-supported value without its tier label |
| **Best-effort output** | When `best_effort` is the minimum, Gnomon publishes the best result it can achieve even under short-history/degraded conditions, rather than abstaining |

The "conservative on short history" flag is partially addressed. A car
sales forecast (108 obs, 12mo horizon) now defaults to
`minimum_support: "best_effort"`, returns a `headline` like
`"conditionally_supported: degraded evaluation, selection underpowered"`
for the agent to relay verbatim, and the verifier blocks an agent from
reporting sub-supported numbers without their tier context.

Still missing: no mechanism suggests to the agent *why* the series is
short or *what to do about it* beyond the existing `recovery_actions`.

#### 2. Horizon Splitting

Auto-splits a forecast into an evaluated prefix (using the model) and a
labelled naive remainder; each forecast step carries its own tier label.
For short history + long horizon (e.g. 12-month forecast on 108 points
with only 2–3 folds), the first months are `supported` or
`conditionally_supported` and the remainder `best_effort` — far more
useful than a flat `degraded` across the entire horizon.

#### 3. Response Budget

| Aspect | Detail |
|---|---|
| **Budget** | `RESPONSE_BUDGET_BYTES = 8192` |
| **Truncation** | Oversize responses trim long arrays to first/last entries with min/max/mean |
| **Flag** | `truncated: true` set on trimmed responses |
| **Never trimmed** | Support assessments, warnings, assumptions, error payloads, repair options |
| **Pointer** | Trimmed response links to the on-disk artifact for full data |

#### 4. Brief-by-Default Forecast

`format: "brief"` is now the default (was `"full"`). Brief output carries
q50 + q10–q90 interval per step + the complete support assessment +
warnings + disclosures; `format: "full"` restores all quantile levels
inline. The on-disk artifact always carries complete data regardless.

#### 5. MCP Profiles

| Profile | Tools |
|---|---|
| **core** | `gnomon_capabilities`, `gnomon_inspect`, `gnomon_forecast`, `gnomon_investigate_change`, `gnomon_detect_anomalies`, `gnomon_get_artifact`, `gnomon_explain_run` |
| **decision** | core + `gnomon_decide`, `gnomon_monitor`, `gnomon_route`, `gnomon_status`, `gnomon_resolve_outcome` |
| **data** | core + `gnomon_ingest`, `gnomon_list_datasets`, `gnomon_submit_actuals` |
| **full** | Everything (default) |

Selected via `gnomon mcp serve --profile <name>` or
`GNOMON_MCP_PROFILE=<name>`.

#### 6. Retired Tools (Default Surface)

`gnomon_covariate_guide`, `gnomon_proposer_skill`,
`gnomon_record_decision` and `gnomon_resolve_decision` (v0.2 compat), and
a duplicate proposer-ledger entry left the default surface; v0.2 compat
tools restore with `GNOMON_V02_COMPAT=1`. *(Incomplete list — see Part 2,
correction C2.)*

#### 7. Bug Fixes (From Dogfood Review)

**F1 — Threshold probabilities inconsistent with quantiles.** On
fold-starved runs, `probability_above_per_step` (e.g. 0.61) contradicted
the published quantiles (e.g. q80=321.4 for threshold 340, implying
P ≤ 0.20): `centre_shift` was zeroed in
`conformal_spreads(recentre=False)` but `threshold_analysis_stage` still
used the raw uncentred residual cloud. Fixed by recentring the residual
cloud by the residual median unconditionally. New test:
`test_threshold_probabilities_agree_with_published_quantiles`.

**F2 — `evaluate_threshold_risk` operator crash.** The operator passed
`residual_quantiles` as `spreads` but the contract required
`dict[int, tuple]` keyed by lead step, so `spreads[1]` raised
`KeyError: 1` on every invocation — surfacing to agents as the opaque
`{"code": "OPERATOR_ERROR", "message": "1"}`. Fixed by building real
per-lead spreads through `conformal_spreads` on pooled residuals. New
test: `test_evaluate_threshold_risk_matches_the_rows_it_reads`.

#### 8. New Benchmark Agents

| File | Lines |
|---|---|
| `benchmarks/temporalbench/mcp_agent.py` | 1,088 |
| `benchmarks/cik/mcp_agent.py` | 728 |
| `benchmarks/mtbench/mcp_agent.py` | 446 |
| `benchmarks/anomllm/parallel_online.py` | 173 |
| `benchmarks/temporalbench/score_per_channel.py` | 244 |
| `benchmarks/cik/router.py` | 195 |
| `benchmarks/cik/route_survey.py` | 100 |
| `benchmarks/cik/classify_rejections.py` | 207 |

Plus 233 benchmark tests across 8 test files. Notable agent features:
`_mcp_mcq_only()` for pure-MCQ tiers, tier-preserving tool surface,
voided-row handling, wall-clock cap as a named abstention, stratified
`--limit`.

### Cross-Reference: MCP Feedback Report vs. `main`

| Issue from feedback report | Priority | Status in `main` (as claimed) |
|---|---|---|
| TSFM models advertised but unreachable | HIGH | ❌ Still open |
| Investigate_change lacks context input | MEDIUM | ❌ Still open *(wrong — see C1)* |
| Conservative on short history | MEDIUM | ✅ Partially addressed |
| Anomaly training is slow (212s) | LOW | ❌ Still open |
| Monitor output thin in response | LOW | ✅ Partially addressed |
| Covariate mapping string format | LOW | ❌ Still open |
| Agent-facing docs are dense | MEDIUM | ✅ Partially addressed |
| Brief format needed | MEDIUM | ✅ Fully addressed |
| Structured errors with repair options | HIGH | ✅ Confirmed intact |
| Cost-optimal monitor alerts | HIGH | ✅ Confirmed intact |

### External Evaluator's Summary

Test suite: 958 main tests + 233 benchmark tests = 1,191 total, all
passed. The merge brings three substantive improvements: graduated
support + horizon splitting clean up the short-history degradation UX;
brief format default reduces token pressure; the threshold probability
fix removes a silent correctness bug in decide/monitor. Remaining open:
TSFM inaccessibility from MCP, investigate_change context inputs
*(disputed — see C1)*, anomaly training speed.

---

## Part 2 — Independent In-Repo Verification

Verified 2026-08-11 against `464b387` by checking source, schemas, and
running both test suites in a fresh clone.

### Confirmed accurate

| Claim | Evidence |
|---|---|
| Diff stat 124 files / +15,888 / −714 | `git diff --stat 20d7981..464b387` matches exactly |
| `minimum_support` parameter | Present across `toolspec.py`, `pipeline.py`, `runtime.py`, `support.py`, `contracts.py`, `cli.py` |
| `headline` field, deterministic | `forecast_headline` / `artifact_headline` in `src/gnomon/support.py` |
| Horizon splitting | Present in `runtime.py`, `support.py`, `lineage.py` |
| `RESPONSE_BUDGET_BYTES = 8192` + `truncated` flag | `src/gnomon/toolspec.py:149`; trim logic at `toolspec.py:194-221` |
| Brief is the default format | `"format": "brief"` default in the forecast tool schema (`toolspec.py:454`) |
| Profile definitions and membership | `PROFILES` in `toolspec.py` matches the Part 1 table tool-for-tool; selected via `--profile` / `GNOMON_MCP_PROFILE` |
| `GNOMON_V02_COMPAT=1` restores compat tools | `toolspec.py:1618`, surfaced in `runtime.py` capabilities |
| F1 test exists and passes | `tests/test_runtime.py:140` |
| F2 test exists and passes | `tests/test_operators.py:144` |
| All 8 benchmark file line counts | `wc -l` matches every figure exactly |
| Test totals | `958 passed, 2 skipped` (main) + `233 passed` (benchmarks) = 1,191 passing |
| TSFM install not reachable over MCP | CLI-only (`gnomon tsfm install <name>`); `gnomon_capabilities` discloses the command (`runtime.py:1327`) but no MCP tool performs the install |
| Covariate mapping still string-format | `name:type:availability` strings in `contracts.py:444`, `covariates.py:127` |
| No anomaly training sampling mechanism | No subsample/downsample path in `src/gnomon/anomaly.py` |

### Corrections

**C1 — `gnomon_investigate_change` context input is NOT still open.**
The evaluation marks this "❌ Still open — Schema unchanged". The schema
changed in this merge: at baseline `20d7981` the tool had no context
parameters; at `464b387` it accepts `context_events` (inline) and
`context_events_file`, sharing the item shape with `gnomon_forecast`'s
context events. What landed is structured context events rather than a
free-text "what do you suspect changed" field, so the original feedback
may be only partially satisfied — but the schema demonstrably gained
context inputs. Correct status: **partially addressed**, not open.

**C2 — Tool counts and the retired-tool list are wrong.** The evaluation
says "20 tools in full profile instead of 24". The default `full` surface
at `464b387` is **17 tools**; the baseline surface was 23–24 (23 tool
entries plus the duplicate proposer-ledger registration this merge
removed). Seven tools — not four — moved behind `GNOMON_V02_COMPAT=1`:
the four listed in Part 1 plus `gnomon_list_open_forecasts`,
`gnomon_model_performance`, and `gnomon_propose_covariates`
(17 + 7 = 24, which is where the evaluator's "24" belongs — it is the
gated total, not the pre-merge default). Four planner tools sit behind a
separate `GNOMON_EXPERIMENTAL_PLANNER=1` gate, `full` profile only.

**C3 — "All passed" needs a footnote.** Both suites do pass, but the main
suite records 2 skips the evaluation does not mention, and the pass
requires the `dev` extra: `uv run --extra dev pytest`. Without it, PyYAML
is absent and `tests/test_future_context.py::test_config_parses_on_off`
fails — `load_config` silently returns `DEFAULT_CONFIG` when PyYAML is
missing (`config.py`), so the on/off flag reads back `False`. That silent
fallback is itself worth revisiting: a config file that exists but cannot
be parsed is quietly ignored, which is the same "setting that is quietly
ignored" failure mode `INERT_KEYS` was built to prevent.

### Verification verdict

The evaluation is substantively accurate: every feature it describes
exists and behaves as described, both bug-fix regression tests are
present and green, and the headline test totals reproduce exactly. Its
two factual errors both understate the merge — investigate_change did
gain context inputs, and the default tool surface shrank further than
claimed (24 → 17, not 24 → 20). The genuinely open items after
correction: TSFM install unreachable over MCP, free-text investigation
context, anomaly training speed, and covariate mappings as packed
strings.
