# Matched Hermes run: Online Retail II development

This extends the frozen numerical benchmark with real Hermes sessions. It uses
the ten existing candidates and precomputed, past-only current CV. This is a
model-selection task with bounded forecast execution, not unrestricted model
code authoring. No validation/final observations are opened.

## Frozen settings

| Setting | Value |
|---|---|
| Agent model | `deepseek-v4.1-flash`, Engy |
| Gnomon | Published/cached exact 1.2.0 wheel |
| Arms | Hermes/direct; Hermes+Gnomon; Hermes+Gnomon+ledger |
| Cases | 48 products × 13 origins = 624 per arm/seed |
| Requested seeds | 7, 19; provider determinism is not assumed |
| Total sessions | 3,744 |
| Model request budget | 12 per session, including corrections and failed requests |
| Maximum output tokens | 3,072 per request |
| Numerical budget | Four forecast attempts per session |
| Time budget | 480 seconds per session, plus bounded shutdown |
| Correction budget | At most two turns within the original limits |
| Memory | Native Hermes text memory/skills; isolated per product/arm/seed; persists between origins |
| Selection | Host-owned typed execution ID; single matching execution recoverable without final JSON |
| Concurrency | Six sessions; each product/arm/seed advances sequentially |

The same current CV scores, visible history and matured original prediction/
actual pairs are available to every arm. The ledger arm additionally queries
validated lifetime/recent RMSLE summaries. Historical records cover the common
fixed candidates, not just the models previously selected by that arm. This is
explicitly simulated replay with assumed day-close availability.

Numerical calculations run in a separate environment matching the completed
baseline package versions. The Hermes control runtime lacks Gnomon. Its
numerical child uses the direct backend; although the shared numerical runtime
has Gnomon installed, the control cannot invoke that interface. No arbitrary
terminal, Python, filesystem or delegation tool is exposed. This is a guarded
tool boundary, not an operating-system sandbox against hostile Python code.

## Quality checks and continuation

First run `agent_eval.run --stub` with the actual pinned Hermes runtime. Its
local synthetic transport exercises six sessions: three arms over two origins,
including native memory, real numerical execution and explicit selection.
It makes zero Engy calls and is not a performance result. Preserve its plan,
source hashes, runtime inventory, transcripts, exit status and FINISHED receipt.
The launch supervisor must compare those sources to the paid bundle before
reading credentials. Do not reuse a stale preflight after changing code.

The paid controller then starts a 36-session pilot: one fixed product from each
stratum, two origins, three arms, two seeds. Continuation requires every pilot
worker to exit zero and at least 90% to resolve a task-matching execution. This
is a completion gate, **not an accuracy gate**. All pilot outcomes, including
failures, remain in the full result and their memory is retained. If the gate
fails, stop and retain the failed attempt; do not selectively rerun it.

After passage, the controller executes the remaining 3,708 sessions using the
same code and settings. It never reruns completed pilot cases. Within each
stage it completes an origin before advancing; memories never cross products,
arms or seeds. Missing or ambiguous selections use the same weekly-naive
fallback and remain in the denominator.

## Example command

Use a fresh destination and an isolated copy of the frozen bundle. The baseline
plan/report/cases and development panel must match the original fingerprints.

```bash
NUMERICAL_PYTHON -m benchmarks.online_retail_ii.agent_eval.run \
  --repo FROZEN_REPO \
  --baseline FROZEN_REPO/results/online-retail-ii-baselines-001 \
  --manifest FROZEN_REPO/results/online-retail-ii-development-001/manifest.json \
  --panel FROZEN_REPO/results/online-retail-ii-development-001 \
  --plain-python HERMES_PLAIN_PYTHON --gnomon-python HERMES_GNOMON_PYTHON \
  --output FRESH_OUTPUT --credentials HOST_ONLY_ENV_FILE --workers 6
```

For preflight, replace `--credentials HOST_ONLY_ENV_FILE` with `--stub` and use a
different fresh output. Credentials are read only by the host proxy; Hermes
receives a local placeholder key. No credentials are stored in the bundle or
API request logs.

`plan.json` freezes source/runtime/settings. `progress.json` is provisional and
may have unequal per-arm counts; compare only matched completed cases when
reporting a partial result. `scores.jsonl` retains individual attempts and
fallbacks. Per-session folders preserve prompts, API payloads, reported/unknown
usage, tool events, numerical subprocess logs, typed execution records and final
selection. Strict final JSON conformance is separate from resolved completion.
`FINISHED.json` is the terminal receipt. `INCOMPLETE.json` preserves a stopped
run's cause. No dollar cost is inferred from tokens.

The 20% ledger target cannot be established by this development run. Final
evaluation and paired uncertainty analysis remain separate, frozen steps.
