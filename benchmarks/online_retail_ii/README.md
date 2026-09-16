# Online Retail II: real-sales benchmark

This benchmark asks whether Gnomon's execution and evidence interface helps an
equally equipped agent, and whether a ledger improves its choices as outcomes
accumulate. It uses real transaction data from the user-supplied
`online+retail+ii.zip`, not synthetic demand curves.

The executable deliverable includes data preparation, ten numerical candidates,
three selection controls, task exports, direct/Gnomon execution adapters, actual
Gnomon ledger replay, and a scorer that retains unsuccessful cases. **The paid
three-arm Hermes experiment has a separate guarded runner**, described in
[agent_eval/RUN.md](agent_eval/RUN.md). A baseline run or imported-submission
score is never presented as a Hermes experiment.

The [frozen protocol](PROTOCOL.md) is the authority for selection, splits and
metrics. Do not revise it to fit observed scores. This is one retailer with
repeated products over time; it cannot establish transfer to other retailers.

## Task and competitors

Predict the next 14 daily **recorded gross sales units** for each product, then
advance the origin 14 days. Zero sales is not evidence of stockouts. Returns,
cancellations, non-product fees and nonpositive-price transactions are excluded
with counters. No customer identifiers are exported. Sheet ownership is split
at December 1, 2010 so overlapping sheets do not double-count transactions.

The training-only cohort contains 48 products: 16 each in sparse, intermittent
and frequent-sales strata. Eligibility and hash-based selection use only sales
through November 30, 2010. Discontinued products remain in the evaluation.

| Split | Origins per product | Cases | Last target date |
|---|---:|---:|---|
| Development | 13 | 624 | 2011-06-05 |
| Validation | 6 | 288 | 2011-08-28 |
| Final | 7 | 336 | 2011-12-04 |

Primary score is arithmetic mean per-case RMSLE. MAE, seasonal MASE, WAPE,
bias, clipping and failures remain available. All current cross-validation and
historical evidence uses RMSLE for selection: there is no MAE/RMSLE objective
mismatch. Undefined MASE denominators are reported, not silently replaced.

The fixed competitors are last value, weekly seasonal naive, 28-day mean,
four-week weekday mean, AutoETS, AutoARIMA, AutoTheta, Croston SBA, log-sales
lagged Ridge, and log-sales lagged histogram gradient boosting. Strong automatic
controls are selection using three current rolling CV folds and a log-space
ensemble of the three CV-best models. Every arm uses the same implementations.
The recent-four-origin historical selector is an additional diagnostic, **not a
Gnomon or Hermes result**. A historical selector beating seasonal naive alone
would not establish ledger value.

## Reproduce the development run

Run from the repository root with Python 3.12. Use a fresh environment and fresh
output directories. Tests use `-c /dev/null` to avoid the repository's test
configuration importing the working Gnomon source instead of the pinned wheel.

```bash
python3 -m venv /tmp/retail-eval-venv
/tmp/retail-eval-venv/bin/python -m pip install -r benchmarks/online_retail_ii/requirements.lock
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/tmp/retail-eval-venv/bin/python -m pytest -c /dev/null benchmarks/online_retail_ii/test_benchmark.py -q
/tmp/retail-eval-venv/bin/python -m benchmarks.online_retail_ii prepare \
  --archive online+retail+ii.zip --output /tmp/retail-panel
/tmp/retail-eval-venv/bin/python -m benchmarks.online_retail_ii run-baselines \
  --panel /tmp/retail-panel --output /tmp/retail-smoke --smoke
/tmp/retail-eval-venv/bin/python -m benchmarks.online_retail_ii run-baselines \
  --panel /tmp/retail-panel --output /tmp/retail-baselines
```

The tested index did not expose Gnomon 1.2.0. The initial failed install was
preserved; execution used the previously verified cached 1.2.0 wheel, SHA-256
`030a5cd063c424482bebdc2a522aa98a8f4bf88285bab31c4e47ddc9afa04f2f`.
If needed, provide its directory using pip's `--find-links`; do not substitute
1.1.9 or a working-source installation. The remaining package versions are in
`requirements.lock`; the run plan records imported distribution versions.

Preparation checks the source ZIP SHA-256. It materializes only the development
period by default. Validation/final preparation requires a separately frozen
release manifest binding the source, protocol, candidate commit and runtime
lock. The baseline runner currently rejects non-development panels. These
guards prevent accidental promotion; they are not a security boundary against
a caller who can edit the code. Implement and freeze the later-phase runner
before opening those periods.

`plan.json` records source, panel, protocol and Python-file hashes, runtime and
planned cases. `host-scores.jsonl` contains every forecast and scoring result.
`report.json` is written only after completion and a source-drift check. Preserve
interrupted attempts and their errors; use a new directory for a retry.

## Agent integration and scoring

Each `agent-cases/SKU-ORIGIN/` directory contains only:

- `task.json`: identity, origin, horizon, candidate names, units and fingerprint.
- `history.csv`: sales visible through the origin.
- `current-cv.json`: three completed CV folds and RMSLE rankings.
- `matured-outcomes.json`: original predictions and actuals from completed prior
  origins, available identically to every arm.

Give the agent access to **one current case only**. The parent directory contains
later cases whose histories reveal later sales. Do not mount the source ZIP,
panel, host scores, other cases or host-owned execution records read/write into
the agent. Task exports themselves are not an access-control sandbox.

The host's authorized tool handler can execute the same provider directly or
through published Gnomon 1.2.0:

```bash
/tmp/retail-eval-venv/bin/python -m benchmarks.online_retail_ii execute \
  --case /tmp/retail-baselines/agent-cases/20757-2010-12-05 \
  --provider ridge_log --backend gnomon --output /tmp/retail-executions
/tmp/retail-eval-venv/bin/python -m benchmarks.online_retail_ii resolve \
  --case /tmp/retail-baselines/agent-cases/20757-2010-12-05 \
  --executions /tmp/retail-executions
```

Use `--backend direct` for Hermes without Gnomon. Execution records bind the
forecast to the case, request fingerprint, series, units and forecast dates.
One matching successful execution is recoverable without final JSON; multiple
executions require `--selection EXECUTION_ID`. An explicit conflicting reference
is rejected. There is no prose scraping or provider-name guessing. Numerical
fallbacks are retained and labelled even when the tool itself succeeds.

For the ledger treatment, the host can build Gnomon's historical evidence from
the **same matured outcomes available to the controls**:

```bash
/tmp/retail-eval-venv/bin/python -m benchmarks.online_retail_ii ledger \
  --case /tmp/retail-baselines/agent-cases/20757-2010-12-19 \
  --output /tmp/retail-history
```

This uses public `TemporalLedger`, `InferenceEngine`, `compare_history`,
execution and actuals reads. It replays previously calculated predictions with
an explicitly simulated historical clock and makes zero numerical refits or
Engy calls. The data lacks real recording timestamps. This tests an accumulated
evidence interface, not genuine historical source revision visibility.

Gnomon's native history comparison metric is MAE and its query limit is eight
providers. The adapter queries pairs against a common anchor, requires identical
origins, steps and actual IDs, and calculates RMSLE from those public referenced
pairs. It never relabels MAE as RMSLE. Native reports and the SQLite ledger are
retained alongside lifetime/recent rankings and exclusions.

Submit host-owned execution-directory references as a JSON array:

```json
[
  {
    "case_id": "20757-2010-12-05",
    "executions": "/tmp/retail-executions",
    "execution_id": "COPY_THE_RETURNED_ID"
  }
]
```

```bash
/tmp/retail-eval-venv/bin/python -m benchmarks.online_retail_ii score \
  --panel /tmp/retail-panel --baseline-run /tmp/retail-baselines \
  --submissions /tmp/submissions.json --arm gnomon --output /tmp/retail-score
```

Allowed arm labels are `hermes`, `gnomon`, and `ledger`. Missing or ambiguous
submissions retain the planned case and use a disclosed weekly-naive fallback.
The scorer checks identities against the host panel and the required execution
backend. It does **not** prove that a human/agent used a particular prompt,
actually consulted the ledger or complied with budgets; its report says so.

## Admission requirements for the paid comparison

Use three matched arms (Hermes, Hermes+Gnomon, Hermes+Gnomon+ledger), Gnomon 1.2.0,
DeepSeek v4.1 Flash through Engy and requested seeds 7 and 19. Before dispatch,
freeze prompts, call/token/time budgets, package and tool access, native-memory
policy and the exact release. All arms have the same numerical models and raw
past information; only the execution/evidence interface differs. Run origins
sequentially and release outcomes only once they mature. Record tool/API calls,
token usage including unknown usage, failures, corrections, latency, typed
completion and strict final-answer conformance separately. Never fabricate costs.

The admission runner must enforce host-owned execution records and per-case
filesystem isolation, audit equal access/budgets and preserve raw transcripts.
This integration is intentionally not represented as completed by a baseline
run or the imported-submission scorer.

The agreed final success target remains at least **20% lower mean RMSLE than
matched no-ledger Gnomon**, with a paired 95% interval excluding zero, while also
reporting performance against the strongest automatic control selected on
validation. Use the clustered/block bootstrap specified in the protocol, report
cold starts and mature periods, and preserve every planned case. Development
scores cannot satisfy this target. Do not promise the data will support it.

Source: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online%2Bretail%2Bii),
Daqing Chen, DOI 10.24432/C5CG6D, CC BY 4.0. The ZIP and raw transactions are not
committed with this harness.
