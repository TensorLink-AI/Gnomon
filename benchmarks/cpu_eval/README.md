# CPU product evaluation

Tests Gnomon 1.4.0 routing with a broad CPU candidate pool, and prepares a separate
agent workflow comparison using the existing matched driver. No production
behavior or default changes. No paid model call, model API or GPU is required for
the forecasting suite. Agent runs remain an explicit separately configured step.

[Current execution status](STATUS.md) and [local pilot evidence](evidence/local-pilot.json).

## Protocol and scope

[protocol.json](protocol.json) is the frozen configuration. Six configurations from
`Salesforce/GiftEval` revision `30841734ac5cfddbd0c3bad6d09d2b6b32becbb0`:
electricity/D, SZ_TAXI/H, hierarchical_sales/D, jena_weather/H,
bizitobs_application and hospital. This is **GIFT-Eval-derived online evaluation**,
not the official GIFT-Eval test split/scorer or a leaderboard submission.

Univariate channels are extracted from multivariate records; covariates are not
used. Each configuration has its own pool; channel units can differ. Scores are
normalised per series, while the released router receives the original values and
MAE evidence. Do not interpret this as multivariate or covariate-aware evaluation.
Original timezone-naive labels receive UTC for ordering, not as a claim about
where the observations were collected.

Choose up to 30 series by a fixed hash of their IDs, never outcome values. There
are 30 electricity, 30 taxi, 30 retail, 21 weather, 2 web/application and 30 hospital
series in the prepared scored manifest. Pilot uses three different reserved series
from each of electricity/taxi. Length exclusions are recorded; missing targets and
zero seasonal scales remain explicit incomplete scored cells, never replacement
series. Missing history is forward-filled causally; leading missing values use 0.

Each series has 8 validation, 8 warmup and 20 scored rolling origins. The horizon
and stride are fixed per configuration. Model history is capped at 512 observations.
The fixed baseline is chosen once from validation forecasts whose entire horizons
have matured before the earliest scored decision across the pool. The router uses
validation/warmup episodes and then receives scored outcomes only after their last
target matures. Warmup losses never enter headline scores. Historical data is not
certified unseen to pretrained models or LLMs.

Nine candidate configurations: last value, seasonal naive, drift, AutoETS, Theta,
Croston-SBA, ridge autoregression, random forest and histogram gradient boosting.
Statistical/ML settings live in [models.py](models.py) and enter the source hash.
This CPU-first scope defers pretrained foundation models rather than silently
claiming coverage of them. Croston on negative history is an explicit failed call
with the same declared fallback as any other provider error.

Compare fixed-validation model, equal-weight ensemble, no-memory routing, current
memory, bounded distance only, and full FASE-inspired mode. The baseline itself is
eligible for routing. All arms share the exact candidate forecasts. No parameter
search, winner-per-dataset tuning, LLM selector, shrinkage or switch penalty.

Primary comparison: bounded versus current memory. Scores are mean per-series
MASE, then equally weighted across configurations. The denominator uses only
history available at the first scored origin. Per-configuration paired circular
block intervals resample origin indices with all series together; block length
is at least the overlapping horizon, with 2,000 draws and seed 140. Too few time
blocks produces a null interval (not a significant claim), including the short
hospital cohort. Other comparisons are exploratory, without multiplicity
adjustment; no aggregate significance or automatic promotion is claimed.

## CPU execution

Use an isolated virtual environment on the existing Targon CPU pod. Do not replace
its other environments, stop unrelated jobs or provision a new instance. Inspect
cgroup quotas/headroom using preflight; physical host CPU/RAM is not the allocation.
One serial worker runs all models with numerical-library threads set to one. It
caches imports only; every model is freshly fitted at every origin. A timeout kills
that worker and records a fallback; the failed model call is not retried.

```bash
python3 -m venv /root/gnomon-cpu-product-v1/venv
/root/gnomon-cpu-product-v1/venv/bin/python -m pip install -e '.[dev,replay]' -r benchmarks/cpu_eval/requirements.txt
/root/gnomon-cpu-product-v1/venv/bin/python -m pip freeze > /root/gnomon-cpu-product-v1/environment.txt
```

Use that Python for the following commands from the frozen checkout:

```bash
python -m benchmarks.cpu_eval.preflight --smoke --output /root/gnomon-cpu-product-v1/preflight.json
python -m pytest -q benchmarks/tests/test_cpu_eval.py
python -m benchmarks.cpu_eval.suite --mode pilot --cache /root/gnomon-cpu-product-v1/data --root /root/gnomon-cpu-product-v1/pilot
python -m benchmarks.cpu_eval.suite --mode evaluation --cache /root/gnomon-cpu-product-v1/data --root /root/gnomon-cpu-product-v1/evaluation --pilot /root/gnomon-cpu-product-v1/pilot
```

The scored command refuses a changed runtime/protocol or a pilot with incomplete
coverage or more than 5% failed candidate calls. Pilot calls share the 30-second
per-model deadline; cumulative dispatch allowance is 1 hour per pilot dataset and
2 hours per scored dataset (12 hours over six configurations). Data download,
preparation and analysis overhead are additional and are reported separately by
suite elapsed time. This is a workload ceiling, not a cloud billing cap.

Every candidate call has an immutable request hash and result receipt. Errors and
timeouts use seasonal-naive forecasts and stay in failure counts. Thus accuracy
is for each candidate-plus-fallback system. Interrupted requests block resume;
they are not erased or automatically retried. Completed receipts can be reused
only with the same source/dependency/manifest identity. Source mutation while a
run is active invalidates it. Run each suite in a single owning process.

Outputs contain the input manifest/source hash, package versions, resource report,
per-call latencies/CPU usage, candidate failures, per-arm decisions, scores and
intervals. Total candidate-generation cost is reported; cached selector replay
latency is not presented as live forecasting cost. Missing configurations prevent
an overall score. Partial cohort scores are explicitly conditional.

## Agent comparison: prepare, then configure

```bash
python -m benchmarks.cpu_eval.agent --output /root/gnomon-cpu-product-v1/agent
```

This prepares 12 new synthetic workflow tasks and three repetitions (108 episodes),
with a seeded arm-order schedule and ordinary/lean/full experiment templates.
The existing 11-task retrospective cohort remains the 33-episode setup pilot.
Lean has only the default Gnomon tools; full adds ledger/time functionality. Ordinary
retains Python and per-task files. All three use the same model/provider access.
The agent chooses its tools and answers; no forced tool choice or host answer repair.

Tasks cover seasonal forecasting, observed statistics, matched model comparison,
call budgeting, unavailable dependencies, malformed forecast outputs, explicit
UTC offsets, scale-normalised scores, unmatured evidence and three outcome/recall
episodes. These are synthetic workflow tests, not 12 independent real-world domains.
Answer scoring alone cannot establish actual tool discovery, storage or absence
of reruns; inspect transcripts separately. No LLM judge is used.

Before execution, fill model/endpoint/credential-variable name, immutable ordinary
and service image IDs, per-arm spending allowance and the shared driver source
manifest using the [existing matched-experiment procedure](../workflow/experiment/README.md).
Build service images from this checkout; no credentials enter artifacts. Templates
have deliberately invalid model/image/cost placeholders and do not enable remote
model requests. Proposed pilot $25 and scored $100 totals are **not authorised**;
per-arm allocations and provider-side limits need confirmation. Reported-cost
stops may overshoot one operation. This module never launches agent calls.

## Current remote access

Recorded existing pod: `wrk-kadzj08j3t1o@ssh.deployments.targon.com`, key-file
`/root/.ssh/targon_gnomon_arena`. On 2026-10-05 the read-only SSH probe returned
`Permission denied (publickey)`. Updated connection details have been requested.
Do not claim a Targon run from local smoke-test results. No new instance was created.
