# Episodic memory: forecasting benchmark results

Evaluation date: **6 October 2026**.

Gnomon's simple episodic-memory router reduced forecast error by **8.54% on
average across six benchmarks**, compared with the same router without episodic
memory. The **95% benchmark-bootstrap interval was 2.84%–16.17% improvement**.
Memory improved five benchmarks; electricity regressed by 0.74%.

This measures the benefit of using similar past forecasting episodes to select
among a fixed model pool. Both arms use the same candidate predictions, outcome
availability rules and cross-series pooling. The no-memory arm still learns from
recent forecast errors; it is not a fixed naive forecast.

## What the tasks represent

Each task asks Gnomon to forecast future numerical values from past observations,
choosing among the same nine candidate models. The counts below describe our
evaluated subsets, not the full source datasets.

| Task | What the data measures | Forecasting task in this evaluation |
| --- | --- | --- |
| Electricity consumption | Electricity use recorded for individual clients. [Source: UCI](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014). | Predict the next **30 days** for each of 30 daily series. |
| Traffic speed from taxi traces | Road traffic speeds derived from taxi trajectories in Shenzhen. This measures speed, not taxi demand. [Source: LibCity dataset documentation](https://bigscity-libcity-docs.readthedocs.io/en/latest/user_guide/data/raw_data.html). | Predict the next **48 hours** for each of 30 hourly road-segment series. |
| Weather measurements | Meteorological observations from Jena, Germany. [Source: Max Planck weather data](https://www.bgc-jena.mpg.de/wetter/weather_data.html). | Predict the next **96 hours** for each of 21 hourly measurement channels, treated as separate series. |
| Application operations | Application observability measurements from a business and IT monitoring dataset. [Source: BizITObs](https://github.com/BizITObs/BizITObservabilityData). | Predict the next **10 minutes** for each of two channels sampled every ten seconds. |
| Hospital patient counts | Monthly patient counts for products related to medical problems; the series do not represent separate hospitals. [Source: original dataset documentation](https://pkg.robjhyndman.com/expsmooth/reference/hospital.html). | Predict the next **12 months** for each of 30 monthly series. |
| Retail sales | Daily quantities sold for individual pasta products at an Italian grocery store. [Source: UCI](https://archive.ics.uci.edu/dataset/611/hierarchical%2Bsales%2Bdata). | Predict the next **30 days** for each of 30 daily product series. |

The mix covers observations from ten seconds to one month apart, and forecast
horizons from ten minutes to one year. It tests whether the same memory router
helps across different domains and timescales. It does not establish why memory
helps or hurts any particular domain.

Candidate models forecast each series from its own history. Weather channels
and application channels are forecast separately; the experiment does not use
road-network graphs, retail promotion inputs or hierarchical reconciliation.
Memory pools past model-selection evidence across series **within each task**.
The prepared dataset variants are described in the
[GIFT-Eval paper](https://arxiv.org/html/2410.10393v1); our evaluation protocol and
sample sizes are specified below.

## Results by task

Lower MASE is better. Positive improvement means memory reduces error:
`100 × (1 − memory MASE / no-memory MASE)`.

| Task | Dataset | No-memory MASE | Memory MASE | Improvement |
| --- | --- | ---: | ---: | ---: |
| Electricity demand | `electricity/D` | 1.5763 | 1.5880 | −0.74% |
| Taxi traffic speed | `SZ_TAXI/H` | 0.4999 | 0.4748 | **5.01%** |
| Weather | `jena_weather/H` | 0.8896 | 0.8067 | **9.32%** |
| Web operations | `bizitobs_application` | 0.5351 | 0.3939 | **26.38%** |
| Hospital demand | `hospital` | 0.7767 | 0.7507 | **3.35%** |
| Retail sales | `hierarchical_sales/D` | 0.8979 | 0.8265 | **7.95%** |

Each benchmark receives equal weight in the average, regardless of how many
series it contains. MASE is mean absolute error scaled by a fixed, pre-scoring
seasonal-naive history scale, averaged across scored origins within each series
and then across series.

| Aggregate | Error reduction | 95% benchmark-bootstrap interval |
| --- | ---: | ---: |
| Arithmetic mean of per-benchmark improvements | **8.54%** | **2.84%–16.17%** |
| Geometric mean error-ratio improvement | 8.99% | 2.92%–16.81% |

The bootstrap resamples **whole benchmarks**, preserving the dependence among
series and overlapping forecast windows inside each benchmark. We enumerate all
46,656 ordered samples of six benchmarks drawn with replacement, and use the
2.5th and 97.5th percentiles with linear interpolation. This describes variation
across these six benchmark results; it is not a within-benchmark time bootstrap
or a guarantee for unseen tasks.

## What was tested

The CPU model pool contained nine configurations: last value, seasonal naive,
drift, AutoETS, Theta, Croston-SBA, ridge regression, random forest and histogram
gradient boosting. The baseline model remained eligible for selection. Every
candidate was shadowed, and an episode became retrieval evidence only after its
full forecast horizon ended. Predeclared seasonal-naive fallback predictions
were retained for failed candidates, including 72 Croston failures on negative
weather histories.

The memory arm used the `levels` feature profile, robust distance, 32 neighbours,
cross-series pooling and equal own-series weight. Minimum effective neighbour
count was eight. Regime matching, recency weighting, deduplication, shrinkage,
confidence penalties, novelty rejection and switch penalties were off.
These are the **evaluated settings**, not a claim that every installed release
uses them by default. See [adaptive routing](adaptive-routing.md) for the API.

| Task | Series | Forecast horizon | Scored windows | Memory history window |
| --- | ---: | --- | ---: | ---: |
| Electricity | 30 | 30 days | 600 | 512 daily observations |
| Taxi | 30 | 48 hours | 600 | 276 hourly observations |
| Weather | 21 | 96 hours | 420 | 512 hourly observations |
| Web operations | 2 | 60 ten-second steps (10 minutes) | 40 | 512 observations |
| Hospital | 30 | 12 months | 600 | 37 monthly observations |
| Retail | 30 | 30 days | 600 | 512 daily observations |

Each series had eight validation, eight warmup and twenty scored origins. Forecast
horizons were task-specific. Memory's long history window was capped using the
history available at the first decision and then held fixed throughout replay;
this corrected infeasible windows on hospital and taxi without tuning on outcomes.

The data came from `Salesforce/GiftEval`, pinned to revision
`30841734ac5cfddbd0c3bad6d09d2b6b32becbb0`. This was a custom rolling-origin
evaluation, **not the official GIFT-Eval leaderboard protocol**.

## Missing values and corrected coverage

Retail includes **all 600 planned scored windows**. Of 18,000 target positions
across those windows, 17,580 were observed: **97.67% target coverage**. Losses use
the same observed-position mask for every model. Missing actuals are not imputed,
and feedback still waits until the original 30-day horizon ends.

This supersedes the earlier retail result of 10.49% improvement, which discarded
whole windows containing any missing actual and scored only 240 of 600 windows.
The corrected comparison restores all validation and warmup windows as well.
Hospital's earlier neutral result was also superseded: its infeasible memory
window prevented memory from activating. Both corrections are included above.

## Interpretation and scope

These results support episodic memory as a useful model-selection mechanism
across the tested tasks. They measure router memory, not agent reasoning,
narrative recall, a full FASE reproduction, or Ephemeris forecasting accuracy.
Alpha, crypto and Favorita are separate experiments and are not included in this
six-benchmark average or bootstrap.

The datasets and configurations have been inspected during development. The
broader experiment evaluated 8,192 configurations per task, but this table reports
the same simple configuration across tasks rather than selecting a winner for
each dataset. The bootstrap does not remove development-selection bias. Fresh
tasks and deployment outcomes remain necessary to establish generalization.

## Evidence and reproduction

- [Exact scores, settings, coverage and bootstrap results](../results/memory-benchmarks-20261006/summary.json).
- [Dataset and runtime fingerprints](../results/memory-benchmarks-20261006/provenance.json).
- [Standalone bootstrap calculation](../results/memory-benchmarks-20261006/bootstrap.py).

From the repository root, recompute the aggregate estimates and verify them
against the saved results with no third-party dependencies:

```bash
python3 results/memory-benchmarks-20261006/bootstrap.py
```

This reproduces the aggregate statistics from the saved benchmark scores. It does
not regenerate the underlying forecasts; that also requires the pinned runtime,
data and original forecast receipt archives, which are not bundled on this page.
