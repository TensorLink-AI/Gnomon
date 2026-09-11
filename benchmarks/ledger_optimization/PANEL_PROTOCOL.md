# New Favorita development and reserved confirmation panel

Registered before preparing/selecting the new panel or inspecting its outcomes.
The unchanged raw-source download was initiated first; it does not select cases.
This extends data access, not the 20% objective or candidate portfolio.

## Source and selection

Use the raw Favorita competition table via the same `datasetsforecast`
download as the original Arena runner. Pin the downloaded archive/table hashes,
dependency versions and Arena source commit. The recovered runner is
`TensorLink-AI/gnomon-arena`, commit
`b600eaa2c2691bebed926dac996d4b03e0c216e9`.

Use its training-only preparation with seed **20260911**, 26 consecutive
14-day forecast origins ending at the source's last complete date, at least
365 initial history days, a 200-item universe known before the first origin,
and the first 2,000 eligible item/store pairs by hash. Use up to all eligible
pairs as the candidate preparation panel, preserving the original round-robin
training-only seasonality/promotion/intermittency/volatility ordering.

Exclude every item and store appearing in the two existing local Favorita
panel manifests (`favorita-panel-12.parquet.manifest.json` and
`favorita-100-v118/panel.parquet.manifest.json`). Traverse the prepared
training-only ordering; keep at most one series per item and at most one per
store. Select **32** series. If too few qualify, report the limitation before
changing the rule. Do not reroll a seed after observing forecast outcomes.

Shuffle the 32 qualifying IDs with seed **20260911**; assign the first **8**
to additional development and the next **24** to reserved confirmation. Thus items and stores do not overlap between
development and confirmation. Store IDs and temporal blocks remain dependence
units; common calendar shocks are not assumed independent.

The full raw table can be read by deterministic preparation to write the panel,
but no confirmation forecasts, scores, outcome summaries or plots may be
computed/read before the final candidate is frozen. Save a separate confirmation
file and manifest. Report its IDs, dates, counts and hashes only.

## Additional development

First compute a provider/CV cache with the original eight pinned recipes and
two historical CV folds. Reuse exact cached requests; retain numerical failure
fallback metadata. Screen policies on only the eight development series.
Live agent runs use the same completion policy and budgets in every arm.
This is additional development; gains are not confirmation evidence.

The first live new-development comparison uses all eight development series
at origins 0, 8, 17 and 25, requested seeds 7 and 19, and no-ledger/original-MAE/
RMSLE-card arms. That is 64 matched case-seed pairs and 192 decisions. The
failed explicit-blend arm is not promoted into this run. Automatic calibration
screens use origins 0–17 for variant selection and 18–25 for development
validation; this internal slice is not the reserved confirmation set.

## Confirmation specification (pending implementation freeze)

All 24 reserved series, all 26 origins, requested agent seeds 7 and 19, three
arms (no ledger, original MAE ledger cards, selected development variant).
That is 1,248 matched case/seed pairs and 3,744 decisions. No optional stopping
on confirmation scores. Preserve source/recording maturation and cold-start
origins. Freeze the implementation, prompt and exact IDs before dispatch.

Primary score: mean per-case RMSLE, equally weighting series/origin/seed.
Paired relative reduction must be at least 20% versus no ledger, with a two-sided
95% interval excluding zero improvement. Also require positive improvement over
the original ledger. Compute a paired hierarchical bootstrap with 5,000 fixed
seed-20260911 replicates: resample series, resample shared circular four-origin
blocks to length 26, retain both agent seeds within each sampled case. Shared
time blocks retain common calendar shocks across series. This finite-panel
analysis is not a claim about all retail series or all agent models.

Report seed-specific, cold-start (origins 0–3), mature (4–25), per-series and
per-arm completion/cost results separately without changing the primary metric.
If the target fails, preserve the failure; do not tune against the confirmation
outcomes and relabel it a fresh confirmation.

## Analysis implementation

`analysis.py` implements the specified bootstrap with Python `random.Random`,
seed 20260911, 5,000 replicates and linearly interpolated percentile endpoints.
Each replicate draws series with replacement and one shared sequence of circular
four-origin blocks, truncated to the original number of origins. Both requested
agent seeds remain together inside each case; averaging them before resampling
is equivalent for this balanced arithmetic-mean metric. No arm is sampled
independently from its control.

The analyzer rejects missing/extra/duplicate decisions, nonfinite or negative
RMSLE, unresolved harness failures and nonconsecutive origin grids. Fallback
forecasts remain in the primary denominator. A zero control mean makes relative
improvement undefined; if any resample has zero control mean, its relative
interval is null and cannot pass the numerical gate. Absolute intervals are
also returned. No resamples are dropped to produce a favorable interval.

It reports the 20% point-estimate criterion, positive lower uncertainty bound,
and improvement over original ledger separately from provenance. The analyzer
always leaves `target_established: false`: verifying the frozen implementation,
untouched partition, visibility rules and matched execution is a separate
required audit. Development results cannot establish the final target.
