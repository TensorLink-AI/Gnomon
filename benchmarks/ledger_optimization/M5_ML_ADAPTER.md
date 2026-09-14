# Synthetic-verified M5 adapter for the authorized ML task

This amendment concerns preparation, not a final dispatch or a new favorable
population. Implement and test it using synthetic rows only while guarded093
development runs. Do not inspect the 24 reserved series' later targets. The
source, store/item selection, eligibility rule, first origin and 26 consecutive
14-day horizons from `M5_PROTOCOL_014.md` remain fixed.

The earlier fixed-recipe preparation supplied 366 observations at its first
origin. The user-authorized ML workflow instead uses 730 history observations,
three rolling backtest folds, and the same seasonal/Ridge/Random Forest search
space in all three arms. Reusing the short preparation would fail the frozen
host's history contract. Supply the missing earlier history; do not move the
forecast dates, fabricate observations, or alter the selected panel.

The first origin remains d_1577. Its history is d_848 through d_1577 inclusive.
The last origin remains d_1927 and its targets end d_1941. Each origin uses only
its preceding 730 observations. Eligibility and its retained initial-history
hash still concern d_1212 through d_1577. If the additional early history is
missing or invalid, stop preparation and report it; never replace that series
or recompute eligibility using forecast outcomes.

Keep the existing period-end timestamp convention: reporting date plus one day,
00:00 UTC. History source/recording availability is assumed at that period-end;
an earlier forecast's complete-horizon score becomes usable only when its last
target has arrived. These are shared replay assumptions, not measured vintages.
Future covariates contain only calendar weekday sine/cosine and the declared
unavailable promotion input set to zero. No future sales or realized prices
enter forecast requests. The original source fixture hashes remain available.

The implementation's CLI exports only the eight already selected development
series and requires the pinned original manifest and source archive. It has no
reserved-data export switch. Pure preparation helpers are tested against synthetic
fixtures, including poisonous reserved-target values, future perturbations,
unchanged initial-history hashes, chronological boundaries, and invalid early
history. Preparation makes zero provider calls and zero Engy requests.

`build_series_jobs` is a shared pure constructor with no file or network access.
The development exporter calls it only after checking fixed development identity.
A future final exporter must use this same constructor behind its final access
gate, so the final requests cannot silently diverge from the tested preparation.
It requires the full daily grid, midnight UTC timestamps, nonnegative numerical
observations and the unchanged selection-prefix hash. Synthetic equivalence
checks retain the exact pre-refactor 208-task output hash.

This does not itself satisfy the final gate. A later final host adapter must
preserve these rules, use the same tested implementation in all three arms,
freeze its input and source hashes, and pass the numerical/visibility checks
before any reserved forecast execution. Final dispatch still requires a complete
matched development result reaching the 20% point target, the frozen final
protocol, and no unresolved fairness or integrity failures. Store-cluster and
shared four-origin time-block inference from the original M5 protocol remains
required; seed repetitions do not create independent stores. The version and
three-arm amendments in `PLAN.md` supersede historical 1.1.9/fixed-recipe wording.

Do not present synthetic adapter tests, the cold-start pilot, or development
scores as proof that the final accuracy objective has been achieved. Main and
PyPI remain unchanged.

## First real development preparation

After synthetic checks and before any M5 ML model execution, prepared the original
8 development series from the hash-pinned archive using the unchanged constructor.
All 208 jobs satisfy the 730-row history contract. Original development overlap:
112,528 history-value comparisons match the retained original export; all future
horizons and original selection hashes match. Calendar boundaries are 2013-05-26
first history, 2015-05-25 first origin, 2016-05-09 last origin and 2016-05-23 last
target, expressed as period-end UTC. The earlier dates in synthetic fixtures
belong only to those fixtures and are not real M5 date claims.

The frozen v6 host prepared the first and last origin of each development series
in all three arms, plus host-only future-target perturbations: 96 preparations,
832 checks. Arm-visible numerical inputs agree byte-for-byte, and changing future
sales changes none of them. This is preparation-only with empty prior records,
not a model, resumed-history or accuracy test. No provider/Engy requests occurred,
no reserved-target numerical cells were accessed, and the running Favorita trial
was unchanged. Receipt: `evidence/m5-ml-development-prepare-001.json`.

The development input file is retained under ignored experiment storage, with
its source and output hashes; it is not a new selected panel and is not a paid
run launch. The final gate remains closed.
