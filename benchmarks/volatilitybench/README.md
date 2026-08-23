# VolatilityBench

VolatilityBench evaluates future residual-dispersion claims independently of
TemporalBench. Cases sweep constant, gradual, abrupt, seasonal, and
heavy-tailed variance regimes across several signal-to-noise ratios. Oracles
are generated from held-out innovations; Gnomon receives only the prefix.

The benchmark reports scale error, QLIKE, direction accuracy, claim rate,
abstention, and interval coverage against a constant-scale baseline. Every
case has an independent seed. Direction labels use the median of 128 sealed
future continuations, avoiding the pseudo-replication and single-short-path
label noise that would otherwise dominate the test. Gnomon receives only the
one observed prefix; generator parameters and oracle continuations never enter
the compiler or runtime.

Schema 0.3 adds a direction-balanced suite. Oracle classes are equalized only
after futures are sealed, so raw class frequency cannot make an always-stable
classifier look skilled. It reports classwise recall, balanced accuracy,
multiclass Brier score, selective claim rate, and claimed accuracy. Entire
generator mechanisms (`step` and `heavy_tail`) are held out from the
development-family view. This is the directional graduation surface;
unbalanced raw accuracy remains a disclosed diagnostic, not an optimization
target.

Schema 0.4 adds a short-history suite: prefixes of 28–60 points across
flat, ramping, and abrupt-transition regimes, each with a sealed 128-draw
oracle for the future/reference residual-scale ratio. These histories are
too short for rolling-origin folds, so the suite measures the executable's
short-history fallback (reference-tail persistence) against two honest
baselines: `abstain_all` — the pre-0.4 behaviour of answering "uncertain"
on every short history, which scores zero — and `always_stable`, the
persistence null. The fallback's weak best estimate must beat both; its
`uncertain_rate` is reported so over-abstention cannot silently return.

The product contract is deliberately broader than the benchmark: the fitted
executable publishes a continuous future/reference residual-scale
distribution, probabilities over decreased/stable/increased, and a separate
decision policy. Weak best estimates remain usable for exploration, but only
out-of-fold probability skill and calibration can make a categorical answer
automation-eligible.

From the repository root, run
`python3 -m benchmarks.volatilitybench.run_volatilitybench`. Running it as a
module keeps the repository package importable in a clean shell.
