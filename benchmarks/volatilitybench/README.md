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

The product contract is deliberately broader than the benchmark: the fitted
executable publishes a continuous future/reference residual-scale
distribution, probabilities over decreased/stable/increased, and a separate
decision policy. Weak best estimates remain usable for exploration, but only
out-of-fold probability skill and calibration can make a categorical answer
automation-eligible.

Run `python3 benchmarks/volatilitybench/run_volatilitybench.py`.
