# Diagnostic069: remaining blend error and forecast-range limitations

Freeze before new comparisons from068. Read only the416 original development
cases and their066current CV pairs. No policy, forecast, fit, API call, protected
validation/final access, or release change. This is hindsight diagnostic geometry,
not a candidate selector or a learning rule. Existing gates remain failed.

For each current forecast horizon and arm, work in log1p space. Let lo/hi be the
minimum/maximum of the seven available forecasts, y the actual, z its projection
onto[lo,hi], and p the produced blend. Define range residual r=z-y and blending
residual b=p-z. Then squared forecast error is r^2+b^2+2rb. The cross term is
nonnegative because p lies within the available range. Preserve all vectors,
verify this identity, and report each term rather than calling all outside-range
error irreducible. Only r^2 is the irreducible error for the unconstrained
per-horizon convex range; four-block weights impose additional constraints.

Report per-case RMSLE for actual blends and projected-range lower bound;
arithmetic mean across416cases, both domains and early/later phases. Also show
point counts below/inside/above the range, signed residual means and the pooled
squared-error decomposition. Pooled energy shares are not shares of arithmetic
mean case RMSLE; label denominators explicitly. Record both control and ledger
without choosing favorable examples. Exact ties to the bounds count as inside.

For each arm's three current observed CV folds, compute the same seven-forecast
range projection and floor RMSLE. Compare average current CV floor with future
production floor across the cohort. This is descriptive information about
backtest/production geometry, not proof that an availability shift or business
regime caused a forecast error. No future-derived feature may enter a learner.

If range floor is near actual error, a different forecast capability or bounded
correction may be needed; if much error remains above the floor, this only
establishes geometrical room for better mixtures, not that the right weights
are predictable or that a20%causal ledger gain is attainable. Do not supply
hindsight projections or weights to subsequent agents or prospective policies.

Keep all source hashes, full per-case vectors, coverage counts and costs. The
mathematical floor permits independent per-horizon weights outside068's
four-block action, so it must not be presented as an attainable068 optimum.
Any follow-up action-space or objective change requires a separate freeze.
