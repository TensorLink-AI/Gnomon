# v0.6 loop I007: hierarchical reconciliation

Decision: **no-build.** Gnomon has no hierarchy surface today, so it does not
mislabel independent panel forecasts as coherent totals. The frozen candidate
would have added only a derived bottom-up q50 total, preserving leaves and
withholding aggregate uncertainty and automation authority.

The preimplementation screen completed 32/32 non-overlapping real-derived
cases. Arithmetic coherence, finiteness, positivity, and uncertainty truth all
passed. Forecast skill did not: summed leaf forecasts were 5.45% worse overall
than forecasting the root directly, and periodic-share retail, sensor, and
traffic strata exceeded the 2% non-inferiority margin. The paired bootstrap
95% interval for root-MAE improvement was -1346.94 to 164.78 error units.

No hierarchy API or production arithmetic was added, and no reconciler was
selected after inspecting outcomes. The serial resumable screen remains under
`benchmarks/hierarchybench`; raw rows remain under
`results/v06-p7-hierarchy-baseline`. The final local suite passed 2,564 tests
with 11 skips. `docs/astrid-btc-agent-plan.md` remains untracked and excluded.
