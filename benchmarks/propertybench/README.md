# PropertyBench

PropertyBench evaluates the fitted temporal-property executables on held-out
synthetic futures. It sweeps behaviours and independent seeds for level,
trend, seasonality, regime shifts, extremes, and paired dependence. It reports
numeric error, interval coverage, calibrated claim rate, supported-claim
accuracy, and best-estimate accuracy across both supported and weak answers.
Separate lanes distinguish deterministic point-path alignment from a fitted
future-process claim. The seasonality lane covers fixed phase, repeatable
phase drift, and unpredictable phase changes that should abstain. The
volatility lane scores genuinely unseen process dispersion across stationary,
gradual-change, recent-regime-shift, trending-level, and heavy-tailed families;
it is intentionally separate from point-forecast smoothness. Additional
stress lanes distinguish paired transient excursions from permanent regime
changes and test dependence under common trends, lag-only association,
sign changes, and heteroskedastic noise.
Graduation therefore checks that weak estimates remain useful without treating
them as automation-grade claims. The generators never expose benchmark labels
to the product code.

Run `python -m benchmarks.propertybench.run_propertybench --output-dir results/propertybench`.

Use multiple independently chosen seeds for a graduation claim. The aggregate
report pools held-out row classifications, retains every constituent score,
and refuses duplicate seeds or mixed code revisions:

```bash
python -m benchmarks.propertybench.aggregate \
  results/propertybench-seed-9100 \
  results/propertybench-seed-94721 \
  results/propertybench-seed-27183 \
  --output results/propertybench-multiseed.json
```

The runner writes an exact-code manifest beside each `summary.json`. Do not use
an output made before manifests existed as release evidence.
