# PropertyBench

PropertyBench evaluates the fitted temporal-property executables on held-out
synthetic futures. It sweeps behaviours and independent seeds for level,
trend, seasonality, regime shifts, extremes, and paired dependence. It reports
numeric error, interval coverage, calibrated claim rate, supported-claim
accuracy, and best-estimate accuracy across both supported and weak answers.
Separate lanes distinguish deterministic point-path alignment from a fitted
future-process seasonality claim; the latter covers fixed phase, repeatable
phase drift, and unpredictable phase changes that should abstain.
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
