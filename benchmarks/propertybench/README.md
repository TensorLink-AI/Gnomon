# PropertyBench

PropertyBench evaluates the fitted temporal-property executables on held-out
synthetic futures. It sweeps behaviours and independent seeds for level,
trend, seasonality, regime shifts, extremes, and paired dependence. It reports
numeric error, interval coverage, calibrated claim rate, supported-claim
accuracy, and best-estimate accuracy across both supported and weak answers.
Graduation therefore checks that weak estimates remain useful without treating
them as automation-grade claims. The generators never expose benchmark labels
to the product code.

Run `python -m benchmarks.propertybench.run_propertybench --output-dir results/propertybench`.
