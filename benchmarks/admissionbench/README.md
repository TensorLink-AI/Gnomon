# AdmissionBench

AdmissionBench measures the model-admission policy independently of
TemporalBench and independently of any named TSFM. It sweeps true transfer
quality, external sample size, local fold count, and noisy future outcomes.
The policy sees only pre-outcome evidence; held-out loss is generated after
the admission decision.

It reports regret, fixed-arm comparisons, candidate weight by true usefulness,
and a reliability curve. It is a policy/property benchmark, not evidence that
any particular TSFM is strong; that evidence must come from held-out real
series and enter a versioned registry.

```bash
python -m benchmarks.admissionbench.run_admissionbench \
  --cases 5000 --output results/admissionbench/summary.json
```
