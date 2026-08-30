# ClaimBench

ClaimBench is the small deterministic forecast claim/support surface check
frozen in `docs/v0.7-q1-claim-support-coherence-protocol.md`. It scores no
future forecast values. It checks whether already-measured model, gain,
support, calibration, and threshold evidence is rendered consistently for
humans and agents.

Run with machine-local TSFM supply isolated:

```bash
GNOMON_TSFM_SANDBOX_ROOT=/tmp/gnomon-empty-tsfm-root \
  python -m benchmarks.claimbench.run --output-dir results/claimbench-run
```

The runner appends one JSONL checkpoint per completed case and resumes without
rerunning those cases. `summary.json` contains the aggregate gates.

