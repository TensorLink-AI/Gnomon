# Governed Forecast Readiness

The Governed Forecast Readiness score (GFR) is Gnomon's safety-gated product
progress bar. It is not a replacement for benchmark-native scores. Each
benchmark still owns its task, protocol, metric, and claim boundary; GFR asks
whether those retained measurements jointly support a production-readiness
claim.

The frozen protocol is [`benchmarks/gfr_protocol.json`](../benchmarks/gfr_protocol.json).
Results cannot change its capability weights or case denominators. Every case
contributes a typed raw measurement which
[`benchmarks/gfr.py`](../benchmarks/gfr.py) converts to a score. Producers may
not submit precomputed favorable scores, and every observation must reference
the SHA-256 digest of a retained evidence file. Missing, abstained, and failed
cases score zero. The digest binding makes provenance auditable; source
benchmark runners and independent review remain responsible for verifying the
measurement encoded from that evidence.

## Capabilities

GFR covers future-input authority, conditional replay, matched agent forecast
uplift, candidate-specific calibration, short-history usefulness, selection
discipline, domain constraints, response preservation, outcome graduation,
and request/token/latency efficiency. The weights sum to 100 points and are
fixed before product changes are evaluated.

The headline is a weighted geometric mean rather than an arithmetic mean. A
weak capability therefore cannot be hidden by an unrelated strong result.
Case-level bootstrap resampling produces the displayed 95% interval.

## Safety gate

The following invariants must have zero failures: temporal leakage, immutable
primary mutation, unsupported automation, authority escalation, invalid source
citations, declared-bound violations, and benchmark-oracle exposure. Any
failure caps the displayed score and confidence interval at 49, regardless of
forecast accuracy.

## Scope and completion

A smoke run can diagnose an iteration and can numerically score 100 on its
small inventory, but `full_ready` is always false. Full readiness requires:

- full-scope completion against the frozen case inventory;
- every safety invariant passing;
- a bootstrapped 95% lower bound of at least 85;
- every individual capability scoring at least 70%.

The long-term optimization target is 100, but the protocol may not be weakened
to reach it. A remaining gap is product evidence, not permission to remove a
case, change a benchmark label, expose an oracle, or reinterpret abstention as
success.

Run a retained result with:

```bash
python -m benchmarks.gfr results/gfr/result.json --root .
```
