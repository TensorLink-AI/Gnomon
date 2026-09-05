# Historical material

This directory contains retired material retained for reproducibility. It is
excluded from installed distributions and the default regression suite.
It makes no current product-quality or readiness claim.

- `benchmarks/reasoningbench`: withdrawn answer-exposing reasoning instrument,
  moved from the active benchmark tree on 2026-09-05. Its historical tests can
  be run explicitly with `pytest archive/benchmarks/reasoningbench/test_historical.py`.

Original source is also recoverable from Git revision `0d3d72e`.

`legacy/router_pre_ledger.py` preserves the pre-cull router as source, not an
active module. Its mutable leaderboard prior had no source/recorded cutoff and
could mix unmatched tasks and overwritten scores. To inspect the original code,
read this file; restoring it to `src/gnomon/router.py` would restore those known
defects and is not a supported runtime configuration.

`legacy/adapter_promotion_pre_ledger.py` preserves the shadow-outcome router before
its routing authority was removed. Its rows use INSERT OR REPLACE and lack local
recording times. Historical diagnostics remain available, but active compatibility
calls retain the explicit champion and cannot nominate a challenger from them.

`legacy/toolspec_pre_session.py` preserves the pre-default-switch registry,
including the now-retired describe/mega profiles and run/track registrations.
Unregistered diagnostic helpers live in `gnomon.legacy_experiments`; no active
profile exposes them. Numeric regression coverage remains in the normal suite.
`legacy/quickstart_mcp_pre_session.md` preserves the former advanced-workflow
quickstart; current onboarding describes the actual default session.
