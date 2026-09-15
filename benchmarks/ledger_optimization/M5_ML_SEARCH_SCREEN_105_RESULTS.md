# M5 fixed ML search screen 105 — negative development result

The prospectively frozen eleven-configuration screen completed all 208 cases
(eight development series, 26 origins). None of its three historical selection
rules improved on current three-fold CV. No rule is promoted. The 20% objective
and untouched-final gate remain unmet.

| Fixed rule | Mean per-case RMSLE | Improvement over current CV |
|---|---:|---:|
| Current CV | 0.594650910 | reference |
| Previous four matured origins | 0.596998476 | −0.395% |
| All prior matured origins | 0.595414305 | −0.128% |
| Lifetime leader with paired-win support | 0.596341888 | −0.284% |
| Hindsight minimum, unavailable to an agent | 0.501504827 | 15.664% |

All 32 cold-start cases remain included. Over the subsequent 176 cases, current
CV scored 0.615971215 versus 0.618745611, 0.616873409 and 0.617969643 for the
three historical rules. The result is not explained solely by cold starts.

Even perfect selection from this particular grid would not reach 20% against
this fixed CV comparator. This does not bound untried configurations or the
actual agent comparator, establish that ledger infrastructure is generally
useless, or authorize weakening the control. The live candidate-100 experiment
and frozen M5 agent protocol are unchanged. No held-out numerical data was read.

## Execution and verification

Protocol commit `a79734bb` preceded the real numerical run; implementation and
preflight were committed as `95e990be`. The run used the frozen common numerical
implementation and authenticated plain runtime, without invoking Gnomon's engine
or Engy. It made 9,152 successful numerical calls, including 6,656 estimator
fits, with zero numerical failures. Run elapsed time was 173.930 seconds.
Separate synthetic preflight cost 220 calls and 160 fits. API cost is zero;
local compute billing is unknown, not asserted free.

The independent auditor passed 87,314 checks. It reconstructs requests,
visibility, past-only selections, scores, aggregates, seasonal predictions,
configuration identity and attempt accounting. It verifies recorded ML outputs
against frozen source/runtime identity but does not independently refit ML.
Five altered copies (future timestamps, actuals, self-consistent wrong selection,
fabricated CV scores and a missing attempt) were all rejected. Original evidence
remained unchanged. Both auditor probe versions and their outputs are retained.

The raw evidence archive contains 2,152 inventoried files plus its inventory,
6,301,128 compressed bytes. Every member was streamed and checked against the
inventory after creation; originals were checked again. Archive SHA-256:
`b4caf0a47ed9a0d21f8a0bc03918387e16808b6bb7cbee61609a2c9a395f76b3`.

## Retained evidence

- Frozen method: [M5_ML_SEARCH_SCREEN_105.md](M5_ML_SEARCH_SCREEN_105.md).
- Compact aggregates, source hashes, costs and audit receipt:
  [evidence/m5-ml-search-screen-105-result-001.json](evidence/m5-ml-search-screen-105-result-001.json).
- Original results: `results/m5-ml-search-screen-105-development-001/`.
- Launch argv/stdout/stderr/exit: `results/m5-ml-search-screen-105-launch-001/`.
- Independent audit: `results/m5-ml-search-screen-105-audit-001/`.
- Altered-copy probes: `results/m5-ml-search-screen-105-auditor-001/` and `-002/`.
- Archive: `results/m5-ml-search-screen-105-archive-001/evidence.tar.gz`.

These ignored raw paths are local retained artifacts, not a claim that their
large contents were uploaded to GitHub. The development branch contains the
protocol, implementation, auditor and compact authenticated results. Main and
PyPI are unchanged.
