# Candidate 100: common current evidence and compact historical overlay

Status: offline only. Running experiment 097, main and PyPI are unchanged.

The original 30-session replay rendered 23 full contrast panels. The new view
re-renders exactly those panels with no task/outcome filtering, fits, Engy calls,
new historical queries or final-target access. It does not yet change an agent's
tools or prompt.

`current_cv_pair_100.current_cv_pair(current)` now owns the shared calculation.
It has no historical-ledger dependency. Both arms can receive the same matched
fold count, mean RMSLE, calculated ranks, per-model fold wins and ties. The
existing `current_history_contrast` calls that same function, then adds the
authenticated historical cohorts. All 23 prior detailed outputs are equal after
this extraction; the earlier artifact and its source hashes remain preserved at
commit b45a2946 and `results/current-history-contrast-100-offline-001/`.

`contrast_view_100.contrast_view` returns a compact view plus full evidence bytes.
Historical windows with the same underlying evidence share a labeled group;
they are not counted as additional samples. No-match, missing configuration,
missing page and unrequested history remain different states. Mean scores,
sample counts, latest matched origin and disagreement with current CV remain
visible without retrieval. The view does not choose a forecast.

The detailed artifact retains canonical configurations, versioned provider
identities, all six current execution IDs, fold origins/endpoints/scores and
the original historical evidence reference. Its path contains its SHA-256;
the view returns both SHA-256 and byte length. **The caller must persist the
returned bytes and make that path readable before exposing the view.** The pure
renderer does neither. Current evidence must originate from trusted execution
records, with availability authenticated by the caller.

## Measured presentation cost

| Quantity, across all 23 frozen panels | Compact UTF-8 bytes |
|---|---:|
| Previous full diagnostic panels | 44,130 |
| Common current-CV views | 16,922 |
| Compact views with historical overlay | 27,458 |
| Additional historical overlay | 10,536 |

The new ledger view is 37.78% smaller than the prior detailed diagnostic view.
Mean ledger view size is 1,193.83 bytes (maximum 1,498); the historical overlay
adds 458.09 bytes on average beyond the common view. These figures measure
compact JSON bytes, not a tokenizer, escaping inside a tool envelope, repeated
context on subsequent requests, or actual API cost. Optional artifact retrieval
would add cost and must be counted in a prospective experiment.

Twenty-four tests pass: the original eight contrast and ten historical
consistency tests, plus six tests for common-arm equality, standalone current
validation, artifact identity and immutability, deterministic ties, unknown
history distinctions and invalid historical inputs. The frozen replay checks
every prior detailed output for equality and every common current-CV view for
equality between arms. Artifacts are written and hash checked in the offline
replay, but actual Hermes retrieval has not yet been integrated or tested.

## Remaining before a prospective candidate

1. Connect the shared summary to trusted, already completed backtests in every
   arm, including the baseline folds computed by `start`; do not grant only the
   ledger arm additional current-CV information.
2. Isolate historical contrast as the ledger treatment. Retain equal raw history,
   memory availability, model choices, fit/agent budgets and failures.
3. Bound response growth when more than two configurations have been tried.
   The present function describes an exact pair; it is not an all-model selector.
4. Store and serve full artifacts through the real allowed evidence interface,
   with a common cost accounting rule. Test actual Hermes behavior without
   paid forecasting before considering a prospective run.
5. Keep 097 frozen through completion and audit. Freeze the next candidate's
   rules before new development calls and preserve its full costs and results.

The no-ledger arm also chose Ridge and made the contradictory fold-2 claim in
round 11 (audit 005). This design therefore treats current-CV clarity as common
infrastructure, not as a demonstrated unique ledger weakness. No numerical
improvement, causal effect or final 20% target is established by this work.

Reproduction: `results/current-history-contrast-100-compact-001/replay.py` reads
the preserved original panels and returned review records in their actual order.
Copy it to a fresh results directory and run with `PYTHONPATH=.` from the repo
root. Outputs are exclusive; original results are not overwritten. The committed
receipt inventories scripts, outputs, tests and every retained artifact.
