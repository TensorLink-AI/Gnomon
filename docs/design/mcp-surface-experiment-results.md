# MCP surface experiment: Phase 3 decision

Date: 2026-08-12  
Code revision measured: `64ae6a7`  
Model: `deepseek-v4-flash-0731`

## Decision

Keep `full` as the default MCP profile. The narrower profiles remain explicit
experimental choices. None earned a default flip under the pre-committed rule.

This is a negative result, not an unfinished product decision. `core` reduced
the schema tax and cumulative tokens, but it still missed the token and call
targets and reduced answer yield on the matched forecasting rows. The
mega-tool and evidence-pack probes did not justify replacing the existing
surface.

## Matched control result

The same first T1, T2, and T4 rows, model, endpoint, prompt, tool-call cap, and
code revision were run against `full` and `core`. T3 was attempted separately
but the provider credential began returning HTTP 401 before either arm made an
LLM request, so it is excluded symmetrically rather than counted as a product
failure.

| Metric | `full` | `core` | Gate |
|---|---:|---:|---:|
| Matched rows | 3 | 3 | — |
| Schema bytes, representative | 42,102 | 20,351 | <= 12,000 target for a narrow default |
| Mean cumulative tokens / task | 139,785 | 109,393 | <= 50,000 |
| Median calls | 4 | 4 | <= 2 |
| p95 calls | 4 | 4 | <= 4 |
| Rows returning an answer | 3/3 | 2/3 | >= 80% |
| Forecast rows completed | 2/2 | 1/2 | no regression from control |
| Leaktrap | 0/40 | profile-independent engine invariant | 0/40 |

The T2 forecast is the binding quality regression: `full` completed it with a
Gnomon artifact, while `core` reached the round cap without submitting. On T4,
both arms completed; `core` published the Gnomon forecast (OW_sMAPE 9.2409),
while `full` submitted an informed direct forecast (OW_sMAPE 8.7547). Three
rows are not an accuracy claim, but accuracy significance is immaterial to the
decision because `core` already fails the economics and yield gates.

## Prototype probes

These probes establish feasibility and failure modes; they are not presented
as matched leaderboard results:

- `describe` exposed useful deterministic evidence but an earlier unconstrained
  run consumed 620,674 tokens and 15 calls.
- `evidence` (`describe` + `forecast`) first abstained without calling a tool;
  after prompt repair it consumed 165,457 tokens and four calls, with worse
  forecast error than the matched Gnomon artifact.
- `mega` produced a T1 descriptive answer at 29,222 tokens and five calls, but
  forecasting probes repeatedly exceeded the practical run window. Moving the
  discriminated union inside one tool did not make the workflow one-shot.

## What Phase 3 changed

The experiment itself is now reproducible rather than a prose proposal:

- executable `full`, `core`, `describe`, `mega`, and `evidence` profiles;
- profile, schema bytes, cumulative tokens, and call median/p95 in summaries;
- `--offset` one-row shards for resumable matched runs;
- a four-tool browsing cap, ten-round cap, final submit-only round, and recovery
  of a complete verified artifact when only submission formatting fails;
- CI tests for profile membership and schema budgets.

No production default changed. A future flip requires a new candidate to pass
the same gates; lower tool count by itself is not sufficient.
