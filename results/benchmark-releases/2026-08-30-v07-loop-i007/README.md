# v0.7 loop I007: decision-coherent seasonal trend answers

Decision: **promote the coherent trend executable and typed agent boundary;
retain weak/abstained authority where the frozen evidence cannot support
automation; continue to Q6.**

The frozen 152-case development matrix reproduced four general defects:
canonical direction disagreed with the numeric estimate on 33 rows, every
weak interval was zero-width, all 24 insufficient-cycle rows returned a
numeric answer instead of abstaining, and the trend executable could measure a
partial seasonal arc rather than phase-adjusted drift. The exact candidate
uses phase-fixed trend estimation for an admitted season, aligns the canonical
direction to its point estimate, publishes an observed-error envelope for
weak calibrated rows, and returns a typed null abstention when two admitted
cycles are not visible. It does not lower the support threshold or silently
admit a detected period.

On the exact development matrix, identifiable direction accuracy rises from
**64.06% to 85.16%**, held-out interval coverage from **0% to 79.49%**, and
mean selective utility from **0.0197 to 0.1661**. Direction/estimate
inconsistencies fall from 33 to zero, and all insufficient-cycle controls
abstain. There are deliberately no supported claims in this two-origin lane,
so its absolute supported-accuracy gates remain false and are not relabelled
as passing.

Fresh seeds `8121`-`8128` confirm the result: accuracy rises from **57.81% to
85.16%**, coverage from **0% to 82.46%**, and utility from **-0.0066 to
0.1711**. Two detached PropertyBench confirmations provide the supported
denominator. Pooled over 160 trend cases, best-estimate accuracy rises from
**80.0% to 83.125%**; all 40 baseline and all 32 candidate supported claims
are directionally correct; supported interval coverage changes from **97.5%
to 96.875%**; and mean absolute error falls from about **0.04305 to 0.02033**.
Every non-trend property summary is unchanged between matched arms. The lower
candidate claim rate is retained as conservative calibration, not counted as
an accuracy gain.

The six-case real-agent boundary baseline exposed a separate integration
defect: predictive trend fell through to observed `up`/`down` report labels,
and the host did not preserve the complete typed choice contract. The final
candidate routes predictive trend through the fitted executable, inherits the
immutable primary's admitted seasonal period, and projects canonical value,
task-facing display value, support, automation eligibility, primary
immutability, and authority together. All **6/6** engine receipts, final
choices, host contracts, primaries, and Gnomon artifact routes pass. The
short-history row now returns `uncertain`, null estimate/interval,
`support=abstained`, and final `Uncertain`.

The frozen two-case ContextBench control also passes. Both engine artifacts
expose `relationship_to_primary=no_distinct_numeric_path`, preserve the
canonical primary, and set `automation_eligible=false`. Both final responses
preserve that meaning, cite the failed gate and validated source, and retain
the authority limit. This distinguishes **2/2 complete engine contracts**
from **2/2 agent-preserved explanations** rather than treating one as proof of
the other.

The exact final isolated suite passed **2,629 tests with 11 skipped**. PR #82's
required CI checks were green when Q2 closed; paid TemporalBench and the
scheduled deterministic suite were correctly skipped. Raw resumable
checkpoints and actual responses remain preserved locally. The
external-evaluation intake remains controlling, and
`docs/astrid-btc-agent-plan.md` remains untracked and excluded.
