# Pre-registered hypothesis: seasonal-regime structural effects

Registered 2026-08-07, at commit `fe86df5`
(claude/gnomon-harness-issues-hgerrj), **before** any implementation or
run of the treatment. `RESULTS.md` will be written against these
predictions.

## Background

The paired spot-checks of 2026-08 found the largest single family
where mandatory Gnomon loses to the plain-LLM control: tasks whose
context states a **dated qualitative state** for the future window —
the measured exemplar is solar irradiance with

> "At 2022-07-15 06:00:00, the weather will become clear."

(~20× worse than control on the checked task-seed). The statement is
decisive, dated, and numberless. It has no home in any existing
warrant: the numeric grammar has nothing to parse, the fold-ablation
gate cannot test a future-only window, and the v1 structural menu
(`trend_ceases`) has no shape for it. The span-recovery census's
honest residue included 11 date-only events of this species. The MCP
arm lets an agent route such tasks away from Gnomon; this registration
is the capability answer rather than the routing answer.

## The division-of-labour principle (unchanged)

**The LLM reads and classifies; it never supplies a number that is
applied.** A stated sky/operating state names *which part of the
series' own history the future will resemble*. Naming that part is
classification; computing what it implies is arithmetic on observed
data. The LLM does the first; the engine does the second.

## Intervention

Two new entries in the closed structural-effect menu (same lane, same
`context.structural_events` flag, same admission checks as
`trend_ceases`; flag-off artifacts remain byte-identical):

`level_matches_seasonal_high`
    The context states the series enters its high regime for a dated
    window (clear skies for irradiance, full production, peak
    operation). At **admission**, the engine computes the per-phase
    seasonal envelope from the observed history: phase = index mod
    season; envelope value per phase = the phase's q90 of observed
    values. At **application**, each covered step's point moves to its
    phase's envelope value and every quantile translates by the same
    delta — a pure location shift per step; interval widths, quantile
    ordering, and the point-to-median gap untouched. Every applied
    number is a quantile of the engine's own observed data.

`level_matches_seasonal_low`
    Symmetric, with the per-phase q10 (overcast/rain suppressing
    output, curtailed operation, minimal demand).

Additional admission guard, `seasonal_profile_resolvable`: requires a
detected season ≥ 2 and at least two full observed cycles; otherwise
the event is rejected with that code (no fallback to a global
quantile — a profile the history cannot support is not resolved from
somewhere else).

Steps already adjusted by one structural event are skipped by later
ones (shared `touched` set with `trend_ceases`).

## Predictions and decision rules (locked now)

All benchmark terms are for the CiK weather/solar task family (tasks
whose context states a deterministic sky/weather state for a dated
window), run by the benchmark session at one pinned revision, both
arms (`--structural-context` on vs off), same seeds.

- **H1 — reachability.** The proposer, given the updated menu
  instructions, produces ≥ 3 *admitted* regime events across the
  family at 1 seed. Fewer means the lane is unreachable in practice:
  record the count and stop — no score claims.
- **H2 — paired improvement.** Over task-seeds with ≥ 1 admitted
  regime event: mean capped-imputed RCRPS (flag-on) improves on
  flag-off by ≥ 10% relative, and flag-on is worse on no more than
  one third of the pairs. Miss either term → the menu entries are
  removed or reworked; the result is recorded either way.
- **H3 — no false fire.** On the cessation-bearing sensor family and
  the numeric-bounds families, regime admissions are expected to be 0;
  more than 2 across those families is a classification leak →
  tighten the instructions or kill the entries.
- **Engine-level restatement** (unit suite, not benchmark): flag-off
  runs emit identical points and no gate record; covered steps land
  exactly on the per-phase envelope; uncovered steps and interval
  widths are untouched.

## Disclosed limitations and risks

- The q90/q10 envelope is a bounded reading of "high/low regime": on a
  history that was mostly cloudy, per-phase q90 under-reaches the true
  clear-sky curve. The prediction is only that the envelope beats the
  history-median path the base forecast follows, not that it equals
  the physical regime.
- A regime event asserts a level *jump* at its window edge — that
  discontinuity is the claimed semantics, unlike `trend_ceases`'
  continuity. The application record carries per-step deltas so the
  jump is quotable.
- Support for influenced runs remains `context_trusted` — text was
  trusted, not fold-proven.

## What may NOT be claimed from this experiment

Wins on this family are evidence about the lane's coverage, never
about proposer trust (the classification is menu-bounded by
construction) and never a general Gnomon-beats-control claim. No score
is quoted without abstention counts beside it; comparisons only
within one code revision.
