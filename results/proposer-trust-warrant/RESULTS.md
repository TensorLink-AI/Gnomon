# Result: precondition not met — the experiment is not run

**FINAL (2026-08-04, completed grid).** The full 355-run census
(`results/structural-effects/census-full.json`, run at `0c9a63e`)
confirms the deferral under the amended rule: 15 `non_numeric_claim`
spans, 14 excluded as absence-of-intervention statements by the
registered embodiment test, **1 genuine warrant-requiring instance
against the ≥ 40 threshold**. Per `AMENDMENT-2026-08-04.md`, this
outcome is final for this data; any future case must come from new
evidence, not reinterpretation. The one genuine instance (a stated
structural cessation) is pursued under its own registration,
`results/structural-effects/`.

---

Census run 2026-08-04 against `HYPOTHESIS.md` (registered at `e442593`).
Outcome recorded per the registration's own rule: *"Precondition census
< 25% and < 40 instances → the experiment is **not run**; recording that
outcome in RESULTS.md closes this registration honestly."*

## The census

`benchmarks/cik/classify_rejections.py` over a flag-on CiK proposing run,
`deepseek/deepseek-v4-flash-0731`. Partial grid: **200 of 355 runs — the
first 37 of 71 tasks in order**, so task families later in the ordering
are unmeasured.

Of 150 classified span-parse rejections:

| bucket | count | share |
| --- | --- | --- |
| `numeric_no_parse` | 139 | 92.7% |
| `non_numeric_claim` | **11** | **7.3%** |

Both pre-registered thresholds are missed by a wide margin: 7.3% < 25%,
and 11 < 40.

## What the numeric_no_parse examples are

The dominant failure is not directional claims needing trust — it is
spans that **do state a number, in a shape the parser cannot read**:

- "4 times the number of usual withdrawals" — a multiple of a baseline
- "3 times the usual electricity" — same
- "The maximal fan speed is 3000 rpm" — an attributive bound
- "the maximal pressure is 37.5 Pa" — same
- "it rapidly and smoothly changes to 1593.0" — a narrated level change
- "(2022-03-23 00:00:00, 0)" — a quoted (timestamp, value) pair

Nobody needs to be trusted for these; the number is in the text. The
roadmap is therefore a **grammar fix** (multiples-of-baseline,
attributive bounds, narrated level changes, quoted point tuples), not a
new warrant.

The 11 `non_numeric_claim` spans undercut the hypothesis further: every
one is an "assume that X will **not** happen" statement (no maintenance,
no glitch) — the absence of an intervention, which a history-only
forecast already assumes and no warrant of any kind is needed for. The
count of genuine directional claims ("demand will be higher") on the
measured grid is zero.

## Status

**Deferred, not falsified.** The hypothesis about directional claims was
never tested — its own precondition ruled the experiment out. The bet
that motivated it ("proposals fail because claims are unverifiable")
was wrong on the measured grid: proposals fail because the parser is
narrow.

Reopening condition, per the partial-grid caveat: finishing the task
grid (tasks 38–71) could move the count. The reopening test is the
amended, denominator-free rule in `AMENDMENT-2026-08-04.md` — the
original share-of-rejections test was voided after the grammar fixes
collapsed its denominator (19 span-parse rejections where the
threshold was calibrated against ~150), which made it measure residue
composition instead of warrant demand. The amendment was registered
while the extended-grid run was in flight, before its data was read.

## Recovery measurement (2026-08-04, same dumps, parser the only variable)

Re-classifying the same census dumps after the first grammar change
(attributive bounds, multiples-of-baseline, narrated changes, point
tuples) — 220 dumps, 153 span-parse rejections:

| bucket | before | after |
| --- | --- | --- |
| `parses_now` | 0 | 68 |
| `numeric_no_parse` | 142 | 74 |
| `non_numeric_claim` | 11 | 11 |

47.9% of the numeric residue recovered. Of the 74 remaining, manual
review found 63 further grammar gaps (44 bare value+window "0.2 from
05:34:29 until 05:34:46", 7 covariate-narration "takes a value of
0.2051", 5 percent-of-usual, 5 zero-state phrasings, 2 conditional
maxima) and only 11 genuinely unparseable date-only events ("the
weather will become clear"). Those 63 shapes were then covered in a
second grammar round, each pinned verbatim in
`tests/test_preflight.py`, and the re-measurement confirmed the floor —
same 153 rejections across all three classifications:

| bucket | pre-fix | round 1 (`2c3e7f7`) | round 2 (`cae6a3c`) |
| --- | --- | --- | --- |
| `parses_now` | 0 | 68 | **131** |
| `numeric_no_parse` | 142 | 74 | 11 |
| `non_numeric_claim` | 11 | 11 | 11 |

92.3% of the numeric residue recovered. The remaining 11
`numeric_no_parse` are all date-only events whose digits are
timestamps ("2024-05-27 is a holiday", "the sensor was repaired and
this additive trend will disappear") — no value is stated, so no
numeric grammar admits them; they belong to the fold-ablation gate or
a structural-effect class this lane deliberately does not have. With
the zero-state spans recovered, the corrected `_ZERO_STATES` claim now
holds on the measured data. `non_numeric_claim` is unchanged at 11
(8× "assume no ATM maintenance", 3× "assume no sensor glitch"), so the
deferral verdict stands at 7.3%.

## Post-census correction

`HYPOTHESIS.md`'s background asserts stated closures already parse as
an override of 0. The census showed the claim was broader than the
code: `_ZERO_STATES` matched a curated noun list ("no production/
output/traffic/flow/generation") that did not include "no withdrawals",
so five zero-state spans sat in the residue. The noun list is extended
(still curated — a bare "no \\w+" would misread "no change" as a value
of 0). The correction does not affect the deferral: these spans need
grammar, not a warrant.
