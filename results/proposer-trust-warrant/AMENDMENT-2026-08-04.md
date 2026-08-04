# Amendment: the share test's denominator was destroyed by our own fix

Registered 2026-08-04, while the extended-grid run (tasks 38–71) is in
flight and **before any gate record beyond the original 37-task prefix
has been read**. This amendment rules on a conflict the original
registration failed to anticipate, and pre-commits the decision rule
for the completed grid before the data it will judge exists.

## What happened

`HYPOTHESIS.md` gates the experiment on:

> non_numeric_claim accounts for ≥ 25% of span-parse rejections, or
> ≥ 40 instances

The share test was calibrated when span-parse rejections numbered ~150
and the parser was narrow. The grammar fixes made in response to the
census (`ae9e28b`, `2c3e7f7`, `cae6a3c`) collapsed that denominator:
on the current partial rerun there are 19 span-parse rejections, of
which 15 are `non_numeric_claim` — 78.9%, clearing the 25% share on a
literal reading, while the absolute count (11 → 15) sits nowhere near
40.

## Ruling

**The share test is void as measured; the count test governs.**

The registration's own rationale sentence — *"Below that, the observed
pain is parser narrowness or malformed proposals — the roadmap is a
grammar or prompt fix"* — states what the share was a proxy for: the
magnitude of unmet warrant demand relative to the observed failure
stream. After the grammar fix, the same expression measures the
*composition of the residue*, which trends to 100% as the parser
improves. A precondition that becomes easier to meet every time the
rival roadmap succeeds no longer measures what it was designed to
measure, and following its literal text would green-light the
experiment on an artifact of our own repair.

The current partial rerun therefore does **not** meet the
precondition: 15 instances against the ≥ 40 count threshold. The
deferral stands.

## Pre-committed rule for the completed grid

For the reopening decision on the full 355-run grid (and any future
census), the precondition is the single denominator-free test:

> **≥ 40 genuine warrant-requiring instances** across the completed
> grid.

"Genuine warrant-requiring" excludes "assume X will **not** happen"
statements (no maintenance, no glitch): a history-only forecast
already embodies the absence of an intervention, so no warrant of any
kind is needed for them. This exclusion is not new to this amendment —
it was recorded in `RESULTS.md` from the first census, before the
extended-grid data existed. Every instance counted toward the 40 must
be quoted verbatim in the census record.

The amended rule is strictly no easier than the original pair: it
keeps the harder of the two original tests and narrows what counts
toward it.

## What this amendment may not do

It may not be revised again after the extended-grid data is read. If
the completed grid shows ≥ 40 genuine instances, the experiment
proceeds exactly as registered in `HYPOTHESIS.md`; below that, the
deferral is final for this data, and any future case must come from
new evidence, not reinterpretation.
