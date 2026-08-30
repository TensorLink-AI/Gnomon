# v0.7 loop I004: safer forecast departures

Decision: **promote the general safety fixes; retain Q1 as unconfirmed and
continue with claim/support coherence.**

The fresh 48-case naturalistic baseline showed that the selector's existing
fold checks did not generalize reliably: 36 departures produced 15 wins, 15
losses, and 6 ties against the strongest of last-value, seasonal-naive, and
historical-mean references. Four departures were specifically invalidated by
historical mean, the clustered 90% median-gain interval began at -14.1%, and
the environmental and operational group medians were below the frozen -2%
floor.

Two general, prefix-only production changes were evaluated without routing on
dataset identity or future labels:

- historical mean is now a mandatory baseline rather than only a hindsight
  benchmark reference;
- ordinary forecasts with at least five disjoint origins reserve independent
  selection, confirmation, interval-calibration, and final-test roles. The
  already-selected contender must remain non-inferior to the already-selected
  strongest baseline on confirmation. This is a veto, not a second model
  tournament.

On exact final commit `de6f6d7`, all 48 naturalistic product cases completed
with zero future leakage and complete provenance. Harmful departures fell
from 15 to 8, while 13 wins remained; wins outnumbered losses, group medians
all remained within -2%, and the clustered lower bound rose to zero. The
actual forecast response reports the contender, baseline, origin, both
scores, confirmation outcome, and fallback. Cross-series VAR evidence also
distinguishes winning selection from failing confirmation.

The stricter original naturalistic gate is not passed or relabelled. Positive
gain precision is 43.3%, median departure gain is zero, and two departures are
still invalidated by historical mean. Q1 therefore remains open.

The unchanged 200-case synthetic screen remains strongly green on the exact
final head: 200/200 complete, 85 departures, 74 wins, 11 losses, 87.1%
precision, +59.0% median departure gain, a 90% interval of +39.6% to +73.8%,
and all ten frozen gates passed. This shows that the safety veto did not erase
known useful mechanisms; it is not substituted for naturalistic evidence.

Validation on the exact final head: **2,606 tests passed, 11 skipped** with an
empty TSFM sandbox root, plus **176 focused boundary tests** before the final
audit fix and **102 multivariate/ensemble/runtime boundary tests** after it.
The machine-local TSFM sandbox and all pre-existing artifacts were preserved.

Raw baseline, intermediate, exploratory, and exact candidate artifacts remain
under their original result directories. Only aggregate exact-commit arms are
included here. `docs/astrid-btc-agent-plan.md` remains untracked and excluded.

