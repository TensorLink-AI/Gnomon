# v0.7 loop I003: bounded timestamp jitter

Decision: **promote S2 and return to Q1 naturalistic confirmation.**

The independently regenerated external finding reproduced on the exact
current head. Of 22 frozen repair-mode boundary runs, the baseline accepted
only the three exact-grid controls. Safe repair refused ordinary one-second
20-minute scheduler jitter, and aggressive repair either refused it or
treated moved timestamps like invented values. Epoch-anchored aggressive
snapping also had authority to merge two observations into one slot before a
later ceiling check rejected the run.

The production boundary now learns a deterministic grid phase from the
series and applies one shared bounded alignment rule in safe and aggressive
repair. Tolerance is 1% of cadence, capped at 60 seconds (12 seconds on the
frozen 20-minute series). Alignment is all-or-nothing: an outside-boundary
point is left for typed strict diagnosis, and a collision or non-increasing
slot map returns `TIMESTAMP_ALIGNMENT_CONFLICT` without emitting a partial
series.

On exact commit `ec99285`, all 13 frozen gates passed:

- accepted boundary runs: 3/22 → 12/22, exactly the newly authorized safe
  and aggressive bounded-jitter paths plus the existing small-gap aggressive
  path;
- all 36 values and observations were preserved in accepted alignment cases;
- 24 one-second timestamp corrections (67% of the series) no longer tripped
  the 30% invented-value ceiling;
- repair evidence exposed cadence `20min`, phase, 12-second tolerance, count,
  and maximum displacement;
- exact grids remained untouched in off, safe, and aggressive modes;
- outside-boundary jitter, collisions, mixed cadence, and long outages all
  remained typed refusals;
- safe repair refused a real gap, while aggressive interpolation retained a
  separate `gap_filled` action and count;
- reordered input retained a distinct `timestamps_reordered` disclosure.

Actual direct-runtime, inspect, MCP forecast/inspect, and CLI forecast/inspect
responses were inspected. They reported `repaired_safe`, retained the typed
alignment metrics, and surfaced `timestamp_jitter_aligned` as a forecast
warning rather than describing it as an imputed value.

Validation on the exact candidate: **182 focused tests passed**. The complete
suite with an empty `GNOMON_TSFM_SANDBOX_ROOT` passed **2,599 tests with 11
skipped**. An initial unisolated run passed 2,584 and failed 14 tests because a
machine-local `toto2_4m` sandbox entered tests and golden artifacts that
intentionally assume no installed TSFM. All 14 failed tests passed under the
empty sandbox root; the installed sandbox was preserved, and no model
selection code or golden was changed to accommodate local supply.

Raw baseline, exact candidate, and exploratory candidate artifacts remain
preserved. Exploratory dirty-tree runs are deliberately excluded from the
aggregate comparison; only exact committed arms carry the promotion claim.

