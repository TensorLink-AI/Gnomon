# Availability-aware warm-up preparation 042

Strict preparation 041 failed: sensor 1 lacks 120 preperiod hour labels, while
the other seven pedestrian sensors cover the requested interval. No count
values or warm-up forecast errors were inspected. Preserve 041 as failed.

Keep its eight proposed earlier origins, fixed sixteen development identities,
dates, units, source hashes and all 416 scored tasks. This separate amendment
models naturally incomplete historical evidence: an earlier origin is usable
only if its entire 730-hour history and 24-hour target exist. Exclude the entire
six-provider cohort otherwise, with exact missing-history and missing-target
counts. Do not interpolate, shorten history, select replacement sensors, move
origins or drop a scored task. This is not completion of the strict 041 design.

Require unique labels and unchanged sensor identity throughout the preperiod.
Allow absent observations as explicit null markers in preparation output only;
they must never be passed to a provider as values. Check metadata first, then
parse only selected development preperiod counts. Check the final 730-hour
overlap with the original scored history exactly. Freeze all origin exclusions
before any model computations. Zero-valued observations remain valid.

Every arm receives the same arrived observations and the same declared usable
warm-up cohorts. Record both attempted/unavailable cohorts and actual numerical
costs. Partial historical support must remain visible in later results. No
scored origin is omitted, and no reserved future counts are accessed.

This authorizes preparation only. Subsequent fitting/selection rules still
require a separate freeze. Source/recording times are the same disclosed
period-end replay assumptions, not genuine historical execution timestamps.
