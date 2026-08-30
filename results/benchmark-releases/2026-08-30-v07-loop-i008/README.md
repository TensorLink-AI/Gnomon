# v0.7 loop I008: regime-aware anomaly events

Decision: **promote regime attribution and causal spike-state cleaning; continue
to Q3.**

The frozen 24-case, three-surface reproduction confirmed both reported
defects. Investigation correctly admitted all eight permanent shifts but
repeated every post-shift observation as an anomaly (384 false events, 4%
event precision). Unlabelled detection preserved all 16 planted events but
flagged five normal rebounds and reached 27.12% precision.

The candidate keeps changepoint detection byte-for-byte outside the change.
Investigation scores point anomalies inside supported regimes and discloses
the 384 re-attributed points through a typed
`explained_by_regime_shift` record. Its precision reaches 84.21% at unchanged
100% recall; three retained points are anomalous within their new regimes.
The one-step detector now prevents one extreme innovation from contaminating
the next causal state while adapting after three same-direction extremes.
Unlabelled precision reaches 30.19%, rebound duplicates fall from five to
zero, and recall remains 100%. On rows where labels were actually supplied,
precision rises from 94.12% to 100% with 100% recall.

All eight shift classifications and changepoints, both nearby genuine events,
selection-basis disclosures, and deterministic replays remain exact. The full
TSFM-isolated suite passed **2,637 tests with 11 skipped**. Raw artifacts and
the first candidate/scorer correction remain preserved locally;
`docs/astrid-btc-agent-plan.md` remains untracked and excluded.
