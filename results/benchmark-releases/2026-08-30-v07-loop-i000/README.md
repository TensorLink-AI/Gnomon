# v0.7 loop checkpoint I000

Status: **loop initialized; Q1 protocol frozen; no production change yet.**

The v0.7 branch starts from the green v0.6 evidence commit `7717cd9`. It keeps
v0.6 and PR #83 stable while addressing five benchmark-observed forecast and
agent-boundary gaps under a durable, resumable protocol.

The first trace separated the apparent last-value problem from panel pooling.
Across the retained 80-row TemporalBench Evidence run, only four of 480
submitted paths differed from last-value. They were two T2/T4 views of the
same two underlying rows, both selected by the fold-starved
`seasonal_naive` baseline admission path. Panel pooling made none of the four
departures. All four departures regressed against last-value.

This is diagnostic evidence only. The benchmark rows and labels will not be
used to tune admission. Q1 freezes a new seed and production-selector screen
in `docs/v0.7-forecast-quality-loop.md` before implementation. The immediate
next action is to make that screen faithfully exercise the shipped selector,
run the baseline, inspect admitted wins and losses, and only then choose the
smallest general fix.

No existing artifact was rewritten. All pre-existing untracked files remain
untracked, and `docs/astrid-btc-agent-plan.md` is excluded.

