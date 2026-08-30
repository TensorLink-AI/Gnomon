# v0.7 loop I002: decision-input integrity

Decision: **promote S1 and advance to S2.**

The external finding reproduced on the exact current head. In the 11 frozen
cases, the baseline produced two internal exceptions and four unsafe
selections: missing scenario payoffs still selected an action, unknown
scenario names were silently weighted as zero, a mixed valid/unknown row
ignored the unknown key, and an exact utility tie selected alphabetically.

The production boundary now validates an exact action-by-scenario matrix
before spending a forecast. Every feasible action must have finite numeric
payoffs for every governed scenario; unknown action or scenario keys are typed
caller errors. Explicitly infeasible actions do not need a utility row. Exact
ties return no action, zero margin, `inconclusive`, and reason `utility_tie`.

On exact commit `5f8ed78`, all 11/11 cases terminated safely and all nine
frozen gates passed:

- internal exceptions: 2 → 0;
- unsafe selections: 4 → 0;
- six malformed matrices returned `INVALID_UTILITIES` with specific problem
  details and an executable `gnomon_decide` argument patch;
- the valid non-tied case retained action `act`, expected utilities 5.2 and
  -0.6, margin 5.8, and supported status;
- the explicitly infeasible incomplete action still allowed the complete
  feasible `wait` action to be selected;
- the no-feasible-action case remained unsupported with no selection.

The actual projected error response was inspected. Its recovery plan retained
the expected actions and scenarios and exposed a ready-to-run
`gnomon_decide` utility matrix. The validator runs before forecast work at the
public tool boundary, so a malformed request no longer spends model or engine
work before being rejected.

Focused independent validation: **150 passed** across the frozen benchmark,
operators, tool surface, error surface, macros, and decision artifacts. Ruff
was not installed in the environment; `git diff --check` passed and the Python
test/import path exercised the changed modules.

Raw baseline and candidate rows remain preserved locally. The aggregate arms
in this release omit per-case payloads while retaining their source digests.

