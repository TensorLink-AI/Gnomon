Maintain a retail time-series ML workflow and forecast the next14 days, minimizing
RMSLE. task.json identifies the current origin and series. history.csv has730
observed sales rows; future.csv has known promotion/calendar variables, no outcomes.
Zeros do not prove stockouts. Do not make unsupported causal claims.

Use this CHECKPOINT-FIRST workflow through the supplied numerical lab:

1. `python lab.py review` — review your own prior evidence. An empty history is fine.
2. `python lab.py start` — backtest a seasonal baseline on three rolling14-day folds,
   fit the current task forecast and save a valid execution-bound checkpoint early.
3. `python lab.py backtest --config '{"model":"ridge","window":365,"lags":14,"alpha":10}'`
   — compare an ML configuration with that baseline. This is a syntax example, not
   a prescribed choice. Change model, window, lags, alpha or depth to test hypotheses.
4. `python lab.py commit --config '{"model":"ridge","window":365,"lags":14,"alpha":10}'`
   — explicitly choose your backtested configuration. Use your actual choice.
   You can reselect the baseline if the comparison supports it. Continue exploring
   only within the remaining budget. A failed operation keeps the last checkpoint.

`python lab.py status` lists tested configurations, scores, selected checkpoint and
budgets. Inspect concise summaries first; do not print whole experiment logs or CSVs.
`python lab.py --help` gives model bounds. Random Forest example:
{"model":"random_forest","window":180,"lags":28,"depth":6}.
All models use the same three rolling folds; only known covariates and past targets
are used. Each ML configuration is fitted, not retrieved from preset predictions.

To COMPLETE the workflow you must compare at least TWO distinct configurations
(including at least one ML model), then explicitly commit your choice AFTER that
comparison. Merely leaving the initial baseline saved is not full completion.

The lab publishes checkpoint.json atomically and keeps immutable checkpoint records.
This saved selected execution—not your final prose or JSON—is authoritative.
Do not write forecast.json, execution.json or checkpoint.json manually. If several
existing forecasts match your chosen configuration, select one with
`python lab.py commit --execution-id EXACT_ID`; never invent an ID.
Keep a concise decision.json explaining your rationale, evidence, assumptions and
one condition that would invalidate your choice. No lengthy reasoning transcript.

Limits:16 model/API requests TOTAL across bounded continuations,480 seconds,60 numerical forecast attempts including
baseline/backtests/final fits. Every response reports remaining capacity. Backtests
reserve one final fit and are rejected before a three-fold batch would exceed the
budget. Only requests 1–12 are available for exploration. Requests 13–16 are
reserved for explicit selection and correction; new exploratory backtests are
rejected then. The last90 seconds also close exploration. Commit early. If you
terminate without completing the workflow, at most two corrective continuations
may use your SAME remaining requests/time. They do not reset any budget.
Once capacity is low, commit a tested configuration and stop. When already
selected, reusing an existing execution does not consume another numerical fit.

Your project files, raw experiments.jsonl and native Hermes memory persist across
this series. previous_runs.json adds your own matured submissions and their observed outcomes.
At each origin, the same outcomes also score every matching production forecast
you already executed, including alternatives you did not submit. These scores
do not change your past submissions; no additional model is fitted. Treat old
notes as dated evidence. No other arm's information or future outcomes are available.

Use only this project and installed libraries. Do not install software, access other
runs/host files/credentials/scoring targets or fetch outside data. Do not change
task.json, history.csv, future.csv, previous_runs.json, backend.json, lab.py, core.py,
numerical.py, policy.py, maturation.py, checkpoint files, experiment logs or agent-budget.json yourself.
Do not execute business actions. Perform modelling/submission through the lab;
you may write Python to analyze the visible data and evidence.
