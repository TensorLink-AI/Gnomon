# Shared configuration catalogue prepared; no new accuracy result

The common hourly adapter now exposes78 configurations: the original three
seasonal/weekly-mean recipes,63 Ridge combinations and12 Random Forest combinations.
Both prospective search arms have the same catalogue and60-attempt limit. The
expansion implements the user's model-iteration task; it is not a stronger
forecasting family or a treatment-only capability.

All six original recipes reproduce038 predictions exactly on synthetic658- and
730-observation requests (twelve matched request pairs). The independent audit
also compares parsed syntax trees: after configuration lookup and label validation,
the entire numerical prediction body and calendar features match038. Canonical
config hashes normalize integer/float parameter representations; implementation
revision is recorded separately. Hourly phase and nominal UTC are enforced.

Protocol/code frozen at4024164 before execution. Four tests pass. Four additional
synthetic boundary configurations return complete finite forecasts. Independent
audit255 checks, no failures. All synthetic costs, including both unit-test and
retained parity executions, are counted:52 computations/28 estimator fits,
5.1469s wall,4.8836s CPU. No source forecast, paid API call or protected-data access.

This is preparation, not an accuracy improvement. No expanded-catalogue model has
been scored on the source data. Before doing that, freeze the common sequential
search policy and precise history-access contract, with equal attempt accounting,
reserved final-fit capacity and no outcomes for unexecuted configurations. Keep
backtest evidence distinct from later production outcomes and retain the previous
strong050 control and061 incumbent as fixed comparison guards.

[Preparation receipt and archive](evidence/common-configuration-064.json).
Prior negative results remain unchanged, including063's selector-only ceiling.
Main/PyPI unchanged; the20% final matched-agent objective remains active/unmet.
