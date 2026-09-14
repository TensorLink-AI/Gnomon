# Sequential search065: synthetic acquisition and budget preflight passed

The shared search policy now proposes new configurations from current observed
backtests, with optional retrieval of comparable prior executed tuning studies.
Both arms use the same kernel/acquisition/final-selection rules. History changes
the evidence available to that rule; it does not grant additional configurations,
fits or a privileged final forecast. Production outcomes remain separate and
are not used by this particular tuning-reuse mechanism.

Protocol/code frozen at aadab1c before the retained synthetic preflight. Eight
tests pass, including recording/same-origin exclusion, ignored production values,
revision/duplicate rejection, identical cold-start acquisition, independent
two-point posterior arithmetic and preservation of the final-fit budget.

Both synthetic arms completed11 sequential proposals after the common six
starters:17 tested configurations each,58 simulated numerical attempts. They
selected the same final configuration on the toy fixture; no accuracy advantage
is inferred. The traces use synthetic labels without fitting providers.
Actual costs were30 kernel solves (eight during tests,22 during search),0 provider
or API calls,0.6291s wall and0.5910s CPU.

Independent audit3537 checks, no failures. It reconstructed selected studies,
normalization, training masses/hashes, scalar kernel values and all22 posterior
rankings using Cholesky factorization independently of the policy's direct solve.
It also verified current-CV final choice, all simulated attempts and actual
surrogate solve accounting. Audit itself added22 verification solves, separately
reported; these are not provider/model fitting costs.

[Receipt and complete synthetic trace archive](evidence/common-search-065.json).
No new real-data accuracy result. The task runner must next be frozen with source
hashes, cache/logical-attempt accounting and chronological evidence maturation,
then evaluated over all416 original tasks with125 warm-ups. Prior strong050 and
incumbent061 forecasts remain guards; the actual1.2.0 Hermes comparison is still
required before final confirmation. Main/PyPI and final reserves unchanged;
the20% final matched-agent objective remains active and unmet.
